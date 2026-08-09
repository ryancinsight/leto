use crate::domain::scalar::Scalar;
use crate::infrastructure::cache::MatmulTilePolicy;
use leto::{Array, ArrayView, ArrayViewMut, Layout, LetoError, Result};

/// Nonzero density of `lhs` at or below which [`matmul_auto`] routes to the
/// sparse CSR kernel instead of dense [`matmul`].
///
/// Cost model: dense matmul is `Θ(m·s·n)`; the sparse route is one `O(m·s)`
/// compression plus `Θ(nnz·n) = Θ(density·m·s·n)` (`spmm`). Ignoring the
/// sub-dominant compression, sparse beats dense by ≈ `1/density`, discounted by
/// the CSR gather's larger per-flop constant. A conservative `0.1` keeps the
/// sparse path strictly winning (measured ~17× at `0.05`) and never regresses the
/// dense majority case, which pays only the `O(m·s)` density scan.
// The policy selects among these existing const-generic instantiations. The
// 32-row specialization remains the conservative common-shape fallback and is
// also used for the fixed-size alpha panel in the depth-batched path.
const MATMUL_ROW_BLOCK: usize = 32;
const MATMUL_DEPTH_BLOCK: usize = 4;
const MATMUL_DEPTH_BATCH_ROWS: usize = 128;

#[derive(Clone, Copy)]
struct MatmulLayout {
    rows: usize,
    shared: usize,
    cols: usize,
    lhs_stride_row: isize,
    lhs_stride_col: isize,
    rhs_stride_row: isize,
    rhs_stride_col: isize,
    out_stride_row: isize,
    out_stride_col: isize,
    lhs_offset: isize,
    rhs_offset: isize,
    out_offset: isize,
}

#[inline]
fn validate_matmul<T>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &ArrayViewMut<'_, T, 2>,
) -> Result<MatmulLayout> {
    let [rows, lhs_shared] = lhs.shape();
    let [rhs_shared, cols] = rhs.shape();
    let [out_rows, out_cols] = out.shape();

    if lhs_shared != rhs_shared || rows != out_rows || cols != out_cols {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    lhs.layout().validate_storage_len(lhs.data().len())?;
    rhs.layout().validate_storage_len(rhs.data().len())?;
    out.layout().validate_storage_len(out.data().len())?;
    if out.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "matmul output layout must not contain zero-stride aliasing".to_string(),
        });
    }

    Ok(MatmulLayout {
        rows,
        shared: lhs_shared,
        cols,
        lhs_stride_row: lhs.strides()[0],
        lhs_stride_col: lhs.strides()[1],
        rhs_stride_row: rhs.strides()[0],
        rhs_stride_col: rhs.strides()[1],
        out_stride_row: out.strides()[0],
        out_stride_col: out.strides()[1],
        lhs_offset: lhs.offset() as isize,
        rhs_offset: rhs.offset() as isize,
        out_offset: out.offset() as isize,
    })
}

#[inline]
fn zero_output<T: Scalar>(layout: MatmulLayout, out: &mut ArrayViewMut<'_, T, 2>) {
    if layout.out_stride_col == 1 && layout.out_stride_row == layout.cols as isize {
        let start = layout.out_offset as usize;
        let len = layout.rows * layout.cols;
        out.data_mut()[start..start + len].fill(T::ZERO);
        return;
    }

    let out_ptr = out.data_mut().as_mut_ptr();
    for row in 0..layout.rows {
        let row_offset = layout.out_offset + row as isize * layout.out_stride_row;
        if layout.out_stride_col == 1 {
            // SAFETY: `validate_matmul` validated the output storage span and
            // this row is unit-stride over `cols` elements.
            unsafe {
                core::slice::from_raw_parts_mut(out_ptr.offset(row_offset), layout.cols)
                    .fill(T::ZERO);
            }
            continue;
        }

        for col in 0..layout.cols {
            let offset = row_offset + col as isize * layout.out_stride_col;
            // SAFETY: `validate_matmul` validated the output storage span and
            // rejects zero-stride mutable aliasing before this write.
            unsafe {
                *out_ptr.offset(offset) = T::ZERO;
            }
        }
    }
}

/// Perform matrix multiplication `out = lhs * rhs` for 2D views.
///
/// The output is caller-owned. The implementation uses `i-k-j` loop ordering
/// for row-major output locality, row-blocks dense RHS/output rows to reuse
/// each RHS row across a small output-row block, handles strided and
/// transposed inputs, and dispatches row partitions through Moirai when the
/// `parallel` feature is enabled and the row count is large enough.
#[inline]
fn copy_back_to_out<T: Scalar>(
    src: &ArrayViewMut<'_, T, 2>,
    dst: &mut ArrayViewMut<'_, T, 2>,
) -> Result<()> {
    let shape = dst.shape();
    let src_ptr = src.data().as_ptr();
    let dst_ptr = dst.data_mut().as_mut_ptr();

    for r in 0..shape[0] {
        let src_row_offset = r as isize * shape[1] as isize;
        let dst_row_offset = dst.layout().offset as isize + r as isize * dst.strides()[0];

        if dst.strides()[1] == 1 {
            // SAFETY: src is C-contiguous, dst is verified valid by validate_matmul,
            // and this row is unit-stride.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    src_ptr.offset(src_row_offset),
                    dst_ptr.offset(dst_row_offset),
                    shape[1],
                );
            }
        } else {
            // SAFETY: dst is validated, this handles strided copy elements.
            for c in 0..shape[1] {
                unsafe {
                    let val = *src_ptr.offset(src_row_offset + c as isize);
                    let dst_addr = dst_ptr.offset(dst_row_offset + c as isize * dst.strides()[1]);
                    *dst_addr = val;
                }
            }
        }
    }
    Ok(())
}

/// Perform matrix multiplication `out = lhs * rhs` for 2D views.
///
/// The output is caller-owned. This function automatically detects when `lhs` is
/// sparse (below `SPARSE_DENSITY_THRESHOLD`) and `out` is contiguous, routing to the
/// sparse `spmm` kernel. Otherwise, it executes the optimized dense `matmul` logic,
/// which uses an `i-k-j` loop order for locality, row-blocks dense rows to reuse
/// RHS values, and dispatches parallel tasks when `parallel` is enabled.
#[cfg(feature = "parallel")]
#[inline]
fn is_parallel_beneficial(layout: MatmulLayout) -> bool {
    layout.rows * layout.cols * layout.shared >= 262_144 && layout.rows >= 64
}

fn serial_dot_matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
    accumulate: bool,
) {
    let [m, k] = lhs.shape();
    let [_, n] = rhs.shape();
    let lhs_offset = lhs.offset();
    let rhs_offset = rhs.offset();
    let out_offset = out.offset();
    let lhs_data = lhs.data();
    let rhs_data = rhs.data();
    let out_data = out.data_mut();

    for i in 0..m {
        let lhs_row = &lhs_data[lhs_offset + i * k..lhs_offset + i * k + k];
        for j in 0..n {
            let rhs_col = &rhs_data[rhs_offset + j * k..rhs_offset + j * k + k];
            let val = T::dot_slice(lhs_row, rhs_col);
            let out_idx = out_offset + i * n + j;
            if accumulate {
                out_data[out_idx] = out_data[out_idx].add(val);
            } else {
                out_data[out_idx] = val;
            }
        }
    }
}

/// Number of output rows processed per parallel task.
///
/// Each task handles `PARALLEL_ROW_BLOCK` consecutive rows of the output
/// matrix.  Blocking reduces the task-dispatch overhead by `PARALLEL_ROW_BLOCK×`
/// vs one-task-per-row, and ensures each task writes to a contiguous output
/// region of at least `PARALLEL_ROW_BLOCK * n * size_of::<T>()` bytes — large
/// enough to avoid false sharing for any practical `n`.
#[cfg(feature = "parallel")]
const PARALLEL_ROW_BLOCK: usize = 4;

#[cfg(feature = "parallel")]
fn parallel_dot_matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
    accumulate: bool,
) {
    let [m, k] = lhs.shape();
    let [_, n] = rhs.shape();
    let lhs_offset = lhs.offset();
    let rhs_offset = rhs.offset();
    let out_offset = out.offset();
    let lhs_ptr = lhs.data().as_ptr() as usize;
    let rhs_ptr = rhs.data().as_ptr() as usize;
    let out_ptr = out.data_mut().as_mut_ptr() as usize;

    // Dispatch in row blocks to amortise per-task scheduling overhead.
    let n_blocks = m.div_ceil(PARALLEL_ROW_BLOCK);
    moirai::for_each_index_with::<moirai::AdaptiveWithThreshold<16>, _>(n_blocks, move |block| {
        let lhs_ptr = lhs_ptr as *const T;
        let rhs_ptr = rhs_ptr as *const T;
        let out_ptr = out_ptr as *mut T;
        let i_start = block * PARALLEL_ROW_BLOCK;
        let i_end = (i_start + PARALLEL_ROW_BLOCK).min(m);
        for i in i_start..i_end {
            unsafe {
                let lhs_row = core::slice::from_raw_parts(lhs_ptr.add(lhs_offset + i * k), k);
                for j in 0..n {
                    let rhs_col = core::slice::from_raw_parts(rhs_ptr.add(rhs_offset + j * k), k);
                    let val = T::dot_slice(lhs_row, rhs_col);
                    let out_addr = out_ptr.add(out_offset + i * n + j);
                    if accumulate {
                        *out_addr = (*out_addr).add(val);
                    } else {
                        *out_addr = val;
                    }
                }
            }
        }
    });
}

fn serial_outer_matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
    accumulate: bool,
) {
    let [m, k] = lhs.shape();
    let [_, n] = rhs.shape();

    if !accumulate {
        zero_output(
            validate_matmul(lhs, rhs, out).expect("route_matmul validated dimensions"),
            out,
        );
    }

    let lhs_offset = lhs.offset();
    let rhs_offset = rhs.offset();
    let out_offset = out.offset();
    let lhs_data = lhs.data();
    let rhs_data = rhs.data();
    let out_data = out.data_mut();

    for i in 0..m {
        unsafe {
            let out_ptr = out_data.as_mut_ptr().add(out_offset + i * n);
            let out_row = core::slice::from_raw_parts_mut(out_ptr, n);
            for kk in 0..k {
                let alpha = *lhs_data.get_unchecked(lhs_offset + kk * m + i);
                if alpha == T::ZERO {
                    continue;
                }
                let rhs_row = &rhs_data[rhs_offset + kk * n..rhs_offset + kk * n + n];
                T::axpy_slice(alpha, rhs_row, out_row);
            }
        }
    }
}

#[cfg(feature = "parallel")]
fn parallel_outer_matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
    accumulate: bool,
) {
    let [m, k] = lhs.shape();
    let [_, n] = rhs.shape();

    if !accumulate {
        zero_output(
            validate_matmul(lhs, rhs, out).expect("route_matmul validated dimensions"),
            out,
        );
    }

    let lhs_offset = lhs.offset();
    let rhs_offset = rhs.offset();
    let out_offset = out.offset();
    let lhs_ptr = lhs.data().as_ptr() as usize;
    let rhs_ptr = rhs.data().as_ptr() as usize;
    let out_ptr = out.data_mut().as_mut_ptr() as usize;

    let n_blocks = m.div_ceil(PARALLEL_ROW_BLOCK);
    moirai::for_each_index_with::<moirai::AdaptiveWithThreshold<16>, _>(n_blocks, move |block| {
        let lhs_ptr = lhs_ptr as *const T;
        let rhs_ptr = rhs_ptr as *const T;
        let out_ptr = out_ptr as *mut T;
        let i_start = block * PARALLEL_ROW_BLOCK;
        let i_end = (i_start + PARALLEL_ROW_BLOCK).min(m);
        for i in i_start..i_end {
            unsafe {
                let out_row = core::slice::from_raw_parts_mut(out_ptr.add(out_offset + i * n), n);
                for kk in 0..k {
                    let alpha = *lhs_ptr.add(lhs_offset + kk * m + i);
                    if alpha == T::ZERO {
                        continue;
                    }
                    let rhs_row = core::slice::from_raw_parts(rhs_ptr.add(rhs_offset + kk * n), n);
                    T::axpy_slice(alpha, rhs_row, out_row);
                }
            }
        }
    });
}

fn route_matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
    accumulate: bool,
    tile_policy: MatmulTilePolicy,
) -> Result<()> {
    let layout = validate_matmul(lhs, rhs, out)?;
    #[cfg(not(feature = "parallel"))]
    let _ = layout;

    // Fast-path selection uses offset-independent dense predicates: the dot/cc/
    // outer kernels address every operand through its layout's own `offset`, so
    // a batched or sliced sub-view that is dense-but-offset (e.g. batch `b` of a
    // C-contiguous 3-D output, offset `b·m·n`) is served in place with no
    // operand copy and no scratch allocation. Pinning `offset == 0` here
    // (`is_c_contiguous`) was forcing those views down the allocating fallback.
    if lhs.is_c_dense() && rhs.is_f_dense() && out.is_c_dense() {
        #[cfg(feature = "parallel")]
        {
            if is_parallel_beneficial(layout) {
                parallel_dot_matmul(lhs, rhs, out, accumulate);
                return Ok(());
            }
        }
        serial_dot_matmul(lhs, rhs, out, accumulate);
        return Ok(());
    }

    if lhs.is_f_dense() && rhs.is_c_dense() && out.is_c_dense() {
        #[cfg(feature = "parallel")]
        {
            if is_parallel_beneficial(layout) {
                parallel_outer_matmul(lhs, rhs, out, accumulate);
                return Ok(());
            }
        }
        serial_outer_matmul(lhs, rhs, out, accumulate);
        return Ok(());
    }

    // Fallback: copy only genuinely non-dense operands to contiguous. A
    // dense-but-offset operand is kept in place (the generic kernel addresses it
    // through its layout offset), so only strided/broadcast operands pay a copy.
    let lhs_contig;
    let lhs_view = if lhs.is_c_dense() {
        lhs.reborrow()
    } else {
        lhs_contig = lhs.to_contiguous();
        lhs_contig.view()
    };

    let rhs_contig;
    let rhs_view = if rhs.is_c_dense() {
        rhs.reborrow()
    } else {
        rhs_contig = rhs.to_contiguous();
        rhs_contig.view()
    };

    let layout = validate_matmul(&lhs_view, &rhs_view, out)?;
    if !accumulate {
        zero_output(layout, out);
    }

    #[cfg(feature = "parallel")]
    {
        if is_parallel_beneficial(layout) {
            parallel_matmul(&lhs_view, &rhs_view, out, layout, tile_policy);
            return Ok(());
        }
    }
    serial_matmul(&lhs_view, &rhs_view, out, layout, tile_policy);
    Ok(())
}

/// Perform matrix multiplication `out = lhs * rhs` for 2D views.
pub fn matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
) -> Result<()> {
    matmul_with_tile_policy(
        lhs,
        rhs,
        out,
        MatmulTilePolicy::fixed(MATMUL_ROW_BLOCK)
            .expect("the measured default row block is supported"),
    )
}

/// Perform matrix multiplication with an explicit bounded row-tile policy.
///
/// This is primarily useful for controlled provider benchmarks and callers
/// that already own a topology policy. Normal callers should use [`matmul`],
/// which retains the measured fixed 32-row production policy.
pub fn matmul_with_tile_policy<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
    tile_policy: MatmulTilePolicy,
) -> Result<()> {
    // Dense-but-offset outputs (a batched/sliced sub-view) route in place: the
    // kernels write through the layout offset, so only a genuinely strided
    // output needs the scratch + copy-back.
    if out.is_c_dense() {
        let mut out_view = out.reborrow();
        route_matmul(lhs, rhs, &mut out_view, false, tile_policy)
    } else if out.is_f_dense() {
        let lhs_t = lhs.transpose([1, 0])?;
        let rhs_t = rhs.transpose([1, 0])?;
        let mut out_t = out.reborrow().transpose_mut([1, 0])?;
        route_matmul(&rhs_t, &lhs_t, &mut out_t, false, tile_policy)
    } else {
        let mut out_contig = Array::from_elem(out.shape(), T::ZERO);
        let mut out_view = out_contig.view_mut();
        route_matmul(lhs, rhs, &mut out_view, false, tile_policy)?;
        copy_back_to_out(&out_view, out)?;
        Ok(())
    }
}

/// Perform accumulating matrix multiplication `out += lhs * rhs` for 2D views.
pub fn matmul_accumulate<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
) -> Result<()> {
    // Dense-but-offset outputs route in place (kernels honor the layout offset);
    // only a genuinely strided output needs the scratch + copy-back.
    if out.is_c_dense() {
        let mut out_view = out.reborrow();
        route_matmul(
            lhs,
            rhs,
            &mut out_view,
            true,
            MatmulTilePolicy::fixed(MATMUL_ROW_BLOCK)
                .expect("the measured default row block is supported"),
        )
    } else if out.is_f_dense() {
        let lhs_t = lhs.transpose([1, 0])?;
        let rhs_t = rhs.transpose([1, 0])?;
        let mut out_t = out.reborrow().transpose_mut([1, 0])?;
        route_matmul(
            &rhs_t,
            &lhs_t,
            &mut out_t,
            true,
            MatmulTilePolicy::fixed(MATMUL_ROW_BLOCK)
                .expect("the measured default row block is supported"),
        )
    } else {
        let mut out_contig = out.to_contiguous();
        let mut out_view = out_contig.view_mut();
        route_matmul(
            lhs,
            rhs,
            &mut out_view,
            true,
            MatmulTilePolicy::fixed(MATMUL_ROW_BLOCK)
                .expect("the measured default row block is supported"),
        )?;
        copy_back_to_out(&out_view, out)?;
        Ok(())
    }
}

#[inline]
fn serial_matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
    layout: MatmulLayout,
    tile_policy: MatmulTilePolicy,
) {
    if can_row_block(layout) {
        row_blocked_matmul_with_policy::<T>(
            lhs.data().as_ptr(),
            rhs.data().as_ptr(),
            out.data_mut().as_mut_ptr(),
            0,
            layout.rows,
            layout,
            tile_policy,
        );
        return;
    }

    let lhs_ptr = lhs.data().as_ptr();
    let rhs_ptr = rhs.data().as_ptr();
    let out_ptr = out.data_mut().as_mut_ptr();

    for row in 0..layout.rows {
        let lhs_row_offset = layout.lhs_offset + row as isize * layout.lhs_stride_row;
        let out_row_offset = layout.out_offset + row as isize * layout.out_stride_row;

        for shared in 0..layout.shared {
            // SAFETY: `validate_matmul` validates the input storage span for
            // every logical `lhs` index used by this loop nest.
            let lhs_value = unsafe {
                *lhs_ptr.offset(lhs_row_offset + shared as isize * layout.lhs_stride_col)
            };
            if lhs_value == T::ZERO {
                continue;
            }

            let rhs_row_offset = layout.rhs_offset + shared as isize * layout.rhs_stride_row;
            multiply_row(
                lhs_value,
                rhs_ptr,
                out_ptr,
                rhs_row_offset,
                out_row_offset,
                layout,
            );
        }
    }
}

#[inline]
fn can_row_block(layout: MatmulLayout) -> bool {
    layout.rows > 1
        && layout.shared > 0
        && layout.cols > 0
        && layout.rhs_stride_col == 1
        && layout.out_stride_col == 1
}

#[inline]
fn row_blocked_matmul_with_policy<T: Scalar>(
    lhs_ptr: *const T,
    rhs_ptr: *const T,
    out_ptr: *mut T,
    start_row: usize,
    end_row: usize,
    layout: MatmulLayout,
    tile_policy: MatmulTilePolicy,
) {
    let row_block = tile_policy.row_block();

    match row_block {
        1 => row_blocked_matmul::<T, 1>(lhs_ptr, rhs_ptr, out_ptr, start_row, end_row, layout),
        2 => row_blocked_matmul::<T, 2>(lhs_ptr, rhs_ptr, out_ptr, start_row, end_row, layout),
        4 => row_blocked_matmul::<T, 4>(lhs_ptr, rhs_ptr, out_ptr, start_row, end_row, layout),
        8 => row_blocked_matmul::<T, 8>(lhs_ptr, rhs_ptr, out_ptr, start_row, end_row, layout),
        16 => row_blocked_matmul::<T, 16>(lhs_ptr, rhs_ptr, out_ptr, start_row, end_row, layout),
        32 => row_blocked_matmul::<T, 32>(lhs_ptr, rhs_ptr, out_ptr, start_row, end_row, layout),
        _ => unreachable!("matmul tile policy must return a power-of-two block <= 32"),
    }
}

#[inline]
fn row_blocked_matmul<T: Scalar, const ROW_BLOCK: usize>(
    lhs_ptr: *const T,
    rhs_ptr: *const T,
    out_ptr: *mut T,
    start_row: usize,
    end_row: usize,
    layout: MatmulLayout,
) {
    debug_assert!(ROW_BLOCK > 0);
    let fused_out_stride = if layout.out_stride_row >= layout.cols as isize {
        Some(layout.out_stride_row as usize)
    } else {
        None
    };

    for row_block_start in (start_row..end_row).step_by(ROW_BLOCK) {
        let row_block_end = (row_block_start + ROW_BLOCK).min(end_row);
        let block_rows = row_block_end - row_block_start;

        if layout.lhs_stride_col == 1
            && layout.lhs_stride_row == layout.shared as isize
            && layout.rhs_stride_col == 1
            && layout.rhs_stride_row == layout.cols as isize
            && layout.out_stride_col == 1
            && layout.out_stride_row == layout.cols as isize
        {
            let a_offset = layout.lhs_offset + (row_block_start * layout.shared) as isize;
            let b_offset = layout.rhs_offset;
            let c_offset = layout.out_offset + (row_block_start * layout.cols) as isize;
            // SAFETY: `validate_matmul` validates all storage spans, layout.cols and
            // layout.shared represent the valid sizes, and this block processes
            // non-overlapping row segments of C-contiguous matrices.
            unsafe {
                let a_slice = core::slice::from_raw_parts(
                    lhs_ptr.offset(a_offset),
                    block_rows * layout.shared,
                );
                let b_slice = core::slice::from_raw_parts(
                    rhs_ptr.offset(b_offset),
                    layout.shared * layout.cols,
                );
                let c_slice = core::slice::from_raw_parts_mut(
                    out_ptr.offset(c_offset),
                    block_rows * layout.cols,
                );
                T::tiled_gemm(
                    a_slice,
                    b_slice,
                    c_slice,
                    block_rows,
                    layout.cols,
                    layout.shared,
                );
            }
            continue;
        }

        if layout.rows >= MATMUL_DEPTH_BATCH_ROWS {
            if let Some(out_stride_row) = fused_out_stride {
                if layout.rhs_stride_row == layout.cols as isize {
                    let out_block_offset =
                        layout.out_offset + row_block_start as isize * layout.out_stride_row;
                    let out_block_len = (block_rows - 1) * out_stride_row + layout.cols;
                    // SAFETY: `validate_matmul` validates the full output storage
                    // span, row blocking is enabled only for unit-stride columns,
                    // and this fused path requires a positive non-overlapping row
                    // stride of at least `cols`.
                    let out_block = unsafe {
                        core::slice::from_raw_parts_mut(
                            out_ptr.offset(out_block_offset),
                            out_block_len,
                        )
                    };

                    for shared_start in (0..layout.shared).step_by(MATMUL_DEPTH_BLOCK) {
                        let depth = (layout.shared - shared_start).min(MATMUL_DEPTH_BLOCK);
                        let mut alphas = [T::ZERO; MATMUL_ROW_BLOCK * MATMUL_DEPTH_BLOCK];
                        for shared_offset in 0..depth {
                            let shared = shared_start + shared_offset;
                            let alpha_start = shared_offset * block_rows;
                            for (block_row, alpha) in alphas[alpha_start..alpha_start + block_rows]
                                .iter_mut()
                                .enumerate()
                            {
                                let row = row_block_start + block_row;
                                let lhs_row_offset =
                                    layout.lhs_offset + row as isize * layout.lhs_stride_row;
                                // SAFETY: `validate_matmul` validates every logical LHS
                                // index used by this row/shared loop nest.
                                *alpha = unsafe {
                                    *lhs_ptr.offset(
                                        lhs_row_offset + shared as isize * layout.lhs_stride_col,
                                    )
                                };
                            }
                        }

                        let rhs_panel_offset =
                            layout.rhs_offset + shared_start as isize * layout.rhs_stride_row;
                        // SAFETY: `validate_matmul` validates the RHS storage span,
                        // row blocking is enabled only for unit-stride RHS rows,
                        // and this batched path requires physically adjacent RHS
                        // rows (`rhs_stride_row == cols`).
                        let rhs_panel = unsafe {
                            core::slice::from_raw_parts(
                                rhs_ptr.offset(rhs_panel_offset),
                                depth * layout.cols,
                            )
                        };
                        T::axpy_rows_batch(
                            &alphas[..depth * block_rows],
                            rhs_panel,
                            out_block,
                            out_stride_row,
                            block_rows,
                            depth,
                            layout.cols,
                        );
                    }
                    continue;
                }
            }
        }

        for shared in 0..layout.shared {
            let rhs_row_offset = layout.rhs_offset + shared as isize * layout.rhs_stride_row;
            // SAFETY: `validate_matmul` validates the RHS storage span, and
            // row blocking is enabled only for unit-stride RHS rows.
            let rhs_row =
                unsafe { core::slice::from_raw_parts(rhs_ptr.offset(rhs_row_offset), layout.cols) };

            if let Some(out_stride_row) = fused_out_stride {
                let mut alphas = [T::ZERO; ROW_BLOCK];
                for (block_row, alpha) in alphas.iter_mut().take(block_rows).enumerate() {
                    let row = row_block_start + block_row;
                    let lhs_row_offset = layout.lhs_offset + row as isize * layout.lhs_stride_row;
                    // SAFETY: `validate_matmul` validates every logical LHS
                    // index used by this row/shared loop nest.
                    *alpha = unsafe {
                        *lhs_ptr.offset(lhs_row_offset + shared as isize * layout.lhs_stride_col)
                    };
                }

                let out_block_offset =
                    layout.out_offset + row_block_start as isize * layout.out_stride_row;
                let out_block_len = (block_rows - 1) * out_stride_row + layout.cols;
                // SAFETY: `validate_matmul` validates the full output storage
                // span, row blocking is enabled only for unit-stride columns,
                // and this fused path requires a positive non-overlapping row
                // stride of at least `cols`.
                let out_block = unsafe {
                    core::slice::from_raw_parts_mut(out_ptr.offset(out_block_offset), out_block_len)
                };
                T::axpy_rows(
                    &alphas[..block_rows],
                    rhs_row,
                    out_block,
                    out_stride_row,
                    block_rows,
                    layout.cols,
                );
                continue;
            }

            for row in row_block_start..row_block_end {
                let lhs_row_offset = layout.lhs_offset + row as isize * layout.lhs_stride_row;
                // SAFETY: `validate_matmul` validates every logical LHS index
                // used by this row/shared loop nest.
                let lhs_value = unsafe {
                    *lhs_ptr.offset(lhs_row_offset + shared as isize * layout.lhs_stride_col)
                };
                if lhs_value == T::ZERO {
                    continue;
                }

                let out_row_offset = layout.out_offset + row as isize * layout.out_stride_row;
                // SAFETY: `validate_matmul` validates the output storage span,
                // rejects zero-stride output aliasing, and each row in this
                // block is updated through a distinct unit-stride row slice.
                let out_row = unsafe {
                    core::slice::from_raw_parts_mut(out_ptr.offset(out_row_offset), layout.cols)
                };
                T::axpy_slice(lhs_value, rhs_row, out_row);
            }
        }
    }
}

#[inline(always)]
fn multiply_row<T: Scalar>(
    lhs_value: T,
    rhs_ptr: *const T,
    out_ptr: *mut T,
    rhs_row_offset: isize,
    out_row_offset: isize,
    layout: MatmulLayout,
) {
    if layout.rhs_stride_col == 1 && layout.out_stride_col == 1 {
        // SAFETY: `validate_matmul` validates both storage spans, both rows
        // are unit-stride over `cols` elements, and `rhs` (input view) and
        // `out` (exclusive `&mut` output view) never alias, so the two raw
        // rows are disjoint slices.
        unsafe {
            let rhs_row = core::slice::from_raw_parts(rhs_ptr.offset(rhs_row_offset), layout.cols);
            let out_row =
                core::slice::from_raw_parts_mut(out_ptr.offset(out_row_offset), layout.cols);
            T::axpy_slice(lhs_value, rhs_row, out_row);
        }
    } else {
        // SAFETY: `validate_matmul` validates all physical offsets spanned by
        // the strided input and output layouts.
        unsafe {
            for col in 0..layout.cols {
                let rhs_value =
                    *rhs_ptr.offset(rhs_row_offset + col as isize * layout.rhs_stride_col);
                let out_ref = out_ptr.offset(out_row_offset + col as isize * layout.out_stride_col);
                *out_ref = (*out_ref).add(lhs_value.mul(rhs_value));
            }
        }
    }
}

#[cfg(feature = "parallel")]
fn parallel_matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
    layout: MatmulLayout,
    tile_policy: MatmulTilePolicy,
) {
    if can_row_block(layout) {
        let lhs_ptr = lhs.data().as_ptr() as usize;
        let rhs_ptr = rhs.data().as_ptr() as usize;
        let out_ptr = out.data_mut().as_mut_ptr() as usize;
        let block_count = layout.rows.div_ceil(MATMUL_ROW_BLOCK);

        moirai::for_each_index_with::<moirai::AdaptiveWithThreshold<2>, _>(
            block_count,
            move |block| {
                let lhs_ptr = lhs_ptr as *const T;
                let rhs_ptr = rhs_ptr as *const T;
                let out_ptr = out_ptr as *mut T;
                let start_row = block * MATMUL_ROW_BLOCK;
                let end_row = (start_row + MATMUL_ROW_BLOCK).min(layout.rows);
                row_blocked_matmul_with_policy::<T>(
                    lhs_ptr,
                    rhs_ptr,
                    out_ptr,
                    start_row,
                    end_row,
                    layout,
                    tile_policy,
                );
            },
        );
        return;
    }

    let lhs_ptr = lhs.data().as_ptr() as usize;
    let rhs_ptr = rhs.data().as_ptr() as usize;
    let out_ptr = out.data_mut().as_mut_ptr() as usize;

    moirai::for_each_index_with::<moirai::AdaptiveWithThreshold<16>, _>(layout.rows, move |row| {
        let lhs_ptr = lhs_ptr as *const T;
        let rhs_ptr = rhs_ptr as *const T;
        let out_ptr = out_ptr as *mut T;
        let lhs_row_offset = layout.lhs_offset + row as isize * layout.lhs_stride_row;
        let out_row_offset = layout.out_offset + row as isize * layout.out_stride_row;

        for shared in 0..layout.shared {
            // SAFETY: `validate_matmul` validates the input storage span and
            // each worker owns one logical output row.
            let lhs_value = unsafe {
                *lhs_ptr.offset(lhs_row_offset + shared as isize * layout.lhs_stride_col)
            };
            if lhs_value == T::ZERO {
                continue;
            }

            let rhs_row_offset = layout.rhs_offset + shared as isize * layout.rhs_stride_row;
            multiply_row(
                lhs_value,
                rhs_ptr,
                out_ptr,
                rhs_row_offset,
                out_row_offset,
                layout,
            );
        }
    });
}

/// Perform batched matrix multiplication `out[i] = lhs[i] * rhs[i]` for rank-3
/// views shaped `[B, M, K] x [B, K, N] -> [B, M, N]`.
///
/// The batch dimension of either input may be `1`, in which case that operand
/// is broadcast across all `B` batches at zero stride (no materialization).
/// Each batch slice is dispatched to the rank-2 [`matmul`] kernel, so there is
/// one authoritative contraction implementation; this function only resolves
/// per-batch 2D layouts.
pub fn batched_matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 3>,
    rhs: &ArrayView<'_, T, 3>,
    out: &mut ArrayViewMut<'_, T, 3>,
) -> Result<()> {
    let [lhs_batch, m, lhs_k] = lhs.shape();
    let [rhs_batch, rhs_k, n] = rhs.shape();
    let [out_batch, out_m, out_n] = out.shape();

    let batch = out_batch;
    let lhs_batches_ok = lhs_batch == batch || lhs_batch == 1;
    let rhs_batches_ok = rhs_batch == batch || rhs_batch == 1;
    if !lhs_batches_ok || !rhs_batches_ok || lhs_k != rhs_k || m != out_m || n != out_n {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    lhs.layout().validate_storage_len(lhs.data().len())?;
    rhs.layout().validate_storage_len(rhs.data().len())?;
    out.layout().validate_storage_len(out.data().len())?;

    let lhs_batch_stride = if lhs_batch == 1 { 0 } else { lhs.strides()[0] };
    let rhs_batch_stride = if rhs_batch == 1 { 0 } else { rhs.strides()[0] };
    let out_batch_stride = out.strides()[0];

    let lhs_mat = |b: usize| {
        Layout::new(
            [m, lhs_k],
            [lhs.strides()[1], lhs.strides()[2]],
            (lhs.offset() as isize + b as isize * lhs_batch_stride) as usize,
        )
    };
    let rhs_mat = |b: usize| {
        Layout::new(
            [rhs_k, n],
            [rhs.strides()[1], rhs.strides()[2]],
            (rhs.offset() as isize + b as isize * rhs_batch_stride) as usize,
        )
    };
    let out_offset = out.offset() as isize;
    let out_strides = [out.strides()[1], out.strides()[2]];

    #[cfg(feature = "parallel")]
    {
        // The parallel path hands each task a `&mut` over only its own batch's
        // physical footprint, so concurrent tasks never hold overlapping `&mut`
        // slices. (Forming N `&mut` over the same full buffer is UB under
        // Stacked/Tree Borrows even when the writes are physically disjoint.)
        // This requires the per-batch footprints to be physically disjoint;
        // they are unless the batch stride is smaller than one matrix's physical
        // span (an interleaved-batch output view), in which case we fall through
        // to the sequential loop below, which reborrows one batch at a time and
        // is unconditionally sound.
        let out_span = 1
            + out_m.saturating_sub(1) * out_strides[0].unsigned_abs()
            + out_n.saturating_sub(1) * out_strides[1].unsigned_abs();
        let batches_disjoint = out_batch_stride.unsigned_abs() >= out_span;
        // Non-empty guard: an empty output matrix has no work and would make the
        // per-batch `min_max_offsets` on a zero-shape layout degenerate, so let
        // the sequential loop handle it.
        if batch > 1 && batches_disjoint && out_m > 0 && out_n > 0 {
            // Hot-path early-out uses a relaxed atomic flag rather than locking a
            // mutex on every batch index; the mutex is acquired only on the
            // (pre-validated, effectively unreachable) error path to record the
            // first failure. The `for_each_index_with` join supplies the
            // happens-before barrier for the recorded error.
            let had_error = std::sync::Arc::new(core::sync::atomic::AtomicBool::new(false));
            let error_slot = std::sync::Arc::new(std::sync::Mutex::new(None::<LetoError>));
            let had_error_w = had_error.clone();
            let error_slot_w = error_slot.clone();
            let lhs_ptr = lhs.data().as_ptr() as usize;
            let rhs_ptr = rhs.data().as_ptr() as usize;
            let out_ptr = out.data_mut().as_mut_ptr() as usize;
            let lhs_len = lhs.data().len();
            let rhs_len = rhs.data().len();

            let lhs_offset = lhs.offset() as isize;
            let rhs_offset = rhs.offset() as isize;
            let lhs_strides = [lhs.strides()[1], lhs.strides()[2]];
            let rhs_strides = [rhs.strides()[1], rhs.strides()[2]];

            moirai::for_each_index_with::<moirai::Adaptive, _>(batch, move |b| {
                if had_error_w.load(core::sync::atomic::Ordering::Relaxed) {
                    return;
                }

                let lhs_ptr = lhs_ptr as *const T;
                let rhs_ptr = rhs_ptr as *const T;
                let out_ptr = out_ptr as *mut T;

                let lhs_layout = Layout::new(
                    [m, lhs_k],
                    lhs_strides,
                    (lhs_offset + b as isize * lhs_batch_stride) as usize,
                );
                let rhs_layout = Layout::new(
                    [rhs_k, n],
                    rhs_strides,
                    (rhs_offset + b as isize * rhs_batch_stride) as usize,
                );

                let lhs_view = unsafe {
                    ArrayView::new(lhs_layout, core::slice::from_raw_parts(lhs_ptr, lhs_len))
                };
                let rhs_view = unsafe {
                    ArrayView::new(rhs_layout, core::slice::from_raw_parts(rhs_ptr, rhs_len))
                };
                let abs_offset = (out_offset + b as isize * out_batch_stride) as usize;
                let out_layout = Layout::new([out_m, out_n], out_strides, abs_offset);
                // Borrow only this batch's physical span `[lo, hi]` and rebase
                // the offset into it. `batches_disjoint` (checked above)
                // guarantees these per-batch slices never overlap across tasks,
                // so no two concurrent `&mut` alias.
                let (lo, hi) = out_layout.min_max_offsets();
                let mut out_view = unsafe {
                    ArrayViewMut::new(
                        Layout::new([out_m, out_n], out_strides, abs_offset - lo),
                        core::slice::from_raw_parts_mut(out_ptr.add(lo), hi - lo + 1),
                    )
                };

                if let Err(e) = matmul(&lhs_view, &rhs_view, &mut out_view) {
                    let mut slot = error_slot_w.lock().expect("mutex not poisoned");
                    if slot.is_none() {
                        *slot = Some(e);
                    }
                    had_error_w.store(true, core::sync::atomic::Ordering::Relaxed);
                }
            });

            if let Some(e) = error_slot.lock().expect("mutex not poisoned").take() {
                return Err(e);
            }
            return Ok(());
        }
    }

    for b in 0..batch {
        let lhs_view = ArrayView::new(lhs_mat(b), lhs.data());
        let rhs_view = ArrayView::new(rhs_mat(b), rhs.data());
        let out_layout = Layout::new(
            [out_m, out_n],
            out_strides,
            (out_offset + b as isize * out_batch_stride) as usize,
        );
        let mut out_view = ArrayViewMut::new(out_layout, out.data_mut());
        matmul(&lhs_view, &rhs_view, &mut out_view)?;
    }

    Ok(())
}
