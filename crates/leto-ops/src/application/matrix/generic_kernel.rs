use super::types::{MatmulLayout, MATMUL_DEPTH_BATCH_ROWS, MATMUL_DEPTH_BLOCK, MATMUL_ROW_BLOCK};
use crate::domain::scalar::Scalar;
use crate::infrastructure::cache::MatmulTilePolicy;
use leto::{ArrayView, ArrayViewMut};

#[inline]
pub(super) fn can_row_block(layout: MatmulLayout) -> bool {
    layout.rows > 1
        && layout.shared > 0
        && layout.cols > 0
        && layout.rhs_stride_col == 1
        && layout.out_stride_col == 1
}

#[inline]
pub(super) fn row_blocked_matmul_with_policy<T: Scalar>(
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

#[inline(always)]
pub(super) fn accumulate_matmul_row<T: Scalar>(
    lhs_ptr: *const T,
    rhs_ptr: *const T,
    out_ptr: *mut T,
    row: usize,
    layout: MatmulLayout,
) {
    let lhs_row_offset = layout.lhs_offset + row as isize * layout.lhs_stride_row;
    let out_row_offset = layout.out_offset + row as isize * layout.out_stride_row;

    for shared in 0..layout.shared {
        // SAFETY: `validate_matmul` validates the input storage span and this
        // row/shared loop only reads logical LHS indices from that validated span.
        let lhs_value =
            unsafe { *lhs_ptr.offset(lhs_row_offset + shared as isize * layout.lhs_stride_col) };
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

#[inline]
pub(super) fn serial_matmul<T: Scalar>(
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
        accumulate_matmul_row(lhs_ptr, rhs_ptr, out_ptr, row, layout);
    }
}

#[cfg(feature = "parallel")]
pub(super) fn parallel_matmul<T: Scalar>(
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
        accumulate_matmul_row(lhs_ptr, rhs_ptr, out_ptr, row, layout);
    });
}
