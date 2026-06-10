use crate::domain::scalar::Scalar;
use leto::{ArrayView, ArrayViewMut, Layout, LetoError, Result};

#[cfg(feature = "parallel")]
const PARALLEL_ROW_THRESHOLD: usize = 16;

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
    let out_ptr = out.data_mut().as_mut_ptr();

    for row in 0..layout.rows {
        let row_offset = layout.out_offset + row as isize * layout.out_stride_row;
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
/// for row-major output locality, handles strided and transposed inputs, and
/// dispatches row partitions through Moirai when the `parallel` feature is
/// enabled and the row count is large enough.
pub fn matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
) -> Result<()> {
    let layout = validate_matmul(lhs, rhs, out)?;
    zero_output(layout, out);

    #[cfg(feature = "parallel")]
    {
        if layout.rows >= PARALLEL_ROW_THRESHOLD {
            parallel_matmul(lhs, rhs, out, layout);
            return Ok(());
        }
    }

    serial_matmul(lhs, rhs, out, layout);
    Ok(())
}

#[inline]
fn serial_matmul<T: Scalar>(
    lhs: &ArrayView<'_, T, 2>,
    rhs: &ArrayView<'_, T, 2>,
    out: &mut ArrayViewMut<'_, T, 2>,
    layout: MatmulLayout,
) {
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
        // SAFETY: `validate_matmul` validates both storage spans. The caller
        // provides one `out_row_offset` at a time, and each `col` maps to the
        // corresponding output row element.
        unsafe {
            let rhs_row = rhs_ptr.offset(rhs_row_offset);
            let out_row = out_ptr.offset(out_row_offset);
            for col in 0..layout.cols {
                let rhs_value = *rhs_row.add(col);
                let out_ref = out_row.add(col);
                *out_ref = (*out_ref).add(lhs_value.mul(rhs_value));
            }
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
) {
    let lhs_ptr = lhs.data().as_ptr() as usize;
    let rhs_ptr = rhs.data().as_ptr() as usize;
    let out_ptr = out.data_mut().as_mut_ptr() as usize;

    crate::infrastructure::parallel::parallel_for(0, layout.rows, move |row| {
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
