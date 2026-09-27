use super::types::MatmulLayout;
use crate::domain::scalar::Scalar;
use leto::{ArrayViewMut, Result};

#[inline]
pub(super) fn zero_output<T: Scalar>(layout: MatmulLayout, out: &mut ArrayViewMut<'_, T, 2>) {
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
pub(super) fn copy_back_to_out<T: Scalar>(
    src: &ArrayViewMut<'_, T, 2>,
    dst: &mut ArrayViewMut<'_, T, 2>,
) -> Result<()> {
    let shape = dst.shape();
    // `validate_matmul` at the dispatch sites validates the *scratch* output
    // view, not `dst`; `dst` is the caller's view and reaches the raw writes
    // below unproven. Establish its storage bound here.
    let dst_len = dst.data().len();
    dst.layout().validate_storage_len(dst_len)?;
    let src_ptr = src.data().as_ptr();
    let dst_ptr = dst.data_mut().as_mut_ptr();

    for r in 0..shape[0] {
        let src_row_offset = r as isize * shape[1] as isize;
        let dst_row_offset = dst.layout().offset() as isize + r as isize * dst.strides()[0];

        if dst.strides()[1] == 1 {
            // SAFETY: src is C-contiguous, `validate_storage_len` above proved
            // every physical offset `dst`'s layout addresses lies inside its
            // buffer, and this row is unit-stride.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    src_ptr.offset(src_row_offset),
                    dst_ptr.offset(dst_row_offset),
                    shape[1],
                );
            }
        } else {
            // SAFETY: `validate_storage_len` above bounds every physical offset
            // `dst`'s layout addresses; this handles strided copy elements.
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
