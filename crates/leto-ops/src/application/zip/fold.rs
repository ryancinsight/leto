use super::traversal::for_each_row_major_indexed;
use crate::application::index::RowMajorTraversal;
use leto::{ArrayView, LetoError, Result};

/// Fold two read-only views into one accumulator.
///
/// This is the reduction analogue of [`crate::application::zip::zip_mut_with`].
/// Both views must have the same logical shape, and every logical pair is
/// visited exactly once. Traversal order is unspecified for strided layouts,
/// so callers should use associative or order-insensitive reductions when
/// bitwise replay is required across layouts.
pub fn zip_fold<T, U, Acc, F, const N: usize>(
    lhs: &ArrayView<'_, T, N>,
    rhs: &ArrayView<'_, U, N>,
    init: Acc,
    mut f: F,
) -> Result<Acc>
where
    F: FnMut(Acc, &T, &U) -> Acc,
{
    if lhs.shape() != rhs.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    lhs.layout().validate_storage_len(lhs.data().len())?;
    rhs.layout().validate_storage_len(rhs.data().len())?;

    if let (Some(lhs_slice), Some(rhs_slice)) = (lhs.as_slice(), rhs.as_slice()) {
        let mut acc = init;
        for (left, right) in lhs_slice.iter().zip(rhs_slice.iter()) {
            acc = f(acc, left, right);
        }
        return Ok(acc);
    }

    let size = lhs.layout().checked_size()?;
    let shape = lhs.shape();
    let lhs_layout = lhs.layout();
    let rhs_layout = rhs.layout();
    let lhs_data = lhs.data();
    let rhs_data = rhs.data();

    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(init);
    };
    let lhs_step = traversal.last_axis_stride(lhs_layout);
    let rhs_step = traversal.last_axis_stride(rhs_layout);
    let mut acc = init;
    for row in 0..traversal.rows() {
        let base = traversal.base_index(row);
        let mut lhs_offset = lhs_layout.offset_of(base)? as isize;
        let mut rhs_offset = rhs_layout.offset_of(base)? as isize;
        for _ in 0..traversal.inner() {
            acc = f(
                acc,
                &lhs_data[lhs_offset as usize],
                &rhs_data[rhs_offset as usize],
            );
            lhs_offset += lhs_step;
            rhs_offset += rhs_step;
        }
    }

    Ok(acc)
}

/// Fold one read-only view with the logical row-major index.
///
/// This is the indexed analogue of [`zip_fold`]. Every logical element is
/// visited exactly once, and the closure receives the logical index before the
/// read-only value. The traversal follows logical row-major order independent
/// of the backing storage layout.
pub fn indexed_fold<T, Acc, F, const N: usize>(
    view: &ArrayView<'_, T, N>,
    init: Acc,
    mut f: F,
) -> Result<Acc>
where
    F: FnMut(Acc, [usize; N], &T) -> Acc,
{
    view.layout().validate_storage_len(view.data().len())?;

    let size = view.layout().checked_size()?;
    let shape = view.shape();
    let layout = view.layout();
    let data = view.data();

    let Some(traversal) = RowMajorTraversal::new(size, shape) else {
        return Ok(init);
    };
    let step = traversal.last_axis_stride(layout);
    let mut acc = Some(init);
    for_each_row_major_indexed(
        traversal,
        |index| Ok(layout.offset_of(index)? as isize),
        |index, offset| {
            let current = acc
                .take()
                .expect("invariant: indexed fold accumulator is always present");
            acc = Some(f(current, index, &data[*offset as usize]));
            *offset += step;
        },
    )?;

    Ok(acc.expect("invariant: indexed fold accumulator is always present"))
}

/// Fold one read-only view with the logical Fortran/column-major index order.
///
/// This is the column-major analogue of [`indexed_fold`]. It visits axis `0`
/// fastest, then axis `1`, and so on, independent of the backing storage
/// layout. Use this only when the logical visitation order is part of the
/// caller's contract.
pub fn indexed_fold_fortran<T, Acc, F, const N: usize>(
    view: &ArrayView<'_, T, N>,
    init: Acc,
    mut f: F,
) -> Result<Acc>
where
    F: FnMut(Acc, [usize; N], &T) -> Acc,
{
    view.layout().validate_storage_len(view.data().len())?;

    let size = view.layout().checked_size()?;
    if size == 0 {
        return Ok(init);
    }
    let shape = view.shape();
    let layout = view.layout();
    let data = view.data();
    let mut acc = init;
    for flat in 0..size {
        let mut index = [0usize; N];
        let mut remaining = flat;
        for axis in 0..N {
            if shape[axis] > 0 {
                index[axis] = remaining % shape[axis];
                remaining /= shape[axis];
            }
        }
        let offset = layout.offset_of(index)?;
        acc = f(acc, index, &data[offset]);
    }

    Ok(acc)
}
