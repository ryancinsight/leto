use crate::application::index::index_from_flat;
use leto::{ArrayView, ArrayViewMut, LetoError, Result};

#[inline]
fn validate_zip_storage<T, U, const N: usize>(
    lhs: &ArrayViewMut<'_, T, N>,
    rhs: &ArrayView<'_, U, N>,
) -> Result<()> {
    lhs.layout().validate_storage_len(lhs.data().len())?;
    rhs.layout().validate_storage_len(rhs.data().len())?;
    if lhs.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "zip mutable output layout must not contain zero-stride aliasing".to_string(),
        });
    }
    Ok(())
}

/// Mutably zip-map elements of a view with elements from another view in place.
///
/// `lhs` owns mutation, `rhs` is read-only, and both views must have identical
/// logical shapes. Strided inputs are traversed by logical row-major index so
/// the result is independent of the backing storage layout.
pub fn zip_mut_with<T, U, F, const N: usize>(
    lhs: &mut ArrayViewMut<'_, T, N>,
    rhs: &ArrayView<'_, U, N>,
    mut f: F,
) -> Result<()>
where
    F: FnMut(&mut T, &U),
{
    if lhs.shape() != rhs.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    validate_zip_storage(lhs, rhs)?;

    if let (Some(lhs_slice), Some(rhs_slice)) = (lhs.as_mut_slice(), rhs.as_slice()) {
        for (left, right) in lhs_slice.iter_mut().zip(rhs_slice.iter()) {
            f(left, right);
        }
        return Ok(());
    }

    let size = lhs.layout().checked_size()?;
    let shape = lhs.shape();
    let lhs_layout = lhs.layout();
    let rhs_layout = rhs.layout();
    let lhs_data = lhs.data_mut();
    let rhs_data = rhs.data();

    for flat_idx in 0..size {
        let index = index_from_flat(flat_idx, &shape);
        let lhs_offset = lhs_layout.offset_of(index)?;
        let rhs_offset = rhs_layout.offset_of(index)?;
        f(&mut lhs_data[lhs_offset], &rhs_data[rhs_offset]);
    }

    Ok(())
}

/// Mutably zip-map a view with elements from two read-only views in place.
///
/// The three-operand analogue of [`zip_mut_with`] (`ndarray`'s
/// `Zip::from(out).and(a).and(b)`). `lhs` owns mutation; `a` and `b` are
/// read-only. All three views must share the same logical shape. Strided inputs
/// are traversed by logical row-major index, independent of backing layout.
pub fn zip2_mut_with<T, A, B, F, const N: usize>(
    lhs: &mut ArrayViewMut<'_, T, N>,
    a: &ArrayView<'_, A, N>,
    b: &ArrayView<'_, B, N>,
    mut f: F,
) -> Result<()>
where
    F: FnMut(&mut T, &A, &B),
{
    if lhs.shape() != a.shape() || lhs.shape() != b.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: a.shape().to_vec(),
        });
    }

    lhs.layout().validate_storage_len(lhs.data().len())?;
    a.layout().validate_storage_len(a.data().len())?;
    b.layout().validate_storage_len(b.data().len())?;
    if lhs.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: "zip mutable output layout must not contain zero-stride aliasing".to_string(),
        });
    }

    if let (Some(lhs_slice), Some(a_slice), Some(b_slice)) =
        (lhs.as_mut_slice(), a.as_slice(), b.as_slice())
    {
        for ((left, av), bv) in lhs_slice.iter_mut().zip(a_slice.iter()).zip(b_slice.iter()) {
            f(left, av, bv);
        }
        return Ok(());
    }

    let size = lhs.layout().checked_size()?;
    let shape = lhs.shape();
    let lhs_layout = lhs.layout();
    let a_layout = a.layout();
    let b_layout = b.layout();
    let lhs_data = lhs.data_mut();
    let a_data = a.data();
    let b_data = b.data();

    for flat_idx in 0..size {
        let index = index_from_flat(flat_idx, &shape);
        let lhs_offset = lhs_layout.offset_of(index)?;
        let a_offset = a_layout.offset_of(index)?;
        let b_offset = b_layout.offset_of(index)?;
        f(
            &mut lhs_data[lhs_offset],
            &a_data[a_offset],
            &b_data[b_offset],
        );
    }

    Ok(())
}
