use crate::domain::scalar::Scalar;
use leto::{ArrayView, LetoError, Result};

/// Dot product of two rank-1 views: `sum_i a[i] * b[i]`.
///
/// Accumulation runs in the native precision of `T` per the `Scalar` contract;
/// no wider accumulator is introduced. Contiguous inputs take a slice fast
/// path; strided inputs (e.g. a row of a transposed matrix) fall back to
/// stride-addressed traversal without materializing a copy.
pub fn dot<T: Scalar>(a: &ArrayView<'_, T, 1>, b: &ArrayView<'_, T, 1>) -> Result<T> {
    if a.shape() != b.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        });
    }
    a.layout().validate_storage_len(a.data().len())?;
    b.layout().validate_storage_len(b.data().len())?;

    if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
        let mut acc = T::ZERO;
        for (&x, &y) in a_slice.iter().zip(b_slice.iter()) {
            acc = acc.add(x.mul(y));
        }
        return Ok(acc);
    }

    let len = a.shape()[0];
    let a_layout = a.layout();
    let b_layout = b.layout();
    let a_data = a.data();
    let b_data = b.data();

    let mut acc = T::ZERO;
    for i in 0..len {
        let a_off = a_layout.offset_of([i])?;
        let b_off = b_layout.offset_of([i])?;
        acc = acc.add(a_data[a_off].mul(b_data[b_off]));
    }
    Ok(acc)
}
