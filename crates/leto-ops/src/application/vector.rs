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
        return Ok(T::dot_slice(a_slice, b_slice));
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



/// Jaccard distance between two binary rank-1 views: `1.0 - (popcount(a & b) / popcount(a | b))`.
pub fn jaccard_distance<T: Scalar>(a: &ArrayView<'_, T, 1>, b: &ArrayView<'_, T, 1>) -> Result<f64> {
    if a.shape() != b.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        });
    }
    a.layout().validate_storage_len(a.data().len())?;
    b.layout().validate_storage_len(b.data().len())?;

    if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
        if let Some(dist) = T::jaccard_distance(a_slice, b_slice) {
            return Ok(dist);
        }
    }

    // Fallback: scalar traversal
    let len = a.shape()[0];
    let a_layout = a.layout();
    let b_layout = b.layout();
    let a_data = a.data();
    let b_data = b.data();

    let mut intersection = 0u64;
    let mut union = 0u64;

    for i in 0..len {
        let a_off = a_layout.offset_of([i])?;
        let b_off = b_layout.offset_of([i])?;
        let x = a_data[a_off];
        let y = b_data[b_off];
        intersection += x.bitand(y).count_ones() as u64;
        union += x.bitor(y).count_ones() as u64;
    }

    if union == 0 {
        Ok(0.0)
    } else {
        Ok(1.0 - (intersection as f64) / (union as f64))
    }
}

/// Hamming distance between two binary rank-1 views: `popcount(a ^ b)`.
pub fn hamming_distance<T: Scalar>(a: &ArrayView<'_, T, 1>, b: &ArrayView<'_, T, 1>) -> Result<u64> {
    if a.shape() != b.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        });
    }
    a.layout().validate_storage_len(a.data().len())?;
    b.layout().validate_storage_len(b.data().len())?;

    if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
        if let Some(dist) = T::hamming_distance(a_slice, b_slice) {
            return Ok(dist);
        }
    }

    // Fallback: scalar traversal
    let len = a.shape()[0];
    let a_layout = a.layout();
    let b_layout = b.layout();
    let a_data = a.data();
    let b_data = b.data();

    let mut distance = 0u64;

    for i in 0..len {
        let a_off = a_layout.offset_of([i])?;
        let b_off = b_layout.offset_of([i])?;
        let x = a_data[a_off];
        let y = b_data[b_off];
        distance += x.bitxor(y).count_ones() as u64;
    }

    Ok(distance)
}
