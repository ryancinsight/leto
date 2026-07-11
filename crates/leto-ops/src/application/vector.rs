use crate::domain::scalar::Scalar;
use leto::{ArrayView, ArrayViewMut, LetoError, Result};

/// Dense matrix–vector product: `out[i] = Σ_j a[i, j]·x[j]`.
///
/// Accumulation runs in the native precision of `T` per the `Scalar` contract;
/// no wider accumulator is introduced. A C-contiguous `a` with contiguous `x`
/// and `out` takes the per-row [`Scalar::dot_slice`] fast path; strided inputs
/// (e.g. a transposed matrix view `a.transpose([1, 0])`, giving `Aᵀx`) fall
/// back to stride-addressed traversal without materializing a copy.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] when `a.shape()[1] != x.len()` or
/// `out.len() != a.shape()[0]`.
pub fn matvec<T: Scalar>(
    a: &ArrayView<'_, T, 2>,
    x: &ArrayView<'_, T, 1>,
    out: &mut ArrayViewMut<'_, T, 1>,
) -> Result<()> {
    let [rows, cols] = a.shape();
    if x.shape()[0] != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: a.shape().to_vec(),
            rhs: x.shape().to_vec(),
        });
    }
    if out.shape()[0] != rows {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows],
            rhs: out.shape().to_vec(),
        });
    }

    // Fast path: C-contiguous matrix rows dotted with a contiguous vector.
    if let (Some(a_slice), Some(x_slice), true) =
        (a.as_slice(), x.as_slice(), out.as_mut_slice().is_some())
    {
        let out_slice = out
            .as_mut_slice()
            .expect("invariant: out contiguity re-checked");
        for (row, out_value) in a_slice.chunks_exact(cols).zip(out_slice.iter_mut()) {
            *out_value = T::dot_slice(row, x_slice);
        }
        return Ok(());
    }

    // Strided fallback: address each element through the layouts.
    let a_layout = a.layout();
    let x_layout = x.layout();
    let out_layout = out.layout();
    let a_data = a.data();
    let x_data = x.data();
    let out_data = out.data_mut();
    for i in 0..rows {
        let mut acc = T::ZERO;
        for j in 0..cols {
            let a_off = a_layout.offset_of([i, j])?;
            let x_off = x_layout.offset_of([j])?;
            acc = acc.add(a_data[a_off].mul(x_data[x_off]));
        }
        let o_off = out_layout.offset_of([i])?;
        out_data[o_off] = acc;
    }
    Ok(())
}

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
pub fn jaccard_distance<T: Scalar>(
    a: &ArrayView<'_, T, 1>,
    b: &ArrayView<'_, T, 1>,
) -> Result<f64> {
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
pub fn hamming_distance<T: Scalar>(
    a: &ArrayView<'_, T, 1>,
    b: &ArrayView<'_, T, 1>,
) -> Result<u64> {
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
