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

/// Batched 3-vector cross product: `out[3i..3i+3] = cross(a[3i..3i+3], b[3i..3i+3])`.
///
/// CPU counterpart of `hephaestus_core::CrossProductOps` (1:1 parity): all
/// three operands share one length that is a multiple of three, and each
/// triple combines as `(ay·bz − az·by, az·bx − ax·bz, ax·by − ay·bx)` — the
/// multiplies in the kernel's order, so float results match the device lane
/// for lane. Contiguous operands take a slice fast path; strided views fall
/// back to stride-addressed traversal without materializing a copy. In-place
/// operation is sound: each output triple reads only its own input triples.
///
/// # Errors
///
/// [`LetoError::ShapeMismatch`] when the three lengths disagree;
/// [`LetoError::StorageError`] when their shared length is not a multiple
/// of three; and the layout errors for invalid storage lengths.
pub fn cross_into<T: Scalar>(
    a: &ArrayView<'_, T, 1>,
    b: &ArrayView<'_, T, 1>,
    out: &mut ArrayViewMut<'_, T, 1>,
) -> Result<()> {
    if a.shape() != b.shape() || a.shape() != out.shape() {
        return Err(LetoError::ShapeMismatch {
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        });
    }
    let len = a.shape()[0];
    if !len.is_multiple_of(3) {
        return Err(LetoError::StorageError {
            reason: format!("cross product operand length {len} is not a multiple of three"),
        });
    }
    a.layout().validate_storage_len(a.data().len())?;
    b.layout().validate_storage_len(b.data().len())?;
    out.layout().validate_storage_len(out.data().len())?;

    if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
        (a.as_slice(), b.as_slice(), out.as_mut_slice())
    {
        for ((a_triple, b_triple), out_triple) in a_slice
            .chunks_exact(3)
            .zip(b_slice.chunks_exact(3))
            .zip(out_slice.chunks_exact_mut(3))
        {
            let (ax, ay, az) = (a_triple[0], a_triple[1], a_triple[2]);
            let (bx, by, bz) = (b_triple[0], b_triple[1], b_triple[2]);
            out_triple[0] = ay.mul(bz).sub(az.mul(by));
            out_triple[1] = az.mul(bx).sub(ax.mul(bz));
            out_triple[2] = ax.mul(by).sub(ay.mul(bx));
        }
        return Ok(());
    }

    let a_layout = a.layout();
    let b_layout = b.layout();
    let out_layout = out.layout();
    let a_data = a.data();
    let b_data = b.data();
    let out_data = out.data_mut();
    for base in (0..len).step_by(3) {
        let ax = a_data[a_layout.offset_of([base])?];
        let ay = a_data[a_layout.offset_of([base + 1])?];
        let az = a_data[a_layout.offset_of([base + 2])?];
        let bx = b_data[b_layout.offset_of([base])?];
        let by = b_data[b_layout.offset_of([base + 1])?];
        let bz = b_data[b_layout.offset_of([base + 2])?];
        out_data[out_layout.offset_of([base])?] = ay.mul(bz).sub(az.mul(by));
        out_data[out_layout.offset_of([base + 1])?] = az.mul(bx).sub(ax.mul(bz));
        out_data[out_layout.offset_of([base + 2])?] = ax.mul(by).sub(ay.mul(bx));
    }
    Ok(())
}
