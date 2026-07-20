//! Sparse matrix–vector product `y = A · x` over CSR, in `O(nnz)` time.

use super::CsrMatrix;
use crate::domain::scalar::Scalar;
use leto::{Array1, ArrayView1, LetoError, Result, Storage};

/// Compute `y = A · x` into the caller-owned slice `y` (length `nrows`),
/// overwriting it. One pass over the stored nonzeros (`O(nnz)`); see the module
/// theorem for correctness and complexity.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] if `x` is not length `ncols` or `y` is not
/// length `nrows`.
pub fn spmv_into<T: Scalar>(a: &CsrMatrix<T>, x: &ArrayView1<'_, T>, y: &mut [T]) -> Result<()> {
    let (nrows, ncols) = a.shape();
    if x.shape() != [ncols] {
        return Err(LetoError::ShapeMismatch {
            lhs: x.shape().to_vec(),
            rhs: vec![ncols],
        });
    }
    if y.len() != nrows {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![y.len()],
            rhs: vec![nrows],
        });
    }

    if let Some(xs) = x.as_slice() {
        return spmv_slice_into(a, xs, y);
    }

    // Materialize non-contiguous x once: the gather `x[col_indices[p]]` then
    // reads a plain slice instead of performing a strided lookup per nonzero.
    let x_contiguous = x.to_contiguous();
    spmv_slice_into(a, x_contiguous.storage().as_slice(), y)
}

fn spmv_slice_into<T: Scalar>(a: &CsrMatrix<T>, xs: &[T], y: &mut [T]) -> Result<()> {
    let (values, col_indices, row_ptr) = a.as_parts();

    // Iterate rows through `row_ptr.windows(2)` zipped with `y` — eliding the
    // `y[i]`/`row_ptr[i]`/`row_ptr[i+1]` bounds checks — and slice each row's
    // value/column runs so the per-nonzero `values[p]`/`col_indices[p]` checks
    // fall away too (one `O(nrows)` slice check replaces `O(nnz)` element
    // checks). The residual `xs[col]` check is data-dependent; the CSR
    // invariant (`col_indices[p] < ncols == xs.len()`, from `from_parts`) proves
    // it in-bounds but the compiler cannot, so it stays for safety.
    for (slot, window) in y.iter_mut().zip(row_ptr.windows(2)) {
        let (start, end) = (window[0], window[1]);
        let cols = &col_indices[start..end];
        let vals = &values[start..end];
        let mut acc = T::ZERO;
        for (&value, &col) in vals.iter().zip(cols) {
            acc = acc.add(value.mul(xs[col]));
        }
        *slot = acc;
    }
    Ok(())
}

/// Compute `y = A · x`, allocating the length-`nrows` result. Thin wrapper over
/// [`spmv_into`] (SSOT).
///
/// # Errors
/// [`LetoError::ShapeMismatch`] if `x` is not length `ncols`.
pub fn spmv<T: Scalar>(a: &CsrMatrix<T>, x: &ArrayView1<'_, T>) -> Result<Array1<T>> {
    let (nrows, _) = a.shape();
    let mut y = vec![T::ZERO; nrows];
    spmv_into(a, x, &mut y)?;
    Array1::from_shape_vec([nrows], y)
}
