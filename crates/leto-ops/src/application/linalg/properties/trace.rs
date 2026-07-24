//! Matrix trace — the sum of the main-diagonal entries.

use crate::domain::scalar::Scalar;
use leto::{ArrayView2, LetoError, Result};

/// Trace `tr(A) = Σᵢ aᵢᵢ` of a square matrix.
///
/// ```
/// use leto::Array2;
/// use leto_ops::trace;
///
/// let matrix = Array2::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();
/// assert_eq!(trace(&matrix.view()).unwrap(), 5);
/// ```
///
/// # Theorem (trace identities)
/// For square `A ∈ Tⁿˣⁿ`:
/// 1. **Spectral:** `tr(A) = Σᵢ λᵢ` — the sum of the eigenvalues with
///    multiplicity. *Proof:* the characteristic polynomial
///    `p(λ) = det(A − λI)` has `λⁿ⁻¹`-coefficient `−tr(A)` (cofactor expansion);
///    by Vieta's formulas that same coefficient equals `−Σᵢ λᵢ`, so
///    `tr(A) = Σᵢ λᵢ`. ∎
/// 2. **Cyclic:** `tr(AB) = tr(BA)` for conformable `A, B`. *Proof:*
///    `tr(AB) = Σᵢ Σₖ aᵢₖ bₖᵢ = Σₖ Σᵢ bₖᵢ aᵢₖ = tr(BA)` by reindexing. ∎
///
/// Evaluated in the native precision of `T` (no widening). The `n` diagonal
/// elements are read directly through strided indexing — zero allocation, no
/// materialization of the diagonal as a separate vector.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] when `matrix` is not square.
#[inline]
pub fn trace<T: Scalar>(matrix: &ArrayView2<'_, T>) -> Result<T> {
    let [rows, cols] = matrix.shape();
    if rows != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows, rows],
        });
    }
    if rows == 0 {
        return Ok(T::ZERO);
    }
    let data = matrix.data();
    let strides = matrix.strides();
    let diag_stride = strides[0] + strides[1];
    let mut offset = matrix.offset() as isize;
    let mut acc = T::ZERO;
    for _ in 0..rows {
        // SAFETY: The index [i, i] is logically in-bounds for a square matrix of size `rows`.
        // The layout is validated, and the precalculated diagonal stride corresponds exactly
        // to the offset delta for the next [i+1, i+1] element, staying within the bounds
        // of the storage slice.
        unsafe {
            acc = acc.add(*data.get_unchecked(offset as usize));
        }
        offset += diag_stride;
    }
    Ok(acc)
}
