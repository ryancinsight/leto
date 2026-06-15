//! Matrix trace — the sum of the main-diagonal entries.

use crate::domain::scalar::Scalar;
use leto::{ArrayView2, LetoError, Result};

/// Trace `tr(A) = Σᵢ aᵢᵢ` of a square matrix.
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
    let mut acc = T::ZERO;
    for i in 0..rows {
        acc = acc.add(*matrix.get([i, i])?);
    }
    Ok(acc)
}
