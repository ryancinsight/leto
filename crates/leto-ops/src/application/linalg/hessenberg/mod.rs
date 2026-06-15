//! Upper Hessenberg reduction `A = Q H Qᵀ` by Householder reflectors.
//!
//! An upper Hessenberg matrix is zero below the first subdiagonal
//! (`hᵢⱼ = 0` for `i > j + 1`). Reducing to Hessenberg form is the standard
//! first phase of the non-symmetric eigenvalue problem: the subsequent Francis
//! QR iteration costs `O(n²)` per step on a Hessenberg matrix instead of `O(n³)`
//! on a full one, and the form is preserved by the iteration.
//!
//! # Theorem (Hessenberg reduction)
//! Every `A ∈ ℝⁿˣⁿ` admits `A = Q H Qᵀ` with `Q` orthogonal and `H` upper
//! Hessenberg. *Proof (constructive):* a Householder reflector
//! `Pₖ = I − βₖ vₖ vₖᵀ` is symmetric and orthogonal; choosing `vₖ` from the
//! sub-column `A[k+1.., k]` maps it onto `e₁`, zeroing `A[k+2.., k]`. Applying
//! `Pₖ` on *both* sides, `A ← Pₖ A Pₖ`, is an orthogonal **similarity**, so it
//! leaves the spectrum unchanged; crucially the right multiplication only mixes
//! columns `k+1..n`, which does not refill the zeros just created in column `k`.
//! After `k = 0 … n−3` the result is upper Hessenberg, with
//! `Q = P₀ P₁ … P_{n−3}` orthogonal (a product of orthogonal matrices). ∎
//!
//! # Corollary (spectrum preservation)
//! `H = Qᵀ A Q` is similar to `A`, so `H` and `A` have the same eigenvalues; in
//! particular `tr(H) = tr(A)` and `‖H‖_F = ‖A‖_F` (orthogonal invariance).
//!
//! Uses the shared [`householder`](super::householder) reflector primitive
//! (SSOT); [`reduce`] is the reduction loop. Generic over [`RealScalar`], native
//! precision throughout.

mod reduce;

use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, LetoError, Result};

/// Upper Hessenberg decomposition `A = Q H Qᵀ`.
#[derive(Debug, Clone)]
pub struct HessenbergDecomposition<T> {
    q: Array2<T>,
    h: Array2<T>,
}

impl<T: RealScalar> HessenbergDecomposition<T> {
    /// Orthogonal factor `Q` (`n × n`).
    #[must_use]
    pub fn q(&self) -> &Array2<T> {
        &self.q
    }

    /// Upper Hessenberg factor `H` (`n × n`), zero below the first subdiagonal.
    #[must_use]
    pub fn h(&self) -> &Array2<T> {
        &self.h
    }
}

/// Reduce a square matrix to upper Hessenberg form `A = Q H Qᵀ`.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] for non-square input;
/// [`LetoError::StorageError`] for a non-finite entry.
pub fn hessenberg<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<HessenbergDecomposition<T>> {
    let [rows, cols] = matrix.shape();
    if rows != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows, rows],
        });
    }
    for row in 0..rows {
        for col in 0..cols {
            if !matrix.get([row, col])?.is_finite() {
                return Err(LetoError::StorageError {
                    reason: "Hessenberg input contains a non-finite value".to_string(),
                });
            }
        }
    }

    let (h, q, n) = reduce::reduce_to_hessenberg(matrix)?;
    Ok(HessenbergDecomposition {
        q: Array2::from_shape_vec([n, n], q).expect("Q shape matches storage"),
        h: Array2::from_shape_vec([n, n], h).expect("H shape matches storage"),
    })
}
