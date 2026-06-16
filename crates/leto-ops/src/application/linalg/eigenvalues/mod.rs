//! Eigenvalues of a general (non-symmetric) real matrix, real and complex.
//!
//! # Theorem (real Schur form ⇒ eigenvalues on the diagonal blocks)
//! Every `A ∈ ℝⁿˣⁿ` is orthogonally similar to a real quasi-upper-triangular
//! `T` (`A = Q T Qᵀ`, real Schur form) with 1×1 blocks for real eigenvalues and
//! 2×2 blocks for complex-conjugate pairs. Similarity preserves the spectrum, so
//! **the eigenvalues of `A` are exactly the eigenvalues of the diagonal blocks of
//! `T`** — each 1×1 block a real eigenvalue, each 2×2 block a conjugate pair (its
//! quadratic). ∎
//!
//! Implementation: delegate to the real Schur decomposition
//! ([`crate::schur`](crate::schur), Francis double-shift QR in **real**
//! arithmetic) and read the eigenvalues off its quasi-triangular factor. This is
//! the single QR iteration in the crate (SSOT): the former complex single-shift
//! Wilkinson iteration is superseded — staying in real arithmetic removes the
//! `Complex` per-element cost and shares the Hessenberg reduction, double-shift
//! step, and block eigenvalue extraction with the Schur-vector path.
//!
//! For symmetric inputs prefer the dedicated Jacobi solver
//! ([`crate::symmetric_eigenvalues_jacobi`]), which returns sorted real
//! eigenvalues.

use crate::domain::real::RealScalar;
use leto::{ArrayView2, Result};
use num_complex::Complex;

/// Compute all eigenvalues of a square real matrix (real and complex).
///
/// Real eigenvalues have zero imaginary part; complex eigenvalues appear in
/// conjugate pairs. Order is unspecified; callers needing a canonical order
/// should sort.
///
/// # Errors
/// [`LetoError::ShapeMismatch`](leto::LetoError) for non-square input;
/// [`LetoError::StorageError`](leto::LetoError) for non-finite input or QR
/// non-convergence.
pub fn eigenvalues<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Vec<Complex<T>>> {
    Ok(crate::schur(matrix)?.eigenvalues())
}
