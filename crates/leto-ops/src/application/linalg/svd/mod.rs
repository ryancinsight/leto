//! Singular value decomposition and the Moore-Penrose pseudoinverse.
//!
//! # Theorem (existence of the SVD)
//! Every `A ∈ ℝᵐˣⁿ` factors as `A = U Σ Vᵀ` with `U ∈ ℝᵐˣᵏ`, `V ∈ ℝⁿˣᵏ`
//! (`k = min(m,n)`) having orthonormal columns and `Σ = diag(σ₁ ≥ … ≥ σₖ ≥ 0)`.
//! *Proof sketch:* `AᵀA` is symmetric positive-semidefinite, so by the spectral
//! theorem it has an orthonormal eigenbasis `{vᵢ}` with eigenvalues `λᵢ ≥ 0`;
//! set `σᵢ = √λᵢ` and, for `σᵢ > 0`, `uᵢ = A vᵢ / σᵢ`. Then `{uᵢ}` are
//! orthonormal (`uᵢᵀuⱼ = vᵢᵀAᵀA vⱼ / (σᵢσⱼ) = λⱼ δᵢⱼ /(σᵢσⱼ) = δᵢⱼ`) and
//! `A vᵢ = σᵢ uᵢ`, i.e. `A V = U Σ`. ∎
//!
//! One implementation backs the whole surface (SSOT): `bidiagonal_qr`,
//! Golub–Reinsch implicit-shift bidiagonal QR. It never forms `AᵀA`, so
//! conditioning stays `κ(A)` rather than `κ(A)²`, and it is **rank-revealing**:
//! zero singular values emerge from the iteration itself, while `U` and `V`
//! stay orthogonal whatever the rank of `A` — both are accumulated products of
//! Householder reflectors and Givens rotations, and a product of orthogonal
//! factors is orthogonal irrespective of the singular values it diagonalizes.
//! Rank is therefore read off `Σ` rather than signalled by an error (ADR 0005).
//!
//! - [`svd_decompose`] — thin SVD `A = U Σ Vᵀ`, all shapes, any rank.
//! - [`singular_values`] — values only, no `U`/`V` accumulation.
//! - [`pinv`] — Moore-Penrose `A⁺` under a relative rank cutoff.

use crate::domain::real::RealScalar;
use leto::{ArrayView2, LetoError, Result};

/// Bidiagonal-QR SVD: thin SVD and singular values (accuracy-preserving).
pub mod bidiagonal_qr;
/// Moore-Penrose pseudoinverse.
pub mod pseudoinverse;

pub use bidiagonal_qr::{singular_values, svd_decompose};
pub use pseudoinverse::pinv;

/// Thin singular value decomposition `A = U Σ Vᵀ`.
///
/// `singular_values` are sorted descending (length `k = min(m, n)`) and are
/// non-negative; `left_singular_vectors` is `U` (`m × k`, columns) and
/// `right_singular_vectors` is `V` (`n × k`, columns). Rank-deficient input
/// yields zero singular values, and `U` and `V` keep orthonormal columns in
/// that case too — the null-space directions are materialized rather than left
/// zero, so `UᵀU = VᵀV = I` holds at every rank.
#[derive(Debug, Clone)]
pub struct SvdDecomposition<T> {
    /// Singular values sorted descending.
    pub singular_values: Vec<T>,
    /// Left singular vectors `U`, stored as columns.
    pub left_singular_vectors: leto::Array2<T>,
    /// Right singular vectors `V`, stored as columns.
    pub right_singular_vectors: leto::Array2<T>,
}

/// Reject empty or non-finite input.
pub(super) fn validate_input<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<()> {
    let [rows, cols] = matrix.shape();
    if rows == 0 || cols == 0 {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows.max(1), cols.max(1)],
        });
    }
    let all_finite = if let Some(slice) = matrix.as_slice() {
        slice.iter().all(|x| x.is_finite())
    } else {
        matrix.iter().all(|x| x.is_finite())
    };
    if !all_finite {
        return Err(LetoError::StorageError {
            reason: "SVD input contains a non-finite value".to_string(),
        });
    }
    Ok(())
}
