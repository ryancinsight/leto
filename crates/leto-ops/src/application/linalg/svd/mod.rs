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
//! Paths share the [`SvdDecomposition`] contract (SSOT for the result type):
//! - `bidiagonal_qr` — implicit-shift bidiagonal QR (Golub–Reinsch): the default
//!   thin SVD ([`svd_decompose`]) and `singular_values`. Avoids `AᵀA`, so
//!   conditioning is `κ(A)` not `κ(A)²`; `svd_decompose` rejects rank-deficient
//!   input to preserve its contract.
//! - `jacobi` — **rank-revealing** one-sided Jacobi SVD; accepts rank-deficient
//!   input and surfaces zero singular values honestly (ADR 0005).
//!
//! `pseudoinverse` builds the Moore-Penrose `A⁺` on the rank-revealing path.

use crate::domain::real::RealScalar;
use leto::{ArrayView2, LetoError, Result};

/// Bidiagonal-QR SVD: default thin SVD, singular values (accuracy-preserving).
pub mod bidiagonal_qr;
/// Rank-revealing one-sided Jacobi SVD.
pub mod jacobi;
/// Moore-Penrose pseudoinverse.
pub mod pseudoinverse;

pub use bidiagonal_qr::{
    singular_values, svd_decompose, svd_decompose_with_tolerance, svd_via_bidiagonal,
};
pub use jacobi::{svd_rank_revealing, svd_rank_revealing_with_tolerance};
pub use pseudoinverse::pinv;

/// Thin singular value decomposition `A = U Σ Vᵀ`.
///
/// `singular_values` are sorted descending (length `k = min(m, n)`);
/// `left_singular_vectors` is `U` (`m × k`, columns) and
/// `right_singular_vectors` is `V` (`n × k`, columns). On the rank-revealing
/// [`self::jacobi`] path, singular values may be zero, in which case the corresponding
/// `U` column is zero (its direction lies in the left null space and is not
/// materialized); `V` is always fully orthonormal.
#[derive(Debug, Clone)]
pub struct SvdDecomposition<T> {
    /// Singular values sorted descending.
    pub singular_values: Vec<T>,
    /// Left singular vectors `U`, stored as columns.
    pub left_singular_vectors: leto::Array2<T>,
    /// Right singular vectors `V`, stored as columns.
    pub right_singular_vectors: leto::Array2<T>,
}

/// Default eigen/orthogonality tolerance: `1e-12` relative.
pub(super) fn default_tolerance<T: RealScalar>() -> T {
    T::ONE.div(T::from_usize(1_000_000_000_000))
}

/// Reject empty, non-finite, or invalid-tolerance input shared by both paths.
pub(super) fn validate_input<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    tolerance: T,
) -> Result<()> {
    let [rows, cols] = matrix.shape();
    if rows == 0 || cols == 0 {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows.max(1), cols.max(1)],
        });
    }
    if !tolerance.is_finite() || tolerance < T::ZERO {
        return Err(LetoError::StorageError {
            reason: "SVD tolerance must be finite and non-negative".to_string(),
        });
    }
    for row in 0..rows {
        for col in 0..cols {
            if !matrix.get([row, col])?.is_finite() {
                return Err(LetoError::StorageError {
                    reason: "SVD input contains a non-finite value".to_string(),
                });
            }
        }
    }
    Ok(())
}
