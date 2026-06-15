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
//! Two paths share the [`SvdDecomposition`] contract (SSOT for the result type):
//! - [`gram`] — full-rank thin SVD via the Gram matrix + symmetric Jacobi
//!   eigensolver; rejects rank-deficient input.
//! - [`jacobi`] — **rank-revealing** one-sided Jacobi SVD; accepts rank-deficient
//!   input and surfaces zero singular values honestly (ADR 0005).
//!
//! [`pseudoinverse`] builds the Moore-Penrose `A⁺` on the rank-revealing path.

use crate::domain::real::RealScalar;
use leto::{ArrayView2, LetoError, Result};

/// Full-rank thin SVD via the Gram matrix.
pub mod gram;
/// Rank-revealing one-sided Jacobi SVD.
pub mod jacobi;
/// Moore-Penrose pseudoinverse.
pub mod pseudoinverse;

pub use gram::{singular_values, svd_decompose, svd_decompose_with_tolerance};
pub use jacobi::{svd_rank_revealing, svd_rank_revealing_with_tolerance};
pub use pseudoinverse::pinv;

/// Thin singular value decomposition `A = U Σ Vᵀ`.
///
/// `singular_values` are sorted descending (length `k = min(m, n)`);
/// `left_singular_vectors` is `U` (`m × k`, columns) and
/// `right_singular_vectors` is `V` (`n × k`, columns). On the rank-revealing
/// [`jacobi`] path, singular values may be zero, in which case the corresponding
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

/// Map a Gram eigenvalue to a singular value, treating small-magnitude negative
/// eigenvalues (round-off) as zero and rejecting genuinely negative ones.
pub(super) fn singular_value_or_zero<T: RealScalar>(eigenvalue: T, tolerance: T) -> Result<T> {
    if eigenvalue < T::ZERO {
        if eigenvalue.neg() > tolerance {
            return Err(LetoError::StorageError {
                reason: "SVD normal matrix has a negative eigenvalue beyond tolerance".to_string(),
            });
        }
        return Ok(T::ZERO);
    }
    Ok(eigenvalue.sqrt())
}
