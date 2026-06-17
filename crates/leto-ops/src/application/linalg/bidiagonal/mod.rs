//! Golub–Kahan bidiagonalization `A = U B Vᵀ` by two-sided Householder
//! reflectors (for `m ≥ n`).
//!
//! `B` is upper bidiagonal (nonzero only on the diagonal and first
//! superdiagonal). Bidiagonalization is the classical first phase of the SVD:
//! the singular values of `A` equal those of the much simpler `B`.
//!
//! # Theorem (bidiagonal reduction)
//! Every `A ∈ ℝᵐˣⁿ` with `m ≥ n` admits `A = U B Vᵀ` with `U` (`m×m`), `V`
//! (`n×n`) orthogonal and `B` upper bidiagonal. *Proof (constructive):*
//! alternately apply Householder reflectors. A left reflector `Lₖ` on the
//! sub-column `A[k.., k]` zeroes everything below the diagonal in column `k`; a
//! right reflector `Rₖ` on the sub-row `A[k, k+1..]` zeroes everything past the
//! superdiagonal in row `k`. `Rₖ` mixes only columns `k+1..n`, so it does not
//! refill column `k`; `L_{k+1}` mixes only rows `k+1..m`, so it does not refill
//! row `k`. Hence the created zeros persist and after `k = 0…n−1` the result is
//! upper bidiagonal, with `U = L₀…L_{n-1}`, `V = R₀…R_{n-2}` orthogonal. ∎
//!
//! # Corollary (singular-value preservation)
//! `B = Uᵀ A V` with `U, V` orthogonal, so `A` and `B` have identical singular
//! values (and `‖B‖_F = ‖A‖_F`).
//!
//! Uses the shared `householder` primitive (SSOT) and the
//! `reduce` loop. Generic over [`crate::RealScalar`], native precision. Wide inputs
//! (`m < n`) are rejected — transpose first (the bidiagonalization of `Aᵀ` is
//! the lower-bidiagonal form of `A`).

mod colmajor;
mod reduce;

use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, LetoError, Result, Storage};

/// Bidiagonal decomposition `A = U B Vᵀ` (`B` upper bidiagonal).
#[derive(Debug, Clone)]
pub struct BidiagonalDecomposition<T> {
    u: Array2<T>,
    b: Array2<T>,
    v: Array2<T>,
}

impl<T: RealScalar> BidiagonalDecomposition<T> {
    /// Left orthogonal factor `U` (`m × m`).
    #[must_use]
    pub fn u(&self) -> &Array2<T> {
        &self.u
    }

    /// Upper bidiagonal factor `B` (`m × n`).
    #[must_use]
    pub fn b(&self) -> &Array2<T> {
        &self.b
    }

    /// Right orthogonal factor `V` (`n × n`).
    #[must_use]
    pub fn v(&self) -> &Array2<T> {
        &self.v
    }
}

/// Golub–Kahan bidiagonalization of a tall-or-square matrix (`m ≥ n`).
///
/// # Errors
/// [`LetoError::ShapeMismatch`] when `m < n` (transpose wide inputs);
/// [`LetoError::StorageError`] for a non-finite entry.
pub fn bidiagonalize<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<BidiagonalDecomposition<T>> {
    let [m, n] = matrix.shape();
    if m < n {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![m, n],
            rhs: vec![n, n],
        });
    }
    // Finiteness is validated inside the reduction's single bulk input copy.
    let (u, b, v) = reduce::reduce_to_bidiagonal(matrix, m, n)?;
    Ok(BidiagonalDecomposition {
        u: Array2::from_shape_vec([m, m], u).expect("U shape matches storage"),
        b: Array2::from_shape_vec([m, n], b).expect("B shape matches storage"),
        v: Array2::from_shape_vec([n, n], v).expect("V shape matches storage"),
    })
}

/// Bidiagonal `(d, e)` of `A` (`m ≥ n`) via the column-major working buffer (the
/// SVD values path's locality experiment; `σ(A) = σ(B)`).
///
/// # Errors
/// [`LetoError::ShapeMismatch`] for `m < n`; [`LetoError::StorageError`] for a
/// non-finite entry.
pub(crate) fn bidiagonal_diag_colmajor<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<(Vec<T>, Vec<T>)> {
    let [m, n] = matrix.shape();
    if m < n {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![m, n],
            rhs: vec![n, n],
        });
    }
    let contiguous = matrix.to_contiguous();
    let a = contiguous.storage().as_slice();
    if !a.iter().all(|x| x.is_finite()) {
        return Err(LetoError::StorageError {
            reason: "bidiagonalization input contains a non-finite value".to_string(),
        });
    }
    Ok(colmajor::reduce_values(a, m, n))
}

