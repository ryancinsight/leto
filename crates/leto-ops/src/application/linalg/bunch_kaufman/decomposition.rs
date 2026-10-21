use super::{decompose, solve};
use crate::domain::real::RealScalar;
use decompose::Factored;
use leto::{Array1, Array2, ArrayView1, ArrayView2, Result};

/// Bunch–Kaufman `P A Pᵀ = L D Lᵀ` decomposition.
#[derive(Debug, Clone)]
pub struct BunchKaufmanDecomposition<T> {
    factor: Factored<T>,
}

/// Compute the Bunch–Kaufman factorization of a symmetric matrix.
///
/// # Errors
/// [`LetoError::ShapeMismatch`](leto::LetoError) for non-square input;
/// [`LetoError::StorageError`](leto::LetoError) for non-symmetric, non-finite, or
/// singular-2×2-pivot input.
pub fn bunch_kaufman<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<BunchKaufmanDecomposition<T>> {
    Ok(BunchKaufmanDecomposition {
        factor: decompose::factor(matrix)?,
    })
}

impl<T: RealScalar> BunchKaufmanDecomposition<T> {
    /// Unit lower-triangular factor `L` (`n × n`).
    #[must_use]
    pub fn l(&self) -> Array2<T> {
        Array2::from_shape_vec([self.factor.n, self.factor.n], self.factor.l.clone())
            .expect("L shape matches storage")
    }

    /// Block-diagonal factor `D` (`n × n`, with 1×1 and 2×2 blocks).
    #[must_use]
    pub fn d(&self) -> Array2<T> {
        Array2::from_shape_vec([self.factor.n, self.factor.n], self.factor.d.clone())
            .expect("D shape matches storage")
    }

    /// Symmetric permutation as `perm[i]` = the original index now at position
    /// `i` (so `(P A Pᵀ)[i,j] = A[perm[i], perm[j]]`).
    #[must_use]
    pub fn permutation(&self) -> &[usize] {
        &self.factor.perm
    }

    /// `true` at index `k` when columns `(k, k+1)` form a 2×2 pivot block.
    #[must_use]
    pub fn is_two_by_two(&self) -> &[bool] {
        &self.factor.two
    }

    /// Determinant `det(A) = det(D)`.
    #[must_use]
    pub fn det(&self) -> T {
        solve::determinant(&self.factor)
    }

    /// Solve `A x = rhs`.
    ///
    /// # Errors
    /// [`LetoError`](leto::LetoError) on dimension mismatch or a singular factor.
    pub fn solve(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        solve::solve(&self.factor, rhs)
    }

    /// Inverse `A⁻¹`.
    ///
    /// # Errors
    /// [`LetoError`](leto::LetoError) on a singular factor.
    pub fn inv(&self) -> Result<Array2<T>> {
        solve::inverse(&self.factor)
    }
}
