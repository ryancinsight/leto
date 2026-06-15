//! LU with complete (full) pivoting: `P A Q = L U`.
//!
//! Complete pivoting selects, at each step, the largest-magnitude entry of the
//! entire trailing submatrix as the pivot — via both a row and a column swap —
//! rather than only the largest in the pivot column (partial pivoting, see
//! [`super::lu`]). This makes the factorization **rank-revealing** and maximally
//! stable, at the cost of the `O(n³)` pivot search.
//!
//! # Theorem (complete-pivoting LU)
//! For every `A ∈ ℝⁿˣⁿ` there exist permutation matrices `P, Q`, a unit
//! lower-triangular `L`, and an upper-triangular `U` with `P A Q = L U`.
//! *Proof (constructive):* if the trailing submatrix is nonzero, a row and a
//! column swap bring a nonzero pivot to `(k,k)`; one Gaussian elimination step
//! clears the column below it. If the trailing submatrix is entirely zero the
//! process stops and the remaining `U` rows are zero. ∎
//!
//! # Corollary (rank and determinant)
//! The number of nonzero pivots is `rank(A)` — complete pivoting orders pivots
//! by decreasing magnitude, so the first negligible pivot reveals the rank.
//! For full rank, `det(A) = sign(P)·sign(Q)·∏ₖ Uₖₖ`.
//!
//! Leaf modules: [`decompose`] (the elimination) and [`solve`] (forward/back
//! substitution and inverse). Generic over [`RealScalar`], native precision.

mod decompose;
mod solve;

use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, ArrayView2, Result};

/// Complete-pivoting LU decomposition `P A Q = L U`.
#[derive(Debug, Clone)]
pub struct FullPivLuDecomposition<T> {
    /// Packed `L\U`: unit `L` strictly below the diagonal, `U` on and above.
    lu: Vec<T>,
    row_perm: Vec<usize>,
    col_perm: Vec<usize>,
    sign: i8,
    rank: usize,
    n: usize,
}

impl<T: RealScalar> FullPivLuDecomposition<T> {
    /// Numerical rank (count of nonzero pivots).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Determinant: `0` when rank-deficient, else `sign(P)·sign(Q)·∏ Uₖₖ`.
    #[must_use]
    pub fn det(&self) -> T {
        if self.rank < self.n {
            return T::ZERO;
        }
        let mut product = if self.sign >= 0 {
            T::ONE
        } else {
            T::ZERO.sub(T::ONE)
        };
        for k in 0..self.n {
            product = product.mul(self.lu[k * self.n + k]);
        }
        product
    }

    /// Unit lower-triangular factor `L` (`n × n`).
    #[must_use]
    pub fn l(&self) -> Array2<T> {
        let n = self.n;
        let mut values = vec![T::ZERO; n * n];
        for i in 0..n {
            for j in 0..n {
                values[i * n + j] = match i.cmp(&j) {
                    core::cmp::Ordering::Greater => self.lu[i * n + j],
                    core::cmp::Ordering::Equal => T::ONE,
                    core::cmp::Ordering::Less => T::ZERO,
                };
            }
        }
        Array2::from_shape_vec([n, n], values).expect("L shape matches storage")
    }

    /// Upper-triangular factor `U` (`n × n`).
    #[must_use]
    pub fn u(&self) -> Array2<T> {
        let n = self.n;
        let mut values = vec![T::ZERO; n * n];
        for i in 0..n {
            for j in i..n {
                values[i * n + j] = self.lu[i * n + j];
            }
        }
        Array2::from_shape_vec([n, n], values).expect("U shape matches storage")
    }

    /// Row permutation: `row_permutation()[k]` is the original row at position `k`.
    #[must_use]
    pub fn row_permutation(&self) -> &[usize] {
        &self.row_perm
    }

    /// Column permutation: `col_permutation()[k]` is the original column at position `k`.
    #[must_use]
    pub fn col_permutation(&self) -> &[usize] {
        &self.col_perm
    }

    /// Solve `A x = b`.
    ///
    /// # Errors
    /// [`LetoError`](leto::LetoError) on a rank-deficient matrix or shape mismatch.
    pub fn solve(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        solve::solve(&self.as_factored(), rhs)
    }

    /// Inverse `A⁻¹`.
    ///
    /// # Errors
    /// [`LetoError`](leto::LetoError) on a rank-deficient matrix.
    pub fn inv(&self) -> Result<Array2<T>> {
        solve::inverse(&self.as_factored())
    }

    fn as_factored(&self) -> decompose::Factored<T> {
        decompose::Factored {
            lu: self.lu.clone(),
            row_perm: self.row_perm.clone(),
            col_perm: self.col_perm.clone(),
            sign: self.sign,
            rank: self.rank,
            n: self.n,
        }
    }
}

/// Factor a square matrix with complete (full) pivoting.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] for non-square input;
/// [`LetoError::StorageError`] for a non-finite entry.
pub fn full_piv_lu<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<FullPivLuDecomposition<T>> {
    let f = decompose::factor(matrix)?;
    Ok(FullPivLuDecomposition {
        lu: f.lu,
        row_perm: f.row_perm,
        col_perm: f.col_perm,
        sign: f.sign,
        rank: f.rank,
        n: f.n,
    })
}
