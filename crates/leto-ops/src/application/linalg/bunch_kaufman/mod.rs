//! Symmetric-indefinite Bunch–Kaufman `P A Pᵀ = L D Lᵀ` factorization with
//! partial pivoting.
//!
//! This is the stable, fully general counterpart of the unpivoted
//! [`udu`](crate::application::linalg::udu) factorization: where unpivoted
//! `U D Uᵀ` fails on a (near-)zero pivot, Bunch–Kaufman selects a symmetric
//! permutation `P` and 1×1 / 2×2 pivot blocks so the factorization exists and is
//! backward stable for *every* symmetric matrix — including indefinite ones with
//! zero diagonals (e.g. `[[0,1],[1,0]]`).
//!
//! # Theorem (Bunch–Kaufman factorization)
//! For every symmetric `A ∈ ℝⁿˣⁿ` there is a permutation `P`, a unit
//! lower-triangular `L`, and a block-diagonal `D` with 1×1 and 2×2 blocks such
//! that `P A Pᵀ = L D Lᵀ`.
//!
//! *Proof (constructive).* Process the matrix from the top-left. With
//! `α = (1+√17)/8`, let `λ = max_{i>0} |A_{i,0}|` occur at row `r`. The pivoting
//! test compares `|A_{0,0}|` against `αλ` and, when needed, the column-`r`
//! quantity `σ` against `αλ²` and `|A_{r,r}|` against `ασ`, choosing either a
//! 1×1 pivot (optionally after the symmetric interchange `0 ↔ r`) or the 2×2
//! pivot drawn from `{0, r}` (brought to `{0,1}` by the interchange `1 ↔ r`).
//! The chosen pivot block `E` (1×1 or 2×2) is nonsingular by the test, so the
//! Schur complement `A' = A_{22} − A_{21} E⁻¹ A_{21}ᵀ` is again symmetric and one
//! or two columns of `L` are fixed by `L_{21} = A_{21} E⁻¹`. Recurse on `A'`.
//! Composing the per-step permutations gives `P`; the accumulated columns give
//! `L` and the pivot blocks give `D`. ∎
//!
//! The α threshold bounds the entry growth of `L` (each `|L_{ij}| ≤ 1/(1−α)` for
//! 1×1 steps), which is what makes the factorization backward stable; the proof
//! of that bound is Bunch & Kaufman (1977).
//!
//! # Corollaries
//! `det(A) = det(D) = ∏ (block determinants)` since `det(L) = 1` and
//! `det(P)² = 1`. Solves use a forward `L`-solve, a block-diagonal `D`-solve
//! (1×1 division and 2×2 inversion), a back `Lᵀ`-solve, and the symmetric
//! permutation on each side.
//!
//! Evidence tier: theorem/proof sketch in rustdoc plus value-semantic tests for
//! the exact reconstruction identity `P A Pᵀ = L D Lᵀ`, determinant, solve and
//! inverse (differential against the general LU solver), and indefinite /
//! zero-diagonal cases that force 2×2 pivots. Generic over [`crate::RealScalar`],
//! native precision.

mod decompose;
mod solve;

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
