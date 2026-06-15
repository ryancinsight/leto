//! Symmetric indefinite unpivoted `U D Uᵀ` factorization.
//!
//! This factorization is the upper-triangular counterpart of `L D Lᵀ`.
//! It targets symmetric matrices whose leading reverse pivots are nonzero. When
//! a zero pivot is encountered, the correct general algorithm is symmetric
//! pivoting (for example Bunch-Kaufman), so this module reports an error rather
//! than fabricating a fallback.
//!
//! # Theorem (unpivoted UDU factorization)
//! Let `A ∈ ℝⁿˣⁿ` be symmetric and suppose each recursively computed pivot
//! `dⱼ = aⱼⱼ − Σₖ>ⱼ uⱼₖ² dₖ` is nonzero. Then there is a unit upper-triangular
//! `U` and diagonal `D` such that `A = U D Uᵀ`.
//! *Proof (constructive):* process columns from `n-1` down to `0`. Assume the
//! trailing block already satisfies
//! `A[i,j] = Σₖ≥max(i,j) uᵢₖ dₖ uⱼₖ` for indices greater than the current
//! column. The formulas
//! `dⱼ = A[j,j] − Σₖ>ⱼ uⱼₖ² dₖ` and
//! `uᵢⱼ = (A[i,j] − Σₖ>ⱼ uᵢₖuⱼₖdₖ) / dⱼ` make the equality true for every
//! entry touching column `j`. Induction reaches column `0`, so every entry of
//! `U D Uᵀ` equals `A`. ∎
//!
//! # Corollaries
//! The determinant is `det(A) = ∏ᵢ dᵢ` because `det(U)=det(Uᵀ)=1`. Linear solves
//! use two triangular solves and one diagonal scale:
//! `U y = b`, `z = D⁻¹ y`, `Uᵀ x = z`.
//!
//! Evidence tier: theorem/proof sketch in rustdoc plus value-semantic tests for
//! reconstruction, determinant, solve, inverse, invalid input, and zero-pivot
//! rejection. The implementation is generic over [`crate::RealScalar`] and executes in
//! native scalar precision.

mod decompose;
mod solve;

use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, ArrayView2, Result};

/// Unpivoted symmetric `U D Uᵀ` decomposition.
#[derive(Debug, Clone)]
pub struct UduDecomposition<T> {
    u: Vec<T>,
    d: Vec<T>,
    n: usize,
}

impl<T: RealScalar> UduDecomposition<T> {
    /// Unit upper-triangular factor `U` (`n × n`).
    #[must_use]
    pub fn u(&self) -> Array2<T> {
        Array2::from_shape_vec([self.n, self.n], self.u.clone()).expect("U shape matches storage")
    }

    /// Diagonal entries of `D`.
    #[must_use]
    pub fn diagonal(&self) -> &[T] {
        &self.d
    }

    /// Determinant `∏ᵢ Dᵢᵢ`.
    #[must_use]
    pub fn det(&self) -> T {
        self.d
            .iter()
            .copied()
            .fold(T::ONE, |acc, value| acc.mul(value))
    }

    /// Solve `A x = rhs` using the factored `A = U D Uᵀ`.
    ///
    /// # Errors
    /// [`leto::LetoError`] on a right-hand-side shape mismatch.
    pub fn solve(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        solve::solve(&self.as_factored(), rhs)
    }

    /// Inverse `A⁻¹`.
    ///
    /// # Errors
    /// Propagates allocation or internal shape errors from the output array
    /// constructors.
    pub fn inv(&self) -> Result<Array2<T>> {
        solve::inverse(&self.as_factored())
    }

    fn as_factored(&self) -> decompose::Factored<T> {
        decompose::Factored {
            u: self.u.clone(),
            d: self.d.clone(),
            n: self.n,
        }
    }
}

/// Factor a symmetric matrix as `A = U D Uᵀ` without pivoting.
///
/// # Errors
/// `LetoError::ShapeMismatch` for non-square input.
/// `LetoError::StorageError` for non-finite entries, nonsymmetric input, or
/// a zero pivot requiring symmetric pivoting.
pub fn udu_decompose<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<UduDecomposition<T>> {
    let f = decompose::factor(matrix)?;
    Ok(UduDecomposition {
        u: f.u,
        d: f.d,
        n: f.n,
    })
}
