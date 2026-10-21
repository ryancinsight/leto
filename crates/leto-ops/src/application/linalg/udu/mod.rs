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
mod decomposition;
mod solve;

pub use decomposition::{udu_decompose, UduDecomposition};
