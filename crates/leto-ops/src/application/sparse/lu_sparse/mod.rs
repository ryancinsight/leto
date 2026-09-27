//! Sparse direct solver: `A x = b` for sparse square systems via partial-pivoting LU.
//!
//! # Algorithm
//!
//! [`SparseLuSolver`] is a **dispatcher**: for inputs that are small or
//! near-dense the existing dense [`lu_decompose`](crate::application::linalg::lu)
//! path (the Atlas SSOT dense LU) is dispatched via the solver's dense
//! fallback; for inputs that are large and sparse a real sparse LU
//! factorization over CSC with partial pivoting runs through
//! [`super::lu_numeric::factor_numeric`] and [`super::lu_symbolic::factor_symbolic`].
//!
//! ## Dispatch criteria
//!
//! The dense path is selected when:
//! - `n <= self.small_switch` (matrix is small enough for the dense path's
//!   constant factor to win the symbolic-factorization overhead), OR
//! - `density >= self.density_threshold` (matrix is near dense so the
//!   sparse path's `O(nnz)` savings vanish and its constant factor becomes
//!   a tax).
//!
//! Otherwise the sparse path runs. Both paths produce a value-identical
//! solution `x = A⁻¹ b` up to floating-point rounding; the differential
//! test suite in [`super::lu_numeric`] verifies this equivalence on a
//! deterministic medium-sized sparse system.
//!
//! # Why the dense path stays
//!
//! A real sparse LU has a non-zero constant factor (symbolic factorization
//! overhead + sparse data structure traversal). For matrices small enough
//! that the dense LU's `n³` cost fits comfortably in cache (<~32 by
//! measurement), the dense path is faster despite its asymptotic
//! disadvantage. The crossover threshold (`small_switch`) expresses
//! this and is configurable for callers with profiled workloads. The
//! dense path is also the proven correctness baseline against which the
//! sparse path is differential-tested — it stays in-tree as the oracle.
//!
//! # Theorem — correctness boundary
//!
//! For any nonsingular `A ∈ T^{n×n}`:
//! - If `n ≤ self.small_switch` or `density(A) ≥ self.density_threshold`,
//!   the dense path produces `x = A⁻¹b` up to IEEE 754 rounding (CSR
//!   expansion is exact, dense LU is partial-pivoting stable).
//! - Else the sparse path produces `x = A⁻¹b` up to IEEE 754 rounding
//!   under the partial-pivoting stability guarantee (Davis 2006 §8.7;
//!   partial-pivoting factorization is backward stable for any nonsingular
//!   matrix).
//!
//! Both paths return the same typed errors on the same failure modes
//! (`LetoError::ShapeMismatch`, `LetoError::StorageError`) so consumers'
//! error-handling is unchanged.
//!
//! `types` holds [`SparseLuSolver`], its dispatch constants, and
//! [`OrderingStrategy`]; `solver` is the dispatch and solve/factor methods;
//! `owned` holds [`OwnedNumericLu`], the symbolic-borrow-free cached
//! factorization; `convenience` names [`sparse_lu_solve`] and the
//! `csr_to_dense` bridge the dense fallback path uses.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

mod convenience;
mod owned;
mod solver;
mod types;

pub use convenience::{csr_to_dense, sparse_lu_solve};
pub use owned::OwnedNumericLu;
pub use types::{OrderingStrategy, SparseLuSolver, DENSE_LIMIT_DEFAULT};

#[cfg(test)]
mod tests;
