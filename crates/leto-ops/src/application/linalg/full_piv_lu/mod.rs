//! LU with complete (full) pivoting: `P A Q = L U`.
//!
//! Complete pivoting selects, at each step, the largest-magnitude entry of the
//! entire trailing submatrix as the pivot — via both a row and a column swap —
//! rather than only the largest in the pivot column (partial pivoting, see
//! [`crate::lu_decompose`]). This makes the factorization **rank-revealing** and maximally
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
//! Leaf modules: `decompose` (the elimination) and `solve` (forward/back
//! substitution and inverse). Generic over [`crate::RealScalar`], native precision.

mod decompose;
mod decomposition;
mod solve;

pub use decomposition::{full_piv_lu, FullPivLuDecomposition};
