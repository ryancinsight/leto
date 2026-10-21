//! QR with column pivoting: `A P = Q R` (rank-revealing).
//!
//! Plain Householder QR ([`crate::qr_decompose`]) factors `A = Q R`; column pivoting adds
//! a permutation `P` that, at each step, moves the column of largest remaining
//! norm to the front. This forces `|R₀₀| ≥ |R₁₁| ≥ … ≥ |R_{r-1,r-1}| > 0` with
//! the trailing diagonal ≈ 0, so the factorization is **rank-revealing**.
//!
//! # Theorem (column-pivoted QR)
//! For every `A ∈ ℝᵐˣⁿ` there is a permutation `P`, an orthogonal `Q`, and an
//! upper-triangular `R` with `A P = Q R` and the diagonal of `R` non-increasing
//! in magnitude. *Proof (constructive):* at step `k`, choosing the
//! largest-tail-norm column as pivot guarantees `|Rₖₖ| ≥ ‖(R P)[k.., j]‖` for
//! all `j > k`; the Householder reflector that follows sets `Rₖₖ` to that pivot
//! column's tail norm, which therefore dominates every later diagonal. ∎
//!
//! # Corollary (rank)
//! The number of above-threshold diagonal entries is `rank(A)`; the monotone
//! diagonal means the first negligible one reveals the rank — more reliably than
//! a Gram-spectrum count for borderline cases.
//!
//! # Finite precision
//! The pivot key of each remaining column is its squared tail norm, downdated by
//! the removed row and recomputed exactly when the downdate cancels, as in
//! LAPACK 3.12.0 `DLAQP2` (`TOL3Z`, `dlaqp2.f` line 187; the update, lines
//! 236-249). The cached key differs from the exact tail norm by rounding, so
//! the pivot dominates the exact tail norm of every later column only up to the
//! relative tolerance derived in ADR 0035, and columns whose tail norms differ
//! by less than it may be ordered either way. The rank test uses the exact tail
//! norm of the selected column.
//!
//! Leaf modules: `decompose` (the pivoted reduction, on the shared
//! `householder` primitive) and the solve logic here.
//! Generic over [`crate::RealScalar`], native precision.

mod decompose;
mod decomposition;

pub use decomposition::{col_piv_qr, ColPivQrDecomposition};
