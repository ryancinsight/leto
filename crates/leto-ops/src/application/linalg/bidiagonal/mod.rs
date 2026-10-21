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
mod decomposition;
mod reduce;

pub(crate) use decomposition::bidiagonal_diag_colmajor;
pub use decomposition::{bidiagonalize, BidiagonalDecomposition};
