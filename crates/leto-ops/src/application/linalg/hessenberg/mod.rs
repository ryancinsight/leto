//! Upper Hessenberg reduction `A = Q H Qᵀ` by Householder reflectors.
//!
//! An upper Hessenberg matrix is zero below the first subdiagonal
//! (`hᵢⱼ = 0` for `i > j + 1`). Reducing to Hessenberg form is the standard
//! first phase of the non-symmetric eigenvalue problem: the subsequent Francis
//! QR iteration costs `O(n²)` per step on a Hessenberg matrix instead of `O(n³)`
//! on a full one, and the form is preserved by the iteration.
//!
//! # Theorem (Hessenberg reduction)
//! Every `A ∈ ℝⁿˣⁿ` admits `A = Q H Qᵀ` with `Q` orthogonal and `H` upper
//! Hessenberg. *Proof (constructive):* a Householder reflector
//! `Pₖ = I − βₖ vₖ vₖᵀ` is symmetric and orthogonal; choosing `vₖ` from the
//! sub-column `A[k+1.., k]` maps it onto `e₁`, zeroing `A[k+2.., k]`. Applying
//! `Pₖ` on *both* sides, `A ← Pₖ A Pₖ`, is an orthogonal **similarity**, so it
//! leaves the spectrum unchanged; crucially the right multiplication only mixes
//! columns `k+1..n`, which does not refill the zeros just created in column `k`.
//! After `k = 0 … n−3` the result is upper Hessenberg, with
//! `Q = P₀ P₁ … P_{n−3}` orthogonal (a product of orthogonal matrices). ∎
//!
//! # Corollary (spectrum preservation)
//! `H = Qᵀ A Q` is similar to `A`, so `H` and `A` have the same eigenvalues; in
//! particular `tr(H) = tr(A)` and `‖H‖_F = ‖A‖_F` (orthogonal invariance).
//!
//! Uses the shared `householder` reflector primitive
//! (SSOT); `reduce` is the reduction loop. Generic over [`crate::RealScalar`], native
//! precision throughout.

mod decomposition;
mod reduce;

pub(crate) use decomposition::hessenberg_values;
pub use decomposition::{hessenberg, HessenbergDecomposition};
