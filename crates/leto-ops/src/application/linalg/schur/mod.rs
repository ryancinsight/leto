//! Real Schur decomposition `A = Q T Qᵀ` via the Francis double-shift QR
//! algorithm (leto `Schur` parity).
//!
//! Unlike [`eigenvalues`](fn@crate::eigenvalues) — which promotes to complex and
//! returns only the spectrum — this routine stays in **real arithmetic** and
//! returns both the orthogonal `Q` and the real quasi-upper-triangular `T`,
//! i.e. the Schur *vectors*.
//!
//! # Theorem (real Schur decomposition)
//! For every `A ∈ ℝⁿˣⁿ` there exist an orthogonal `Q ∈ ℝⁿˣⁿ` and a block
//! upper-triangular `T ∈ ℝⁿˣⁿ` — with 1×1 blocks for real eigenvalues and 2×2
//! blocks (each carrying a complex-conjugate eigenvalue pair) — such that
//! `A = Q T Qᵀ`.
//!
//! *Proof (algorithmic).* Reduce `A` to upper Hessenberg `H = U₀ᵀ A U₀` with
//! orthogonal `U₀` (Householder; reused via [`hessenberg`](fn@crate::hessenberg)).
//! Each Francis double-shift step replaces `H` by `Zₖᵀ H Zₖ` for an orthogonal
//! `Zₖ` that equals one step of unshifted QR applied to
//! `(H − μ₁I)(H − μ₂I)` — the implicit-Q theorem guarantees the bulge-chasing
//! similarity is that QR step. The shifts `μ₁, μ₂` are the eigenvalues of the
//! trailing 2×2 block (a real pair or a conjugate pair), so the iteration stays
//! real and drives a trailing subdiagonal entry to zero, deflating a 1×1 or 2×2
//! block. Accumulating `Q = U₀ Z₁ Z₂ ⋯` gives `A = Q T Qᵀ` with `T` the limiting
//! quasi-triangular matrix; a final rotation splits any 2×2 block with real
//! eigenvalues. Orthogonality of `Q` is preserved because every factor is
//! orthogonal. ∎
//!
//! # Corollary (spectrum)
//! The eigenvalues of `A` are the eigenvalues of the diagonal blocks of `T`:
//! each 1×1 block is a real eigenvalue, each 2×2 block contributes a conjugate
//! pair (its quadratic). Similarity preserves the spectrum.
//!
//! Leaf modules: `decomposition` (the [`RealSchur`] type and [`schur()`] entry),
//! `francis` (the double-shift iteration), and `standardize` (2×2 real-block
//! splitting). Generic over [`crate::RealScalar`], native precision.
//!
//! Evidence tier: theorem/proof sketch in rustdoc plus value-semantic tests for
//! the exact reconstruction `A = Q T Qᵀ`, `Q` orthogonality, quasi-triangular
//! structure (2×2 blocks only for complex pairs), and eigenvalue agreement with
//! both [`eigenvalues`](fn@crate::eigenvalues) and leto across real and complex
//! spectra.

mod decomposition;
mod francis;
mod standard_block;
mod standardize;
#[cfg(test)]
mod tests;

#[cfg(test)]
use decomposition::francis_bound;
pub(crate) use decomposition::real_eigenvalues;
pub use decomposition::{schur, RealSchur};
