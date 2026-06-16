//! Real Schur decomposition `A = Q T Qᵀ` via the Francis double-shift QR
//! algorithm (nalgebra `Schur` parity).
//!
//! Unlike [`eigenvalues`](crate::eigenvalues) — which promotes to complex and
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
//! orthogonal `U₀` (Householder; reused via [`hessenberg`](crate::hessenberg)).
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
//! Leaf modules: `francis` (the double-shift iteration) and `standardize` (2×2
//! real-block splitting). Generic over [`crate::RealScalar`], native precision.
//!
//! Evidence tier: theorem/proof sketch in rustdoc plus value-semantic tests for
//! the exact reconstruction `A = Q T Qᵀ`, `Q` orthogonality, quasi-triangular
//! structure (2×2 blocks only for complex pairs), and eigenvalue agreement with
//! both [`eigenvalues`](crate::eigenvalues) and nalgebra across real and complex
//! spectra.

mod francis;
mod standardize;

use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, LetoError, Result, Storage};
use num_complex::Complex;

/// Real Schur decomposition `A = Q T Qᵀ`.
#[derive(Debug, Clone)]
pub struct RealSchur<T> {
    q: Vec<T>,
    t: Vec<T>,
    n: usize,
}

/// Compute the real Schur decomposition of a square real matrix.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] for non-square input;
/// [`LetoError::StorageError`] for non-finite input or QR non-convergence.
pub fn schur<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<RealSchur<T>> {
    let [rows, cols] = matrix.shape();
    if rows != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows, rows],
        });
    }
    let n = rows;

    // Reduce to Hessenberg (validates finiteness; reused — SSOT). `H = Qᴴᵀ A Qᴴ`.
    let hess = crate::hessenberg(matrix)?;
    let mut t: Vec<T> = hess.h().storage().as_slice().to_vec();
    let mut q: Vec<T> = hess.q().storage().as_slice().to_vec();

    francis::run(&mut t, &mut q, n)?;
    standardize::standardize(&mut t, &mut q, n);

    Ok(RealSchur { q, t, n })
}

impl<T: RealScalar> RealSchur<T> {
    /// The orthogonal Schur-vector matrix `Q` (`n × n`).
    #[must_use]
    pub fn q(&self) -> Array2<T> {
        Array2::from_shape_vec([self.n, self.n], self.q.clone()).expect("Q shape matches storage")
    }

    /// The real quasi-upper-triangular factor `T` (`n × n`).
    #[must_use]
    pub fn t(&self) -> Array2<T> {
        Array2::from_shape_vec([self.n, self.n], self.t.clone()).expect("T shape matches storage")
    }

    /// Eigenvalues read off the diagonal blocks of `T` (real and complex).
    #[must_use]
    pub fn eigenvalues(&self) -> Vec<Complex<T>> {
        let n = self.n;
        let mut eigs = Vec::with_capacity(n);
        let mut i = 0usize;
        while i < n {
            let is_block = i + 1 < n && self.t[(i + 1) * n + i] != T::ZERO;
            if is_block {
                let a = self.t[i * n + i];
                let b = self.t[i * n + i + 1];
                let c = self.t[(i + 1) * n + i];
                let d = self.t[(i + 1) * n + i + 1];
                let tr = a.add(d);
                let det = a.mul(d).sub(b.mul(c));
                let half = T::from_f64(0.5);
                let four = T::from_f64(4.0);
                let disc = tr.mul(tr).sub(four.mul(det));
                if disc < T::ZERO {
                    let re = tr.mul(half);
                    let im = disc.neg().sqrt().mul(half);
                    eigs.push(Complex::new(re, im));
                    eigs.push(Complex::new(re, im.neg()));
                } else {
                    let root = disc.sqrt();
                    eigs.push(Complex::new(tr.add(root).mul(half), T::ZERO));
                    eigs.push(Complex::new(tr.sub(root).mul(half), T::ZERO));
                }
                i += 2;
            } else {
                eigs.push(Complex::new(self.t[i * n + i], T::ZERO));
                i += 1;
            }
        }
        eigs
    }
}
