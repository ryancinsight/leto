//! Eigenvalues of a general (non-symmetric) real matrix, real and complex.
//!
//! # Theorem (Schur form ⇒ eigenvalues on the diagonal)
//! Every `A ∈ ℂⁿˣⁿ` is unitarily similar to an upper-triangular `T`
//! (`A = U T Uᴴ`, Schur). *Proof:* induct on `n`; pick an eigenpair `Av = λv`
//! with `‖v‖=1`, extend `v` to a unitary `U₁`; then `U₁ᴴ A U₁ = [[λ, ∗],[0, B]]`
//! and `B` is `(n−1)×(n−1)`; apply the hypothesis to `B`. ∎ Because `T` is
//! triangular and similarity preserves the spectrum, **the eigenvalues of `A`
//! are exactly the diagonal entries of `T`**.
//!
//! Algorithm (ADR 0006, Phase 2): reduce to upper Hessenberg
//! ([`super::hessenberg`], reused — SSOT), promote to complex, then run the
//! **single-shift Wilkinson QR iteration** ([`qr`]). Each shifted step
//! `H ← Qᴴ H Q` with `QR = H − μI` is a unitary similarity (preserving the
//! spectrum); the Wilkinson shift gives generically cubic convergence, driving a
//! trailing subdiagonal to zero so one eigenvalue **deflates** at a time. A 2×2
//! block is resolved by the closed-form quadratic, so real *and* complex
//! conjugate pairs are produced without leaving real input behind.
//!
//! Leaf modules: [`complex`] (the `Cplx<T>` compute type) and [`qr`] (the
//! iteration). Generic over [`RealScalar`]; complex values appear only in the
//! result. For symmetric inputs prefer the dedicated Jacobi solver
//! ([`super::eigen`]), which is faster and returns sorted real eigenvalues.

mod complex;
mod qr;

use crate::domain::real::RealScalar;
use leto::{ArrayView2, LetoError, Result};
use num_complex::Complex;

/// Compute all eigenvalues of a square real matrix (real and complex).
///
/// Returns the spectrum in deflation order (bottom-up); callers needing a
/// canonical order should sort. Real eigenvalues have zero imaginary part.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] for non-square input;
/// [`LetoError::StorageError`] for non-finite input or QR non-convergence.
pub fn eigenvalues<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Vec<Complex<T>>> {
    let [rows, cols] = matrix.shape();
    if rows != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows, rows],
        });
    }
    if rows == 0 {
        return Ok(Vec::new());
    }

    // Reuse the Householder Hessenberg reduction (validates finiteness too).
    let hess = crate::hessenberg(matrix)?;
    let h_real = hess.h();
    let n = rows;
    let mut h = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            h.push(complex::Cplx::real(*h_real.get([i, j])?));
        }
    }

    let spectrum = qr::run_qr(&mut h, n)?;
    Ok(spectrum
        .into_iter()
        .map(|z| Complex::new(z.re, z.im))
        .collect())
}
