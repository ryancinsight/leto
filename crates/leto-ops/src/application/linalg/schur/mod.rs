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
//! Leaf modules: `francis` (the double-shift iteration) and `standardize` (2×2
//! real-block splitting). Generic over [`crate::RealScalar`], native precision.
//!
//! Evidence tier: theorem/proof sketch in rustdoc plus value-semantic tests for
//! the exact reconstruction `A = Q T Qᵀ`, `Q` orthogonality, quasi-triangular
//! structure (2×2 blocks only for complex pairs), and eigenvalue agreement with
//! both [`eigenvalues`](fn@crate::eigenvalues) and leto across real and complex
//! spectra.

mod francis;
mod standardize;

use crate::application::linalg::scaling;
use crate::domain::real::RealScalar;
use leto::Complex;
use leto::{Array2, ArrayView2, LetoError, Result, Storage};

/// Degree and dimension factor for the Francis double-shift reflector
/// (`schur::francis::stack_reflector`, fed by the `x`/`y`/`zz` formed at
/// `francis.rs`'s double-shift step): an orthogonal similarity preserves the
/// 2-norm, so every Hessenberg entry is bounded by `‖A‖_2 ≤ ‖A‖_F ≤ n·‖A‖_max`.
/// `x = h00·h00 + h01·h10 − s·h00 + t` sums four such degree-2 terms
/// (`s`, the local trace, is itself `≤ 2n·‖A‖_max`; `t`, the local
/// determinant, `≤ 2n²·‖A‖_max²`), giving `|x| ≤ 6n²·‖A‖_max²`; `y`, `zz` are
/// bounded the same way. The reflector then forms `x² + y² + zz²`
/// (`stack_reflector`'s `norm_sq`) — a **second** squaring, so the relied-upon
/// intermediate is degree 4 in the original entries, bounded by
/// `3·(6n²)²·‖A‖_max⁴ = 108n⁴·‖A‖_max⁴`; rounded up to `128n⁴` for margin.
fn francis_dimension_factor<T: RealScalar>(n: usize) -> T {
    let n = T::from_usize(n.max(1));
    let n2 = n.mul(n);
    let n4 = n2.mul(n2);
    T::from_usize(128).mul(n4)
}

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
    if n == 0 {
        return Ok(RealSchur {
            q: vec![],
            t: vec![],
            n: 0,
        });
    }

    // Balance by an exact power of two; `Q` is scale-invariant, `T` scales.
    let balanced = scaling::balanced_recentered(matrix, 4, francis_dimension_factor::<T>(n));
    let (view, exponent) = match &balanced {
        Some((scaled, exponent)) => (scaled.view(), *exponent),
        None => (*matrix, 0),
    };

    // Reduce to Hessenberg (validates finiteness; reused — SSOT). `H = Qᴴᵀ A Qᴴ`.
    let hess = crate::hessenberg(&view)?;
    let mut t: Vec<T> = hess.h().storage().as_slice().to_vec();
    let mut q: Vec<T> = hess.q().storage().as_slice().to_vec();

    francis::run::<T, true>(&mut t, &mut q, n)?;
    standardize::standardize(&mut t, &mut q, n);
    scaling::restore(
        &mut t,
        exponent,
        "Schur form entry exceeds the scalar range",
    )?;

    Ok(RealSchur { q, t, n })
}

/// Eigenvalues of a square real matrix, **without** forming the Schur vectors.
///
/// Reduces to Hessenberg, runs the Francis double-shift QR with Q-accumulation
/// disabled (the `apply_right(z, …)` similarity update is DCE'd at
/// monomorphization — zero cost), and reads the eigenvalues off the resulting
/// quasi-triangular blocks. Standardization is skipped because the eigenvalues of
/// a 2×2 block are extracted from its quadratic regardless of whether the block
/// is triangularized. This is the fast path backing
/// [`eigenvalues`](crate::eigenvalues); the full [`schur`] retains the `Q` path.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] for non-square input;
/// [`LetoError::StorageError`] for non-finite input or QR non-convergence.
pub(crate) fn real_eigenvalues<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Vec<Complex<T>>> {
    let [rows, cols] = matrix.shape();
    if rows != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows, rows],
        });
    }
    let n = rows;
    if n == 0 {
        return Ok(Vec::new());
    }
    // Eigenvalues-only: reduce to Hessenberg without accumulating Q (similarity
    // invariance means the Schur vectors are never needed), saving the O(n³) Q
    // update. Mirrors the `ACCUMULATE_Q = false` Francis stage below.
    // Balance by an exact power of two; eigenvalues scale with the matrix.
    let balanced = scaling::balanced_recentered(matrix, 4, francis_dimension_factor::<T>(n));
    let (view, exponent) = match &balanced {
        Some((scaled, exponent)) => (scaled.view(), *exponent),
        None => (*matrix, 0),
    };
    let (mut h, hn) = crate::application::linalg::hessenberg::hessenberg_values(&view)?;
    debug_assert_eq!(hn, n);
    // No Schur vectors: pass an empty accumulator; the const-generic guarantees
    // it is never touched.
    let mut unused: [T; 0] = [];
    francis::run::<T, false>(&mut h, &mut unused, n)?;
    let mut eigenvalues = eigenvalues_from_quasi_triangular(&h, n);
    if exponent != 0 {
        let mut parts: Vec<T> = eigenvalues.iter().flat_map(|z| [z.re, z.im]).collect();
        scaling::restore(&mut parts, exponent, "eigenvalue exceeds the scalar range")?;
        for (z, pair) in eigenvalues.iter_mut().zip(parts.chunks_exact(2)) {
            *z = Complex::new(pair[0], pair[1]);
        }
    }
    Ok(eigenvalues)
}

/// Read the eigenvalues off a real quasi-upper-triangular matrix `t` (`n × n`):
/// each 1×1 block is a real eigenvalue, each 2×2 block (nonzero subdiagonal) a
/// conjugate pair from its quadratic. Shared by [`RealSchur::eigenvalues`] and
/// [`real_eigenvalues`] (SSOT).
///
/// The quadratic squares the block's trace, which underflows or overflows for
/// entries near the ends of the range — a block at `2⁻⁸⁶` in `f32` squares to
/// `2⁻¹⁷²`, below the smallest subnormal, and returned `{3, 3}` for `{2, 4}`.
/// Each block is therefore balanced by its own even power of two
/// (`linalg::scaling`) before the quadratic and its eigenvalues restored after, both
/// exactly.
pub(crate) fn eigenvalues_from_quasi_triangular<T: RealScalar>(
    t: &[T],
    n: usize,
) -> Vec<Complex<T>> {
    let mut eigs = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        let is_block = i + 1 < n && t[(i + 1) * n + i] != T::ZERO;
        if is_block {
            let mut block = [
                t[i * n + i],
                t[i * n + i + 1],
                t[(i + 1) * n + i],
                t[(i + 1) * n + i + 1],
            ];
            // Degree 2, dimension factor 12: `tr = a+d`, `det = ad-bc` are
            // each degree-2 bounded by `2*block_max^2`; the discriminant
            // `tr^2 - 4*det` sums those, giving `|disc| <= 4*block_max^2 +
            // 8*block_max^2 = 12*block_max^2` (a fixed 2x2, no n-dependence).
            let exponent = scaling::balancing_exponent(block, 2, T::from_usize(12)).unwrap_or(0);
            scaling::scale_by_power_of_two(&mut block, -exponent);
            let [a, b, c, d] = block;
            let restore = |x: T| x.scale_binary(exponent);
            let tr = a.add(d);
            let det = a.mul(d).sub(b.mul(c));
            let half = T::from_f64(0.5);
            let four = T::from_f64(4.0);
            let disc = tr.mul(tr).sub(four.mul(det));
            if disc < T::ZERO {
                let re = restore(tr.mul(half));
                let im = restore(disc.neg().sqrt().mul(half));
                eigs.push(Complex::new(re, im));
                eigs.push(Complex::new(re, im.neg()));
            } else {
                let root = disc.sqrt();
                eigs.push(Complex::new(restore(tr.add(root).mul(half)), T::ZERO));
                eigs.push(Complex::new(restore(tr.sub(root).mul(half)), T::ZERO));
            }
            i += 2;
        } else {
            eigs.push(Complex::new(t[i * n + i], T::ZERO));
            i += 1;
        }
    }
    eigs
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
        eigenvalues_from_quasi_triangular(&self.t, self.n)
    }
}
