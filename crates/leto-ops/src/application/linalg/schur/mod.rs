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
#[cfg(test)]
mod tests;

use crate::application::linalg::scaling::{self, GateBound, KernelWindow};
use crate::domain::real::RealScalar;
use leto::Complex;
use leto::{Array2, ArrayView2, LetoError, Result, Storage};

/// The matrix-tier gate for the Francis family (`schur`, `eigenvalues`):
/// degree 2, bound `2^(2r)`, `r = ⌈log₂(‖A‖_F/‖A‖_max)⌉`
/// ([`scaling::norm_ratio_log2`]).
///
/// The kernels form every local product scale-safely (`francis.rs`'s first
/// column and `stack_reflector`, `standardize.rs`, the 2×2 eigenvalue
/// quadratic below, the Hessenberg reflector). What remains is the
/// reflector application `w = vᵀ·H[rows, cols]` (`francis::apply_left`,
/// `apply_right`): `stack_reflector` leaves `v` unscaled up to its window's
/// upper end, `‖v‖₂ ≤ √(12·Ω/16) < √Ω`, and every column of `H` has 2-norm at
/// most `‖A‖₂ ≤ ‖A‖_F ≤ 2^r·‖A‖_max` (orthogonal similarity), so
/// `|w| ≤ √Ω·2^r·‖A‖_max` — finite when `2^(2r)·‖A‖_max² ≤ Ω`: a degree-2
/// product of an unscaled reflector (itself up to a square of entries) with
/// an entry. The degree-1 sums (the Hessenberg reduction's `vᵀ·A` with
/// `‖v‖₂ ≤ 4√n`, `householder::reflect_in_place`; the deflation test's
/// `|hᵢᵢ| + |hᵢ₊₁ᵢ₊₁|`) are at most `4√n·√Ω` inside this range, finite for
/// every order `n ≤ Ω/16` a format can index.
///
/// Deflation floor: `francis::run` also deflates a subdiagonal at or below
/// [`francis::deflation_floor`]`(n) = 2^⌈log₂ n⌉·safmin`, so the gate's lower
/// end is raised to keep that floor below `ε·‖A‖_max`.
fn francis_bound<T: RealScalar>(n: usize) -> impl FnOnce(&[T], T) -> GateBound {
    move |values, largest| GateBound {
        factor_log2: 2 * scaling::norm_ratio_log2(values, largest),
        floor_log2: francis::deflation_floor_log2(n),
    }
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
    let balanced = scaling::balanced(matrix, 2, francis_bound(n));
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
    let balanced = scaling::balanced(matrix, 2, francis_bound(n));
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
/// Each block is therefore formed scale-safely (the kernel tier of
/// `linalg::scaling`, as LAPACK `dlanv2` guards the same quadratic): unscaled
/// while its largest entry keeps the discriminant normal and finite, and
/// otherwise divided by the power of two bringing that entry into `[1, 2)`,
/// its eigenvalues multiplied back after — both exactly.
pub(crate) fn eigenvalues_from_quasi_triangular<T: RealScalar>(
    t: &[T],
    n: usize,
) -> Vec<Complex<T>> {
    // `tr² ≤ 4m²` and `4·|det| ≤ 8m²` (`m` a block's largest entry), so
    // `|disc| ≤ 12m² < 2⁴m²`: degree 2, bound `2⁴`.
    let window = KernelWindow::new(2, 2, 4);
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
            let exponent = window.exponent(&block);
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
