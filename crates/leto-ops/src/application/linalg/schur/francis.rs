//! Francis double-shift implicit QR iteration on a real upper-Hessenberg matrix,
//! accumulating the orthogonal similarity.
//!
//! Operates entirely in real arithmetic: complex eigenvalues surface as isolated
//! 2×2 diagonal blocks (standardized later), so no complex type is needed here.
//!
//! # Theorem (Francis step = one implicit double-shift QR step)
//! Let `H` be unreduced upper Hessenberg and `μ₁, μ₂` a shift pair that is either
//! two reals or a complex-conjugate pair. One Francis step computes an orthogonal
//! `Z` with `Zᵀ H Z` again upper Hessenberg, and `Zᵀ H Z` equals the matrix
//! produced by one explicit double-shift QR step — i.e. `Q` from the QR
//! factorization `(H − μ₁I)(H − μ₂I) = Q R`, applied as `Qᵀ H Q` — without ever
//! forming the product or leaving real arithmetic.
//!
//! *Proof (implicit-Q).* `M = (H − μ₁I)(H − μ₂I)` is real: a conjugate pair gives
//! `M = H² − (μ₁+μ₂)H + μ₁μ₂I` with real coefficients `s = μ₁+μ₂`, `t = μ₁μ₂`.
//! The step builds the Householder `P₀` mapping `M e₁` (its first column,
//! computed directly from `s, t` and the top-left of `H` — the `x, y, zz` below)
//! to a multiple of `e₁`, then forms `P₀ᵀ H P₀`, which bulges `H` just below the
//! subdiagonal, and chases the bulge with Householders `P₁ … P_{n-2}` that
//! restore Hessenberg form. Set `Z = P₀ P₁ … P_{n-2}`; then `Zᵀ H Z` is upper
//! Hessenberg and `Z e₁ = P₀ e₁ ∝ M e₁`. The implicit-Q theorem states that for
//! unreduced `H`, an orthogonal `Z` with `Zᵀ H Z` unreduced Hessenberg is
//! determined, up to column signs, by `Z e₁`. The explicit step's `Q` satisfies
//! `Q e₁ ∝ M e₁` as well (first column of `Q R = M`), so `Z` and `Q` agree up to
//! signs and yield the same Hessenberg form. Hence the bulge chase realizes the
//! double-shift QR step in real arithmetic. ∎
//!
//! # Corollary (convergence and deflation)
//! With Wilkinson-type shifts (eigenvalues of the trailing 2×2 block) the bottom
//! subdiagonal entry converges quadratically to zero; the iteration zeroes it
//! (precision-exact deflation test), splitting off a 1×1 (real eigenvalue) or 2×2
//! (conjugate pair) block, and recurses on the leading submatrix. Exceptional
//! ad-hoc shifts every few stalls break the rare non-convergent cycles.
//!
//! # Theorem (eigenvalues-only within-block apply window — LAPACK `dlahqr`)
//! For the spectrum it suffices to apply each bulge-chasing reflector `Pₖ` only on
//! the window columns `[k, hi]` (left) and rows `[lo, k+len]` (right), provided the
//! annihilated bulge column `k−1` is set to its known image `(α, 0, 0)ᵀ`. The
//! eigenvalues read off the converged quasi-triangular `H` are unchanged.
//!
//! *Proof.* `H` stays similar to the original under every two-sided reflector, so
//! the spectrum is preserved regardless of which entries are stored. The
//! eigenvalues are read from the **diagonal blocks** only. An entry skipped by the
//! window is one of: (i) the bulge subdiagonal in column `k−1`, whose post-reflector
//! value is exactly `(α, 0, 0)` — written explicitly, so no information is lost; or
//! (ii) an entry with row `< lo` or column `> hi`, which is strictly above the
//! active diagonal block (`row < lo ≤ col` or `row ≤ hi < col`) and hence never
//! lies on a diagonal block, is never a shift source (shifts come from the trailing
//! 2×2 of `[lo, hi]`), and never enters the bulge band. Because `hi` is
//! non-increasing and `lo` is non-decreasing for fixed `hi` (deflation sets
//! `h[lo][lo−1]` to exact zero, a hard floor), such an entry is never read by a
//! future active block either. The window thus omits only never-read entries, so
//! the diagonal blocks — hence the eigenvalues — match the full sweep. ∎
//!
//! *Numerical note (evidence tier: differential + empirical).* The window reorders
//! the floating-point updates relative to a full sweep, so on a **defective**
//! eigenvalue (perturbation `O(√(ε‖A‖))`) the computed value can differ from a full
//! sweep — and from a backward-stable reference — by `O(√(ε‖A‖))`. This is within
//! backward stability, not an error; the eigenvalue battery asserts the derived
//! `8·√(ε‖A‖)` tolerance accordingly. The `ACCUMULATE_Q` (Schur) path keeps the
//! full sweep because `T` and the Schur vectors are outputs.

use crate::domain::real::RealScalar;
use leto::{LetoError, Result};

/// Iteration cap before declaring non-convergence (Wilkinson + exceptional
/// shifts converge in `O(n)` steps; this is a safety bound).
const MAX_ITER: usize = 2000;

/// Minimum left-apply column span at which the vectorized row-oriented sweep
/// (two contiguous `axpy_slice` passes) overtakes the per-column scalar sweep.
/// Below it the SIMD dispatch and extra `w` traversal are not amortized; the
/// active block narrows as the iteration deflates, so most applies on small
/// matrices stay scalar. Derived empirically (f64 AVX2: crossover ≈ 32 columns).
const SPAN_SIMD_MIN: usize = 32;

struct StackReflector<T> {
    v: [T; 3],
    len: usize,
    beta: T,
}

fn stack_reflector<T: RealScalar>(x: &[T]) -> Option<(StackReflector<T>, T)> {
    let len = x.len();
    if len == 0 || len > 3 {
        return None;
    }
    let mut norm_sq = T::ZERO;
    for &xi in x {
        norm_sq = norm_sq.add(xi.mul(xi));
    }
    let norm = norm_sq.sqrt();
    if norm <= T::ZERO {
        return None;
    }

    let sign = if x[0] < T::ZERO {
        T::ZERO.sub(T::ONE)
    } else {
        T::ONE
    };
    let alpha = T::ZERO.sub(sign.mul(norm)); // α = −sign·‖x‖

    let mut v = [T::ZERO; 3];
    v[..len].copy_from_slice(&x[..len]);
    v[0] = v[0].sub(alpha); // v₀ = x₀ − α

    let mut vnorm_sq = T::ZERO;
    for &vi in &v[..len] {
        vnorm_sq = vnorm_sq.add(vi.mul(vi));
    }
    if vnorm_sq <= T::ZERO {
        return None;
    }
    let beta = T::ONE.add(T::ONE).div(vnorm_sq);
    Some((StackReflector { v, len, beta }, alpha))
}

#[inline]
fn at<T: Copy>(h: &[T], i: usize, j: usize, n: usize) -> T {
    h[i * n + j]
}

/// Left-apply a Householder reflector `P = I − β v vᵀ` (positioned at base row
/// `k`, `v.len()` rows) across columns `c_lo..=c_hi`: `H ← P H`.
///
/// Row-oriented: accumulate `w = (β vᵀ)·H[rows, c_lo..=c_hi]` by sweeping each
/// reflector row contiguously into the caller-owned `scratch`, then apply
/// `H −= v·w` row by row — both inner sweeps are contiguous `axpy_slice` updates
/// (SSOT SIMD path). The per-`w[j]` summation order (reflector rows ascending)
/// and the `vᵢ·(β·w[j])` grouping match the column-oriented form exactly, so the
/// result is bitwise-identical (hermes `axpy` performs no FMA contraction); the
/// reflector spans only 2–3 rows but the column span is the active-block width,
/// where the vectorized sweep pays off. `scratch` must hold `≥ c_hi − c_lo + 1`
/// elements (the caller sizes it to `n`, reused across the whole iteration —
/// allocation-free hot path).
// Eight tight primitive parameters (matrix, reflector vector + β, base row, dim,
// column range, scratch); each is a distinct kernel input and bundling them into
// a struct would add an artificial indirection on this hot inner routine.
#[allow(clippy::too_many_arguments)]
fn apply_left<T: RealScalar>(
    h: &mut [T],
    v: &[T],
    beta: T,
    k: usize,
    n: usize,
    c_lo: usize,
    c_hi: usize,
    scratch: &mut [T],
) {
    if c_hi < c_lo {
        return;
    }
    let span = c_hi - c_lo + 1;
    if span < SPAN_SIMD_MIN {
        // Narrow span (the common case late in deflation, and every span on small
        // matrices): the per-column scalar sweep beats the vectorized two-pass —
        // the `axpy_slice` dispatch and the extra `w` traversal are not amortized
        // over so few columns. Bitwise-identical to the wide path (same per-`w[j]`
        // order and `vᵢ·(β·w[j])` grouping).
        for j in c_lo..=c_hi {
            let mut acc = T::ZERO;
            for (i, &vi) in v.iter().enumerate() {
                acc = acc.add(vi.mul(h[(k + i) * n + j]));
            }
            acc = acc.mul(beta);
            for (i, &vi) in v.iter().enumerate() {
                let cell = (k + i) * n + j;
                h[cell] = h[cell].sub(vi.mul(acc));
            }
        }
        return;
    }
    // Wide span: row-oriented, both inner sweeps contiguous `axpy_slice` (SIMD).
    let w = &mut scratch[..span];
    w.fill(T::ZERO);
    for (i, &vi) in v.iter().enumerate() {
        let base = (k + i) * n + c_lo;
        T::axpy_slice(vi, &h[base..base + span], w); // w += vᵢ · H[row, c_lo..=c_hi]
    }
    for wj in w.iter_mut() {
        *wj = beta.mul(*wj);
    }
    for (i, &vi) in v.iter().enumerate() {
        let base = (k + i) * n + c_lo;
        T::axpy_slice(T::ZERO.sub(vi), w, &mut h[base..base + span]); // H −= vᵢ · w
    }
}

/// Right-apply a Householder reflector (base column `k`) across rows
/// `r_lo..=r_hi`: `H ← H P`.
fn apply_right<T: RealScalar>(
    h: &mut [T],
    v: &[T],
    beta: T,
    k: usize,
    n: usize,
    r_lo: usize,
    r_hi: usize,
) {
    for i in r_lo..=r_hi {
        let mut w = T::ZERO;
        for (c, &vc) in v.iter().enumerate() {
            w = w.add(h[i * n + (k + c)].mul(vc));
        }
        w = w.mul(beta);
        for (c, &vc) in v.iter().enumerate() {
            let cell = i * n + (k + c);
            h[cell] = h[cell].sub(w.mul(vc));
        }
    }
}

/// One Francis double-shift step on the active block `[lo, hi]` (`hi − lo ≥ 2`),
/// updating `h` (the Hessenberg matrix) and `z` (the accumulated similarity).
///
/// The implicit shift forms the first column of `(H − μ₁I)(H − μ₂I)` from the
/// trailing-2×2 shift pair `μ₁, μ₂` (or an exceptional ad-hoc pair when the
/// iteration stalls), then chases the resulting bulge down the band with size-3
/// (and a final size-2) Householder reflectors — a single orthogonal similarity
/// equal to one double-shifted QR step (the implicit-Q theorem).
fn francis_step<T: RealScalar, const ACCUMULATE_Q: bool>(
    h: &mut [T],
    z: &mut [T],
    lo: usize,
    hi: usize,
    n: usize,
    exceptional: bool,
    scratch: &mut [T],
) {
    // Shift sum `s = μ₁ + μ₂` and product `t = μ₁ μ₂`.
    let (s, t) = if exceptional {
        // Ad-hoc Wilkinson shift to break cycles.
        let scale = at(h, hi, hi - 1, n)
            .abs()
            .add(at(h, hi - 1, hi - 2, n).abs());
        let three_halves = T::from_f64(1.5);
        (three_halves.mul(scale), scale.mul(scale))
    } else {
        let a = at(h, hi - 1, hi - 1, n);
        let d = at(h, hi, hi, n);
        let b = at(h, hi - 1, hi, n);
        let c = at(h, hi, hi - 1, n);
        (a.add(d), a.mul(d).sub(b.mul(c)))
    };

    // First column of (H² − sH + tI) restricted to the block top.
    let h00 = at(h, lo, lo, n);
    let h01 = at(h, lo, lo + 1, n);
    let h10 = at(h, lo + 1, lo, n);
    let h11 = at(h, lo + 1, lo + 1, n);
    let h21 = at(h, lo + 2, lo + 1, n);
    let mut x = h00.mul(h00).add(h01.mul(h10)).sub(s.mul(h00)).add(t);
    let mut y = h10.mul(h00.add(h11).sub(s));
    let mut zz = h10.mul(h21);

    for k in lo..=(hi - 1) {
        let len = if k < hi - 1 { 3 } else { 2 };
        let refl_opt = if len == 3 {
            let arr = [x, y, zz];
            stack_reflector(&arr)
        } else {
            let arr = [x, y, T::ZERO];
            stack_reflector(&arr[..2])
        };
        if let Some((refl, alpha)) = refl_opt {
            let v_slice = &refl.v[..refl.len];
            if ACCUMULATE_Q {
                // Schur path: the full quasi-triangular `T` and the Schur vectors
                // are outputs, so the apply spans the whole matrix.
                apply_left(h, v_slice, refl.beta, k, n, lo, n - 1, scratch);
                apply_right(h, v_slice, refl.beta, k, n, 0, hi);
                apply_right(z, v_slice, refl.beta, k, n, 0, n - 1);
            } else {
                // Eigenvalues-only: the **within-block window** (LAPACK `dlahqr`
                // WANTT=false). Only the diagonal blocks are read, so the apply is
                // confined to `[k, hi]` (left) × `[lo, k+len]` (right): entries to
                // the left of column `k` and below row `k+len` either are the
                // annihilated bulge (set explicitly below) or are off every
                // diagonal block and never feed back (`hi` only decreases; `lo` is
                // monotone non-decreasing for fixed `hi` via exact-zero deflation).
                // This is ≈ half the apply work of the `[lo, hi]²` confinement. It
                // is backward-stable but reorders rounding, so on a near-defective
                // eigenvalue it differs from a full sweep (and from the reference)
                // by `O(√(ε‖A‖))` — within the eigenvalue battery's derived
                // backward-error tolerance. Evidence tier: differential and
                // empirical validation, not machine-checked proof.
                if k > lo {
                    h[k * n + (k - 1)] = alpha;
                    h[(k + 1) * n + (k - 1)] = T::ZERO;
                    if refl.len == 3 {
                        h[(k + 2) * n + (k - 1)] = T::ZERO;
                    }
                }
                apply_left(h, v_slice, refl.beta, k, n, k, hi, scratch);
                apply_right(h, v_slice, refl.beta, k, n, lo, (k + refl.len).min(hi));
            }
        }
        if k + 1 < hi {
            x = at(h, k + 1, k, n);
            y = at(h, k + 2, k, n);
            zz = if k + 3 <= hi {
                at(h, k + 3, k, n)
            } else {
                T::ZERO
            };
        }
    }
}

/// Drive the Francis iteration to convergence: `h` becomes real
/// quasi-upper-triangular (real Schur form `T`) and `z` accumulates the
/// orthogonal similarity so that `H₀ = z T zᵀ`.
///
/// # Errors
/// [`LetoError::StorageError`] if a block fails to converge within [`MAX_ITER`].
pub(super) fn run<T: RealScalar, const ACCUMULATE_Q: bool>(
    h: &mut [T],
    z: &mut [T],
    n: usize,
) -> Result<()> {
    if n < 3 {
        return Ok(()); // 0/1: trivial; 2: a single block, standardized later.
    }
    // Reusable left-apply accumulator `w`, sized to the widest possible column
    // span (`n`); reused across every reflector so the hot path allocates once.
    let mut scratch_stack = [T::ZERO; 128];
    let mut scratch_vec = Vec::new();
    let scratch = if n <= 128 {
        &mut scratch_stack[..n]
    } else {
        scratch_vec.resize(n, T::ZERO);
        &mut scratch_vec[..]
    };
    let mut hi = n - 1;
    let mut iter = 0usize;
    loop {
        // Bottom-most unreduced block: scan up while the subdiagonal is
        // non-negligible (precision-exact `d + |sub| == d`).
        let mut lo = hi;
        while lo > 0 {
            let sub = at(h, lo, lo - 1, n).abs();
            let d = at(h, lo - 1, lo - 1, n).abs().add(at(h, lo, lo, n).abs());
            if d.add(sub) == d {
                h[lo * n + (lo - 1)] = T::ZERO;
                break;
            }
            lo -= 1;
        }

        if lo == hi {
            // 1×1 deflation (real eigenvalue).
            if hi == 0 {
                break;
            }
            hi -= 1;
            iter = 0;
            continue;
        }
        if lo == hi - 1 {
            // 2×2 deflation (real pair or complex conjugate pair).
            if lo == 0 {
                break;
            }
            hi = lo - 1;
            iter = 0;
            continue;
        }

        iter += 1;
        if iter > MAX_ITER {
            return Err(LetoError::StorageError {
                reason: "Schur QR iteration failed to converge".to_string(),
            });
        }
        francis_step::<T, ACCUMULATE_Q>(h, z, lo, hi, n, iter.is_multiple_of(10), &mut *scratch);
    }
    Ok(())
}
