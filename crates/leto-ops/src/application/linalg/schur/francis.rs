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
//! (LAPACK `dlahqr`'s small-subdiagonal test with the Ahues–Tisseur
//! refinement), splitting off a 1×1 (real eigenvalue) or 2×2 (conjugate pair)
//! block, and recurses on the leading submatrix. `dlahqr`'s exceptional
//! shifts every ten stalled iterations break the rare non-convergent cycles.
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

use crate::application::linalg::scaling::KernelWindow;
use crate::domain::real::RealScalar;
use leto::{LetoError, Result};

/// Iteration cap per deflation before declaring non-convergence (Wilkinson +
/// exceptional shifts converge in `O(n)` steps; this is a safety bound, and
/// the backward-error bounds of the test suite are taken at it).
const MAX_ITER: usize = 2000;

/// Minimum left-apply column span at which the vectorized row-oriented sweep
/// (two contiguous `axpy_slice` passes) overtakes the per-column scalar sweep.
/// Below it the SIMD dispatch and extra `w` traversal are not amortized; the
/// active block narrows as the iteration deflates, so most applies on small
/// matrices stay scalar. Derived empirically (f64 AVX2: crossover ≈ 32 columns).
const SPAN_SIMD_MIN: usize = 32;

/// `⌈log₂ n⌉`, the exponent of [`deflation_floor`] over `safmin`.
pub(super) fn deflation_floor_log2(n: usize) -> i32 {
    crate::application::linalg::thresholds::ceil_log2_count(n)
}

/// The absolute deflation threshold of an order-`n` run,
/// `2^⌈log₂ n⌉·safmin ≥ n·safmin`.
///
/// LAPACK `dlahqr` deflates `|h_{k,k−1}| ≤ smlnum` with
/// `smlnum = safmin·(nh/ulp)`. That form assumes `ulp² ≫ safmin`, which
/// fails in `F16` (`ε² = 2⁻²⁰ < safmin = 2⁻¹⁴`): there it is `≈ 0.19` for
/// `n = 3`, deflating at unit scale. The floor keeps `dlahqr`'s purpose —
/// a subdiagonal driven into the subnormals, where no relative test can
/// only ever be met by an exact zero, still deflates — at `n·safmin`
/// (LAPACK `dbdsqr`'s `unfl`-based form), and the matrix-tier gate
/// (`schur/mod.rs`) keeps it below `ε·‖A‖_max`.
pub(super) fn deflation_floor<T: RealScalar>(n: usize) -> T {
    crate::application::linalg::thresholds::safe_min::<T>().scale_binary(deflation_floor_log2(n))
}

/// The kernel window of the stack reflector, computed once per run:
/// `vᵀv ≤ 12m²`, degree 2 with bound `2⁴`.
fn step_window<T: RealScalar>() -> KernelWindow<T> {
    KernelWindow::new(2, 2, 4)
}

/// Per-run state a Francis step reuses: the left-apply accumulator
/// (`apply_left`'s `scratch`, sized `n`) and the step's [`KernelWindow`].
struct StepWorkspace<'a, T> {
    scratch: &'a mut [T],
    window: KernelWindow<T>,
}

struct StackReflector<T> {
    v: [T; 3],
    len: usize,
    beta: T,
}

/// The reflector mapping the stack vector `x` (length 2 or 3) to `α·e₁`;
/// returns `(reflector, α)`.
///
/// Scale-safe (LAPACK `dlarfg` via `dlapy2`/`dnrm2`): with `m = max|xᵢ|`,
/// `‖x‖² ≤ 3m²` and `vᵀv ≤ 4‖x‖² ≤ 12m²` (degree 2, bound `2⁴`). They are
/// formed unscaled while `m` keeps them normal and finite (`window`,
/// [`step_window`]); otherwise `x` is first divided by the
/// power of two bringing `m` into `[1, 2)`. The reflector `β·v·vᵀ` is
/// invariant under `v ← 2⁻ᵏv`, `β ← 2²ᵏβ` (both exact), and `α` is
/// multiplied back, so inside the window the result is bit-for-bit the
/// unscaled one. The window's upper end also bounds the unscaled
/// `‖v‖₂ ≤ 2‖x‖₂ ≤ 2√3·√(Ω/16) < √Ω` the applications multiply into `H` —
/// the degree-2 matrix-tier gate in `schur/mod.rs` keeps `‖H‖_F ≤ √Ω`, so
/// `vᵀ·H` stays finite.
fn stack_reflector<T: RealScalar>(
    x: &[T],
    window: KernelWindow<T>,
) -> Option<(StackReflector<T>, T)> {
    let len = x.len();
    if len == 0 || len > 3 {
        return None;
    }
    let exponent = window.exponent(x);
    let mut v = [T::ZERO; 3];
    for (slot, &xi) in v.iter_mut().zip(x) {
        *slot = xi.scale_binary(-exponent);
    }
    let mut norm_sq = T::ZERO;
    for &xi in &v[..len] {
        norm_sq = norm_sq.add(xi.mul(xi));
    }
    let norm = norm_sq.sqrt();
    if norm <= T::ZERO {
        return None;
    }

    let sign = if v[0] < T::ZERO {
        T::ZERO.sub(T::ONE)
    } else {
        T::ONE
    };
    let alpha = T::ZERO.sub(sign.mul(norm)); // α = −sign·‖x‖
    v[0] = v[0].sub(alpha); // v₀ = x₀ − α

    let mut vnorm_sq = T::ZERO;
    for &vi in &v[..len] {
        vnorm_sq = vnorm_sq.add(vi.mul(vi));
    }
    if vnorm_sq <= T::ZERO {
        return None;
    }
    let beta = T::ONE.add(T::ONE).div(vnorm_sq);
    Some((
        StackReflector { v, len, beta },
        alpha.scale_binary(exponent),
    ))
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

/// Which shift pair a Francis step uses — LAPACK `dlahqr`'s choice by the
/// iteration count since the last deflation (`KDEFL`, `KEXSH = 10`).
#[derive(Clone, Copy)]
enum Shift {
    /// The eigenvalues of the trailing 2×2 block.
    Wilkinson,
    /// Exceptional, from the top of the active block (`KDEFL ≡ 10 mod 20`).
    ExceptionalTop,
    /// Exceptional, from the bottom of the active block (`KDEFL ≡ 0 mod 20`).
    ExceptionalBottom,
}

impl Shift {
    /// `dlahqr`: every `2·KEXSH` iterations an exceptional shift from the
    /// bottom, every other `KEXSH` one from the top.
    fn for_iteration(iteration: usize) -> Self {
        const KEXSH: usize = 10;
        if iteration.is_multiple_of(2 * KEXSH) {
            Self::ExceptionalBottom
        } else if iteration.is_multiple_of(KEXSH) {
            Self::ExceptionalTop
        } else {
            Self::Wilkinson
        }
    }
}

/// `dlahqr`'s shift pair `(rt1r, rt1i, rt2r, rt2i)` from the 2×2
/// `[[h11, h12], [h21, h22]]`, computed on the block divided by
/// `s = |h11| + |h12| + |h21| + |h22|` (so no product over- or underflows).
/// A complex pair is returned as is; two real shifts are replaced by the one
/// nearer `h22`, used twice.
fn shift_pair<T: RealScalar>(h11: T, h12: T, h21: T, h22: T) -> (T, T, T, T) {
    let s = h11.abs().add(h12.abs()).add(h21.abs()).add(h22.abs());
    if s == T::ZERO {
        return (T::ZERO, T::ZERO, T::ZERO, T::ZERO);
    }
    let (h11, h12, h21, h22) = (h11.div(s), h12.div(s), h21.div(s), h22.div(s));
    let half = T::from_f64(0.5);
    let tr = h11.add(h22).mul(half);
    let det = h11.sub(tr).mul(h22.sub(tr)).sub(h12.mul(h21));
    let rtdisc = det.abs().sqrt();
    if det >= T::ZERO {
        let re = tr.mul(s);
        let im = rtdisc.mul(s);
        (re, im, re, im.neg())
    } else {
        let rt1r = tr.add(rtdisc);
        let rt2r = tr.sub(rtdisc);
        let nearer = if rt1r.sub(h22).abs() <= rt2r.sub(h22).abs() {
            rt1r
        } else {
            rt2r
        };
        let re = nearer.mul(s);
        (re, T::ZERO, re, T::ZERO)
    }
}

/// One Francis double-shift step on the active block `[lo, hi]` (`hi − lo ≥ 2`),
/// updating `h` (the Hessenberg matrix) and `z` (the accumulated similarity).
///
/// The implicit shift forms the first column of `(H − μ₁I)(H − μ₂I)` from the
/// shift pair `μ₁, μ₂` ([`shift_pair`], exceptional pairs by [`Shift`]), then
/// chases the resulting bulge down the band with size-3 (and a final size-2)
/// Householder reflectors — a single orthogonal similarity equal to one
/// double-shifted QR step (the implicit-Q theorem).
fn francis_step<T: RealScalar, const ACCUMULATE_Q: bool>(
    h: &mut [T],
    z: &mut [T],
    lo: usize,
    hi: usize,
    n: usize,
    shift: Shift,
    workspace: &mut StepWorkspace<'_, T>,
) {
    let window = workspace.window;
    let scratch = &mut *workspace.scratch;
    // LAPACK `dlahqr` (3.10): exceptional shifts use `DAT1 = 3/4`,
    // `DAT2 = −0.4375` around a diagonal entry.
    let dat1 = T::from_f64(0.75);
    let dat2 = T::from_f64(-0.4375);
    let (h11, h12, h21, h22) = match shift {
        Shift::ExceptionalBottom => {
            let s = at(h, hi, hi - 1, n)
                .abs()
                .add(at(h, hi - 1, hi - 2, n).abs());
            let h11 = dat1.mul(s).add(at(h, hi, hi, n));
            (h11, dat2.mul(s), s, h11)
        }
        Shift::ExceptionalTop => {
            let s = at(h, lo + 1, lo, n)
                .abs()
                .add(at(h, lo + 2, lo + 1, n).abs());
            let h11 = dat1.mul(s).add(at(h, lo, lo, n));
            (h11, dat2.mul(s), s, h11)
        }
        Shift::Wilkinson => (
            at(h, hi - 1, hi - 1, n),
            at(h, hi - 1, hi, n),
            at(h, hi, hi - 1, n),
            at(h, hi, hi, n),
        ),
    };
    let (rt1r, rt1i, rt2r, rt2i) = shift_pair(h11, h12, h21, h22);

    // First column of `(H − μ₁I)(H − μ₂I)` at the block top, as `dlahqr`
    // forms it: divided by `S = |h00 − μ₂| + |Im μ₂| + |h10|` before any
    // product, then normalized by `|v₁| + |v₂| + |v₃|` — every product is of
    // ratios, so none over- or underflows for entries anywhere in the range.
    // Only the direction enters the first reflector (its `α` is not written
    // back).
    let h00 = at(h, lo, lo, n);
    let h01 = at(h, lo, lo + 1, n);
    let h10 = at(h, lo + 1, lo, n);
    let h11b = at(h, lo + 1, lo + 1, n);
    let h21b = at(h, lo + 2, lo + 1, n);
    let s = h00.sub(rt2r).abs().add(rt2i.abs()).add(h10.abs());
    let h21s = h10.div(s);
    let mut x = h21s
        .mul(h01)
        .add(h00.sub(rt1r).mul(h00.sub(rt2r).div(s)))
        .sub(rt1i.mul(rt2i.div(s)));
    let mut y = h21s.mul(h00.add(h11b).sub(rt1r).sub(rt2r));
    let mut zz = h21s.mul(h21b);
    let norm1 = x.abs().add(y.abs()).add(zz.abs());
    if norm1 > T::ZERO {
        x = x.div(norm1);
        y = y.div(norm1);
        zz = zz.div(norm1);
    }

    for k in lo..=(hi - 1) {
        let len = if k < hi - 1 { 3 } else { 2 };
        let refl_opt = if len == 3 {
            let arr = [x, y, zz];
            stack_reflector(&arr, window)
        } else {
            let arr = [x, y, T::ZERO];
            stack_reflector(&arr[..2], window)
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

/// LAPACK `dlahqr`'s small-subdiagonal test for `h_{k,k−1}`: negligible when `|h_{k,k−1}| ≤ floor`
/// ([`deflation_floor`], in place of `dlahqr`'s `SMLNUM`), or when it passes
/// the ulp-relative pre-check `|h_{k,k−1}| ≤ ulp·tst` (`tst = |h_{k−1,k−1}| +
/// |h_{k,k}|`) **and** the Ahues–Tisseur test (Ahues & Tisseur 1997, LAPACK
/// Working Note 122): with `ab = max(|h_{k,k−1}|, |h_{k−1,k}|)`,
/// `ba = min(…)`, `aa = max(|h_{k,k}|, |h_{k−1,k−1} − h_{k,k}|)`,
/// `bb = min(…)`, `s = aa + ab`,
/// `ba·(ab/s) ≤ max(floor, ulp·(bb·(aa/s)))`. `ulp = ε` (`dlamch('P')`).
/// `dlahqr`'s fallback to the neighbouring subdiagonals when `tst = 0` is not
/// taken: there only the floor deflates, and the next step changes the
/// block (no test input distinguishes the two).
fn negligible_subdiagonal<T: RealScalar>(h: &[T], n: usize, k: usize, ulp: T, floor: T) -> bool {
    let sub = at(h, k, k - 1, n).abs();
    if sub <= floor {
        return true;
    }
    let tst = at(h, k - 1, k - 1, n).abs().add(at(h, k, k, n).abs());
    if sub > ulp.mul(tst) {
        return false;
    }
    let sup = at(h, k - 1, k, n).abs();
    let (ab, ba) = if sub > sup { (sub, sup) } else { (sup, sub) };
    let hkk = at(h, k, k, n).abs();
    let gap = at(h, k - 1, k - 1, n).sub(at(h, k, k, n)).abs();
    let (aa, bb) = if hkk > gap { (hkk, gap) } else { (gap, hkk) };
    let s = aa.add(ab);
    let relative = ulp.mul(bb.mul(aa.div(s)));
    ba.mul(ab.div(s)) <= if floor > relative { floor } else { relative }
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
    let floor = deflation_floor::<T>(n);
    let ulp = crate::application::linalg::thresholds::machine_epsilon::<T>();
    let mut workspace = StepWorkspace {
        scratch,
        window: step_window(),
    };
    let mut hi = n - 1;
    let mut iter = 0usize;
    loop {
        // Bottom-most unreduced block: scan up while the subdiagonal is
        // non-negligible ([`negligible_subdiagonal`]).
        let mut lo = hi;
        while lo > 0 {
            if negligible_subdiagonal(h, n, lo, ulp, floor) {
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
        francis_step::<T, ACCUMULATE_Q>(
            h,
            z,
            lo,
            hi,
            n,
            Shift::for_iteration(iter),
            &mut workspace,
        );
    }
    Ok(())
}
