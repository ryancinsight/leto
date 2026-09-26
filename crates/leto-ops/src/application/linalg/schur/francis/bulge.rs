//! The bulge chase: the stack reflector, its left and right applications, and
//! one Francis double-shift step.

use super::at;
use super::shift::{first_column, shift_pair, Shift};
use crate::application::linalg::scaling::KernelWindow;
use crate::domain::real::RealScalar;

/// Minimum left-apply column span at which the vectorized row-oriented sweep
/// (two contiguous `axpy_slice` passes) overtakes the per-column scalar sweep.
/// Below it the SIMD dispatch and extra `w` traversal are not amortized; the
/// active block narrows as the iteration deflates, so most applies on small
/// matrices stay scalar. Derived empirically (f64 AVX2: crossover ≈ 32 columns).
const SPAN_SIMD_MIN: usize = 32;

/// The kernel window of the stack reflector, computed once per run:
/// `‖x‖² ≤ 3m²`, degree 2 with bound `2²`.
pub(super) fn step_window<T: RealScalar>() -> KernelWindow<T> {
    KernelWindow::new(2, 2, 2)
}

/// Per-run state a Francis step reuses: the left-apply accumulator
/// (`apply_left`'s `scratch`, sized `n`) and the step's [`KernelWindow`].
pub(super) struct StepWorkspace<'a, T> {
    pub(super) scratch: &'a mut [T],
    pub(super) window: KernelWindow<T>,
}

/// `P = I − τ·v·vᵀ` with `v₀ = 1` (LAPACK `dlarfg`'s normalization):
/// `|vᵢ| ≤ 1` and `τ ∈ [1, 2]`.
struct StackReflector<T> {
    v: [T; 3],
    len: usize,
    tau: T,
}

/// The reflector mapping the stack vector `x` (length 2 or 3) to `α·e₁`, as
/// LAPACK `dlarfg` forms it; returns `(reflector, α)`, or `None` when
/// `x₁ = x₂ = 0` (`dlarfg`'s `τ = 0`: the identity, `α = x₀`).
///
/// `α = −sign(x₀)·‖x‖₂`, `τ = (α − x₀)/α`, `vᵢ = xᵢ/(x₀ − α)`: `x₀ − α` adds
/// magnitudes, so `|x₀ − α| ≥ ‖x‖₂ ≥ |xᵢ|` and every `|vᵢ| ≤ 1`. The
/// applications therefore form only entry-scale products `vᵢ·h` and
/// `τ·(vᵀh)` — degree 1 — so no component of `v` loses precision against
/// the entries it multiplies (an unnormalized `v` of entry scale made `vᵀh`
/// degree 2, which underflowed in its small components at the gate's lower
/// end and stalled the iteration on skew-symmetric tridiagonals there).
///
/// Scale-safe (`dlapy2`/`dnrm2`): with `m = max|xᵢ|`, `‖x‖² ≤ 3m²` (degree 2,
/// bound `2²`) is formed unscaled while `m` keeps it normal and finite
/// (`window`, [`step_window`]); otherwise `x` is first divided by the power
/// of two bringing `m` into `[1, 2)`. `τ` and `v` are invariant under that
/// scaling, and `α` is multiplied back, so inside the window the result is
/// bit-for-bit the unscaled one.
fn stack_reflector<T: RealScalar>(
    x: &[T],
    window: KernelWindow<T>,
) -> Option<(StackReflector<T>, T)> {
    let len = x.len();
    if !(2..=3).contains(&len) || x[1..].iter().all(|&xi| xi == T::ZERO) {
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
    let alpha = if v[0] < T::ZERO { norm } else { norm.neg() }; // α = −sign(x₀)·‖x‖
    let head = v[0].sub(alpha); // x₀ − α, magnitudes added
    let tau = alpha.sub(v[0]).div(alpha);
    v[0] = T::ONE;
    for vi in &mut v[1..len] {
        *vi = vi.div(head);
    }
    Some((StackReflector { v, len, tau }, alpha.scale_binary(exponent)))
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
/// shift pair `μ₁, μ₂` ([`shift_pair`], exceptional pairs by [`Shift`]) at
/// the row `m` `dlahqr` starts from (two consecutive small subdiagonals, else
/// `lo`), then chases the resulting bulge down the band with size-3 (and a
/// final size-2) Householder reflectors — a single orthogonal similarity
/// equal to one double-shifted QR step on rows `m ..= hi` (the implicit-Q
/// theorem), `h_{m,m−1}` scaled by `1 − τ` as `dlahqr` writes it.
pub(super) fn francis_step<T: RealScalar, const ACCUMULATE_Q: bool>(
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

    // `dlahqr` (loop 50): start the bulge at the lowest row `m` whose
    // subdiagonal the start would leave negligible — two consecutive small
    // subdiagonals, `|h_{m,m−1}|·(|v₂| + |v₃|) ≤ ulp·|v₁|·(|h_{m−1,m−1}| +
    // |h_{m,m}| + |h_{m+1,m+1}|)` — else at `lo`. Starting above a
    // subdiagonal too small for the deflation test but too large to vanish
    // under the chase lets the bulge die there, and the rows below never
    // receive the shifts (skew-symmetric tridiagonals at the gate's lower
    // end stalled this way).
    let ulp = crate::application::linalg::thresholds::machine_epsilon::<T>();
    let mut m = hi - 2;
    let (mut x, mut y, mut zz) = first_column(h, n, m, (rt1r, rt1i, rt2r, rt2i));
    while m > lo {
        let coupling = at(h, m, m - 1, n).abs().mul(y.abs().add(zz.abs()));
        let local = ulp.mul(x.abs()).mul(
            at(h, m - 1, m - 1, n)
                .abs()
                .add(at(h, m, m, n).abs())
                .add(at(h, m + 1, m + 1, n).abs()),
        );
        if coupling <= local {
            break;
        }
        m -= 1;
        (x, y, zz) = first_column(h, n, m, (rt1r, rt1i, rt2r, rt2i));
    }

    for k in m..=(hi - 1) {
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
            // `dlahqr`: past the first reflector the bulge column `k − 1` is
            // written as its known image `(α, 0, 0)ᵀ` rather than computed, and
            // the reflector is applied from the left to columns `k ..` and from
            // the right to rows `.. min(k + 3, hi)` — the entries outside are
            // zero in the Hessenberg form, and applying to their rounding
            // residue fed it back into the iteration (the Schur path stalled on
            // skew-symmetric tridiagonals that the eigenvalues path solved).
            if k > m {
                h[k * n + (k - 1)] = alpha;
                h[(k + 1) * n + (k - 1)] = T::ZERO;
                if refl.len == 3 {
                    h[(k + 2) * n + (k - 1)] = T::ZERO;
                }
            } else if m > lo {
                // `dlahqr`: the first reflector's image of column `m − 1`,
                // `h_{m,m−1}·(1 − τ)`, its components along `v₁, v₂`
                // negligible by the choice of `m`.
                h[k * n + (k - 1)] = h[k * n + (k - 1)].mul(T::ONE.sub(refl.tau));
            }
            let last_row = (k + refl.len).min(hi);
            if ACCUMULATE_Q {
                // Schur path (`WANTT`, `WANTZ`): `T` and the Schur vectors are
                // outputs, so the rows extend right to `n` and the columns up
                // to row `0`, and `Z` takes every row.
                apply_left(h, v_slice, refl.tau, k, n, k, n - 1, scratch);
                apply_right(h, v_slice, refl.tau, k, n, 0, last_row);
                apply_right(z, v_slice, refl.tau, k, n, 0, n - 1);
            } else {
                // Eigenvalues-only (`WANTT = false`): only the diagonal blocks
                // are read, so the apply is confined to columns `[k, hi]` and
                // rows `[lo, last_row]`: entries right of `hi` or above `lo` are
                // off every diagonal block and never feed back (`hi` only
                // decreases; `lo` is monotone non-decreasing for fixed `hi` via
                // exact-zero deflation).
                apply_left(h, v_slice, refl.tau, k, n, k, hi, scratch);
                apply_right(h, v_slice, refl.tau, k, n, lo, last_row);
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
