//! Relative accuracy on graded bidiagonals, whichever end is small
//! (`LETO-BIDIAGONAL-CHASE-DIRECTION-2026-09-25`).
//!
//! The oracle is bisection on the Golub–Kahan tridiagonal in `f64`
//! (Demmel & Kahan 1990, §6): a zero-diagonal symmetric tridiagonal whose
//! eigenvalues are `±σᵢ`, counted with the Sturm recurrence
//! `qᵢ = (0 − b²ᵢ₋₁/qᵢ₋₁) − x`. A computed count of `j` at `x` brackets
//! `σ` within the factors `1/(1 − (3n − 1.5)ε)` and
//! `(1 − (6n − 2)ε)/(1 − (3n − 1.5)ε)` of `x`, so the reference is
//! relatively accurate to `(6n − 2)·ε_f64` plus the bisection width
//! ([`REFERENCE_WIDTH`]) — independent of the magnitudes, and of the code
//! under test.

use super::chase::{Chase, Down, Up};
use super::rotation::TransposedFactors;
use super::sweep::{qr_iterate, SweepWindows};
use super::tests::check_bidiagonal;
use super::zero_shift::zero_shift_sweep;
use super::RealScalar;
use crate::application::linalg::thresholds;
use crate::domain::rng::Xorshift64;
use eunomia::RealField;

/// Relative width at which bisection stops: two `f64` ulps.
const REFERENCE_WIDTH: f64 = 2.0 * f64::EPSILON;

/// Singular values of `B` below `x > 0`: the Golub–Kahan tridiagonal of
/// order `2n` (off-diagonal `d₀, e₀, d₁, …, d_{n−1}`) has `n + count`
/// eigenvalues below `x`.
fn count_below(d: &[f64], e: &[f64], x: f64) -> usize {
    let off_diagonal = d
        .iter()
        .zip(e.iter().map(Some).chain([None]))
        .flat_map(|(&di, ei)| [Some(di), ei.copied()])
        .flatten();
    let mut q = -x;
    let mut negatives = usize::from(q < 0.0);
    for b in off_diagonal {
        // A zero pivot is perturbed to the smallest normal (LAPACK `dlaebz`'s
        // `pivmin`), a relative change far below the bracketing factors.
        if q == 0.0 {
            q = -f64::MIN_POSITIVE;
        }
        q = (0.0 - b * b / q) - x;
        negatives += usize::from(q < 0.0);
    }
    negatives - d.len()
}

/// Singular values of the upper bidiagonal `(d, e)`, ascending, by
/// bisection to [`REFERENCE_WIDTH`].
fn reference(d: &[f64], e: &[f64]) -> Vec<f64> {
    let bound = 2.0 * d.iter().chain(e).map(|v| v.abs()).sum::<f64>();
    (0..d.len())
        .map(|j| {
            let (mut lo, mut hi) = (bound * f64::MIN_POSITIVE, bound);
            while hi - lo > REFERENCE_WIDTH * hi {
                let mid = (lo * hi).sqrt();
                if count_below(d, e, mid) > j {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            hi
        })
        .collect()
}

/// Relative error of the reference: Demmel & Kahan's bisection factor
/// `(6n − 2)·ε` plus the bisection width.
fn reference_error(n: usize) -> f64 {
    f64::from(u32::try_from(6 * n).expect("invariant: test orders fit in u32")) * f64::EPSILON
        + REFERENCE_WIDTH
}

/// A graded bidiagonal of order `k` with its entries rounded to `T`: `|dᵢ|`
/// falls by `2^(span/(k−1))` per step towards the small end, each `eᵢ`
/// sits halfway (in binades) between its two diagonal neighbours, and
/// mantissas are seeded. `bottom_heavy` puts the small end at the top (the
/// case the top-to-bottom-only chase mishandled).
fn graded<T: RealScalar>(
    rng: &mut Xorshift64,
    k: usize,
    span: f64,
    bottom_heavy: bool,
) -> (Vec<f64>, Vec<f64>) {
    let step = span / f64::from(u32::try_from(k - 1).expect("invariant: small order"));
    let binade = |i: usize| {
        let from_large = if bottom_heavy { k - 1 - i } else { i };
        -step * f64::from(u32::try_from(from_large).expect("invariant: small order"))
    };
    let mut round =
        |exponent: f64| T::from_f64((1.0 + rng.next_unit_f64()) * exponent.exp2()).to_f64();
    let mut d = Vec::with_capacity(k);
    let mut e = Vec::with_capacity(k - 1);
    for i in 0..k {
        let magnitude = round(binade(i));
        d.push(if i % 3 == 1 { -magnitude } else { magnitude });
        if i + 1 < k {
            e.push(round(0.5 * (binade(i) + binade(i + 1))));
        }
    }
    (d, e)
}

/// The grading span: twice the precision in binades (so `σ_min/σ_max ≈ ε²`,
/// far below what the normwise bound resolves), capped by a quarter of the
/// normal range so no entry or rotation product is subnormal.
fn span<T: RealScalar + RealField>() -> f64 {
    let precision = -<T as RealField>::EPSILON.to_f64().log2();
    let range = -thresholds::safe_min::<T>().to_f64().log2();
    (2.0 * precision).min(range / 4.0)
}

/// Demmel & Kahan's Theorem 6 relative bound for one zero-shift sweep of
/// order `n`: `ω/(1 − ω)`, `ω = 69n²ε`.
fn zero_shift_bound(n: usize, eps: f64) -> f64 {
    let n = f64::from(u32::try_from(n).expect("invariant: small order"));
    let omega = 69.0 * n * n * eps;
    assert!(omega < 1.0, "Theorem 6 needs 69n²ε < 1");
    omega / (1.0 - omega)
}

/// Assert each computed singular value within `bound` (relative) of the
/// reference singular values of `(d, e)`.
fn assert_relative(got: &[f64], d: &[f64], e: &[f64], bound: f64, context: &str) {
    let want = reference(d, e);
    let mut got = got.iter().map(|x| x.abs()).collect::<Vec<_>>();
    got.sort_by(f64::total_cmp);
    for (g, w) in got.iter().zip(&want) {
        let relative = (g - w).abs() / w;
        assert!(
            relative <= bound,
            "{context}: σ {g:e} vs {w:e}, relative {relative:e} > {bound:e}; d {d:?} e {e:?}"
        );
    }
}

/// One zero-shift sweep in orientation `C`, checked against Theorem 6: the
/// singular values of the swept bidiagonal (rounded to `T`, read back
/// exactly) differ from the input's by at most `ω/(1 − ω)` relative, plus
/// both references' error.
fn one_sweep<T: RealScalar + RealField, C: Chase>(d0: &[f64], e0: &[f64]) {
    let k = d0.len();
    let mut d: Vec<T> = d0.iter().map(|&x| T::from_f64(x)).collect();
    let mut e: Vec<T> = e0
        .iter()
        .map(|&x| T::from_f64(x))
        .chain([T::ZERO])
        .collect();
    let before = reference(d0, e0);
    zero_shift_sweep::<T, false, C>(
        &mut d,
        &mut e,
        0,
        k - 1,
        &mut TransposedFactors::none(),
        SweepWindows::<T>::new().rotation,
    );
    let swept_d: Vec<f64> = d.iter().map(|x| x.to_f64()).collect();
    let swept_e: Vec<f64> = e[..k - 1].iter().map(|x| x.to_f64()).collect();
    let eps = <T as RealField>::EPSILON.to_f64();
    let bound = zero_shift_bound(k, eps) + 2.0 * reference_error(k);
    assert_relative(&before, &swept_d, &swept_e, bound, "one zero-shift sweep");
}

fn zero_shift_sweep_is_relatively_accurate<T: RealScalar + RealField>() {
    let mut rng = Xorshift64::new(0x5eed_b1d1);
    for case in 0..240 {
        let k = 3 + case % 6;
        for bottom_heavy in [false, true] {
            let (d, e) = graded::<T>(&mut rng, k, span::<T>(), bottom_heavy);
            one_sweep::<T, Down>(&d, &e);
            one_sweep::<T, Up>(&d, &e);
        }
    }
}

/// Theorem 6 holds for the zero-shift sweep chased in either direction, on
/// graded input with either end small: the bottom-to-top chase is the
/// reflection `P Bᵀ P` of the top-to-bottom one (`chase.rs`), so it inherits
/// the bound.
#[test]
fn zero_shift_sweep_keeps_relative_accuracy_in_both_directions() {
    zero_shift_sweep_is_relatively_accurate::<f32>();
    zero_shift_sweep_is_relatively_accurate::<f64>();
}

/// The relative bound on each singular value after a full reduction of order
/// `k` in `K` sweeps, composed from the per-event bounds:
///
/// - a zero-shift sweep: Theorem 6, `ω₀ = 69k²ε/(1 − 69k²ε)`;
/// - a shifted step, taken only when `dbdsqr`'s first zero-shift test fails,
///   i.e. `σ_max/σ̃ < k·tol/max(ε, tol/100)` with `σ̃` the block's
///   recurrence minimum and `σ_min ≥ σ̃/√k` (Demmel & Kahan eq. 2.5): its
///   normwise error `8kε·σ_max` (Golub & Van Loan §8.6.3, the `p = 8k` of
///   `tests.rs`) is `ω_s = 8k^{5/2}·(tol/max(ε, tol/100))·ε` relative;
/// - a split (`k − 1` at most), by convergence criteria 1a/1b or the
///   threshold `tol·σ̃_min/√k ≤ tol·σ_min`: Theorem 4, `k·tol/√2`;
/// - the one 2×2 each value passes through (`dlasv2`, "a few ulps"),
///   bounded by `ω₀` at order 2.
///
/// `(1 + max(ω₀, ω_s))^K · (1 + k·tol/√2)^(k−1) · (1 + ω₀(2)) − 1`, plus the
/// reference's error.
fn reduction_bound<T: RealScalar + RealField>(k: usize, sweeps: usize) -> f64 {
    let eps = <T as RealField>::EPSILON.to_f64();
    let machine = thresholds::machine_epsilon::<T>().to_f64();
    let tolmul = machine.powf(-0.125).clamp(10.0, 100.0);
    let tol = tolmul * machine;
    let order = f64::from(u32::try_from(k).expect("invariant: small order"));
    let shifted = 8.0 * order.powf(2.5) * (tol / machine.max(tol / 100.0)) * eps;
    let per_sweep = zero_shift_bound(k, eps).max(shifted);
    let split = order * tol / core::f64::consts::SQRT_2;
    let sweeps = i32::try_from(sweeps).expect("invariant: sweep count fits in i32");
    let splits = i32::try_from(k - 1).expect("invariant: small order");
    (1.0 + per_sweep).powi(sweeps) * (1.0 + split).powi(splits) * (1.0 + zero_shift_bound(2, eps))
        - 1.0
        + reference_error(k)
}

fn graded_reductions_are_relatively_accurate<T: RealScalar + RealField>() {
    let mut rng = Xorshift64::new(0x0b07_70e5);
    for case in 0..600 {
        let k = 3 + case % 6;
        for bottom_heavy in [true, false] {
            let (d0, e0) = graded::<T>(&mut rng, k, span::<T>(), bottom_heavy);
            let mut d: Vec<T> = d0.iter().map(|&x| T::from_f64(x)).collect();
            let mut e: Vec<T> = e0
                .iter()
                .map(|&x| T::from_f64(x))
                .chain([T::ZERO])
                .collect();
            let sweeps = qr_iterate::<T, false>(&mut d, &mut e, k, &mut TransposedFactors::none())
                .expect("a graded bidiagonal converges");
            let got: Vec<f64> = d.iter().map(|x| x.to_f64()).collect();
            let bound = reduction_bound::<T>(k, sweeps);
            assert_relative(&got, &d0, &e0, bound, "graded reduction");
        }
    }
}

/// Every singular value of a graded bidiagonal — its smallest `≈ ε²` of its
/// largest — keeps high relative accuracy whichever end is small. Without
/// `dbdsqr`'s chase-direction choice the bottom-heavy half returns
/// smallest singular values as `0` (relative error 1) at `f32`.
#[test]
fn graded_bidiagonals_keep_relative_accuracy_at_either_end() {
    graded_reductions_are_relatively_accurate::<f32>();
    graded_reductions_are_relatively_accurate::<f64>();
}

/// The bottom-to-top chase accumulates its rotations into the reflected
/// factors: `B = U Σ Vᵀ` reconstructs the bottom-heavy input and `U`, `V`
/// stay orthonormal (`check_bidiagonal`'s normwise bounds).
#[test]
fn bottom_heavy_iteration_reconstructs_its_input() {
    fn check<T: RealScalar + RealField>() {
        let mut rng = Xorshift64::new(0x00c0_ffee);
        for k in 3..=8 {
            let (d, e) = graded::<T>(&mut rng, k, span::<T>(), true);
            let mut expected = reference(&d, &e);
            expected.reverse();
            check_bidiagonal::<T>(&d, &e, &expected);
        }
    }
    check::<f32>();
    check::<f64>();
}
