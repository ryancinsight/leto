//! The Frobenius-to-max norm ratio bound a gate charges for the matrix's
//! spread of magnitudes.

use crate::application::linalg::thresholds;
use crate::domain::real::RealScalar;

/// `⌈log₂(‖A‖_F / ‖A‖_max)⌉` for the entries `values` whose largest
/// magnitude is `largest`: the power-of-two bound on the ratio that lets a
/// gate state `‖A‖₂ ≤ ‖A‖_F ≤ 2^r·‖A‖_max` tightly (a diagonal-dominated
/// matrix has `r = 0`, where the dimension bound `√(mn)` would charge up to
/// `log₂ n`).
///
/// The sum of squared ratios is formed pairwise with a running binary
/// exponent ([`RatioSum`]), so it neither overflows (`256²` unit `F16`
/// entries sum to `2¹⁶ > Ω`) nor stagnates (a recursive `F16` sum stops at
/// `2048`, since `2048 + 1 = 2048`). Each term passes through at most
/// `ℓ = ⌈log₂ len⌉ + 2` roundings, and underflow — of a squared ratio or of
/// a partial sum aligned to a larger exponent — loses at most `η = safmin·ε`
/// per term and per alignment, each against a total of at least `1` (the
/// largest term is exactly `1`). So `|ŝ − s| ≤ γ_ℓ·s + 2·len·η·s` (Higham,
/// *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §4.2), and with
/// `c` the power of two at least twice the larger of `2^⌈log₂(4ℓ + 4)⌉·u` and
/// `2^⌈log₂(4·len)⌉·η`, `(1 − c)·s ≤ ŝ ≤ (1 + c)·s`. A computed sum strictly
/// inside `(2^k·(1 − c), 2^k)` may hide a true one above `2^k` and is charged
/// `k + 1`; one exactly on `2^k` is taken as exact — the diagonal case,
/// `ŝ = 1`, whose off-diagonal squares underflow, is the one where rounding
/// can land there from above, and then by less than `c·2^k`. Where `c ≥ ½`
/// the dimension bound `⌈⌈log₂ len⌉/2⌉` is used.
pub(crate) fn norm_ratio_log2<T: RealScalar>(values: &[T], largest: T) -> i32 {
    let dimension_bound = (thresholds::ceil_log2_count(values.len()) + 1) / 2;
    let Some(sum) = RatioSum::of(values, largest) else {
        return dimension_bound;
    };
    // `mantissa ≤ 2^k`; `1 − slack` is exact (a power of two at least `ε`,
    // below one half), so the band test is exact.
    let mut k = thresholds::ceil_log2(sum.mantissa);
    let power = T::ONE.scale_binary(k);
    if sum.mantissa < power && sum.mantissa > power.mul(T::ONE.sub(sum.slack)) {
        k += 1;
    }
    ((k + sum.exponent + 1) / 2).min(dimension_bound)
}

/// A lower bound `l` on `log₂(‖A‖_F / ‖A‖_max)`, `2^l ≤ ‖A‖_F/‖A‖_max`, from
/// the same sum as [`norm_ratio_log2`]: `s ≥ ŝ/(1 + c) ≥ ŝ·(1 − c)`, and the
/// product `ŝ·(1 − 2c)` rounds (upward by at most `u ≤ c`) to at most
/// `ŝ·(1 − c)`, so `l = ⌊⌊log₂ fl(ŝ·(1 − 2c))⌋/2⌋`. `0` (the ratio is at
/// least `1`) where the sum is not formed.
pub(crate) fn norm_ratio_floor_log2<T: RealScalar>(values: &[T], largest: T) -> i32 {
    RatioSum::of(values, largest).map_or(0, |sum| {
        let lower = sum.mantissa.mul(T::ONE.sub(sum.slack.scale_binary(1)));
        let bits = lower.binary_exponent().unwrap_or(0) + sum.exponent;
        bits.max(0) / 2
    })
}

/// `Σ (vᵢ/largest)² = mantissa·2^exponent`, summed pairwise, with the power
/// of two `slack = c` bounding its relative rounding ([`norm_ratio_log2`]).
struct RatioSum<T> {
    mantissa: T,
    exponent: i32,
    slack: T,
}

impl<T: RealScalar> RatioSum<T> {
    /// `None` where `c ≥ ½` or the sum is not finite.
    fn of(values: &[T], largest: T) -> Option<Self> {
        let log2_len = thresholds::ceil_log2_count(values.len());
        let roundings =
            usize::try_from(log2_len + 2).expect("invariant: a bit count is non-negative");
        let log2_eps = thresholds::ceil_log2(thresholds::machine_epsilon::<T>());
        let rounding_log2 = log2_eps - 1 + thresholds::ceil_log2_count(4 * roundings + 4);
        let underflow_log2 = thresholds::ceil_log2(thresholds::safe_min::<T>())
            + log2_eps
            + thresholds::ceil_log2_count(4 * values.len());
        let slack = T::ONE.scale_binary(rounding_log2.max(underflow_log2) + 1);
        if slack.scale_binary(1) >= T::ONE {
            return None;
        }
        let (mantissa, exponent) = pairwise_squares(values, largest);
        (mantissa.is_finite() && mantissa > T::ZERO).then_some(Self {
            mantissa,
            exponent,
            slack,
        })
    }
}

/// `Σ (vᵢ/largest)²` by pairwise summation as `(m, k)`, value `m·2^k`: halves
/// recursively, the smaller exponent aligned to the larger (exact but for
/// underflow) and a sum reaching `2` halved (exact), so `m < 2` for `k > 0`
/// and nothing overflows. Every term passes through at most `⌈log₂ len⌉`
/// additions besides its division and square.
fn pairwise_squares<T: RealScalar>(values: &[T], largest: T) -> (T, i32) {
    match values {
        [] => (T::ZERO, 0),
        [value] => {
            let ratio = value.div(largest);
            (ratio.mul(ratio), 0)
        }
        _ => {
            let (left, right) = values.split_at(values.len() / 2);
            let (a, ka) = pairwise_squares(left, largest);
            let (b, kb) = pairwise_squares(right, largest);
            let k = ka.max(kb);
            let sum = a.scale_binary(ka - k).add(b.scale_binary(kb - k));
            if sum >= T::ONE.add(T::ONE) {
                (sum.scale_binary(-1), k + 1)
            } else {
                (sum, k)
            }
        }
    }
}
