//! Exact power-of-two scaling of dense factorization inputs, in two tiers.
//!
//! **Kernel tier.** The Francis double-shift and Golub–Kahan bidiagonal QR
//! kernels form their products scale-safely. A Givens norm, a reflector
//! norm, and the Golub–Kahan shift are formed unscaled while their local
//! magnitude lies in the kernel's representable window
//! ([`thresholds::kernel_window`](super::thresholds::kernel_window)) and, only
//! outside it, from operands rescaled by the power of two bringing the local
//! magnitude into `[1, 2)` ([`KernelWindow`]) — the LAPACK `dlartg` / `dnrm2`
//! pattern; inside the window the arithmetic is the unscaled arithmetic. The
//! Francis shift and first column (`dlahqr`) and the 2×2 standardization
//! (`dlanv2`) are scale-safe by construction: every product in them is of
//! ratios or square roots.
//!
//! **Matrix tier.** What the kernels cannot rescale locally — an intermediate
//! that is a product of entries accumulated across the whole matrix
//! (a pivoted-QR column norm, the QL chase's `e₁·eₗ`) or a degree-1 sum whose
//! bound grows with the dimension — is guarded by the gate: each routine
//! states the degree `d` and a power-of-two bound `2^f` with
//! `|intermediate| ≤ 2^f·‖A‖_max^d`, derived from its formulas at the call
//! site, and [`thresholds::homogeneous_safe_range`](super::thresholds::homogeneous_safe_range)
//! turns that into the range of `‖A‖_max` factored unscaled. An input inside
//! it is factored unscaled. Outside it the routine factors `2⁻ᵏ·A`, `k` the minimal move
//! (either sign) bringing `‖A‖_max` back inside ([`balancing_exponent`]),
//! and multiplies the scale-carrying results back by `2ᵏ` ([`restore`]).
//!
//! **Exactness.** Multiplying by a power of two changes only the exponent, so
//! it is exact whenever the result stays normal. Scaling *up* is therefore
//! always exact; scaling *down* can underflow an entry far below the largest
//! one (`diag(1e300, 1e-300)` would lose `1e-300` if `f64` had to scale it
//! down). The minimal move scales down only as far as the gate's upper end
//! requires, and the upper ends are the overflow threshold itself (not the
//! LAPACK `ε/safmin` margin), so such loss occurs only for inputs within
//! `2^f` of overflowing; there the factorizations' backward-error bound, not
//! entrywise exactness, is the guarantee.

use crate::application::linalg::thresholds;
use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, LetoError, Result};

/// The largest finite magnitude in `values`, or `None` when it is empty, all
/// zero, or holds a non-finite entry (which the caller's validation reports).
pub(crate) fn largest_magnitude<T: RealScalar>(values: impl IntoIterator<Item = T>) -> Option<T> {
    let mut largest = T::ZERO;
    for value in values {
        if !value.is_finite() {
            return None;
        }
        let magnitude = value.abs();
        if magnitude > largest {
            largest = magnitude;
        }
    }
    if largest == T::ZERO {
        None
    } else {
        Some(largest)
    }
}

/// `⌈log₂(‖A‖_F / ‖A‖_max)⌉` for the entries `values` whose largest
/// magnitude is `largest`: the power-of-two bound on the ratio that lets a
/// gate state `‖A‖₂ ≤ ‖A‖_F ≤ 2^r·‖A‖_max` tightly (a diagonal-dominated
/// matrix has `r = 0`, where the dimension bound `√(mn)` would charge up to
/// `log₂ n`).
///
/// The sum of squared ratios is formed pairwise ([`pairwise_squares`]), so
/// each term passes through at most `ℓ = ⌈log₂ len⌉ + 2` roundings and the
/// computed `ŝ` satisfies `ŝ ≥ s·(1 − γ_ℓ) − len·η` (Higham, *Accuracy and
/// Stability of Numerical Algorithms*, 2nd ed., §4.2; `η = safmin·ε` bounds
/// the absolute loss of a term that underflows), with `s ≥ 1` (the largest
/// term is exactly `1`). With `c` the power of two at least twice the larger
/// of `2^⌈log₂(4ℓ + 4)⌉·u` and `2^⌈log₂(2·len)⌉·η`, `c ≥ 2γ_ℓ + 2·len·η`, and
/// `s ≤ ŝ/(1 − c)`: a computed sum strictly inside `(2^k·(1 − c), 2^k)` may
/// hide a true one above `2^k` and is charged `k + 1`. A computed sum
/// exactly on `2^k` is taken as exact — the diagonal case, `ŝ = 1`, whose
/// off-diagonal squares underflow, is the one where rounding can land there
/// from above, and then by less than `c·2^k`. Recursive summation instead
/// stagnates in the narrow formats (in `F16`, `2048 + 1 = 2048`: `128²` unit
/// entries summed to `2048`, charging `r = 6` for a ratio of `2⁷`). Where
/// `c ≥ ½`, or the sum leaves the range, the dimension bound
/// `⌈⌈log₂ len⌉/2⌉` is used.
pub(crate) fn norm_ratio_log2<T: RealScalar>(values: &[T], largest: T) -> i32 {
    let dimension_bound = (thresholds::ceil_log2_count(values.len()) + 1) / 2;
    let Some((sum, slack)) = ratio_sum(values, largest) else {
        return dimension_bound;
    };
    // `sum ≤ 2^k`; `1 − slack` is exact (a power of two at least `ε`, below
    // one half), so the band test is exact.
    let mut k = thresholds::ceil_log2(sum);
    let power = T::ONE.scale_binary(k);
    if sum < power && sum > power.mul(T::ONE.sub(slack)) {
        k += 1;
    }
    ((k + 1) / 2).min(dimension_bound)
}

/// A lower bound `l` on `log₂(‖A‖_F / ‖A‖_max)`, `2^l ≤ ‖A‖_F/‖A‖_max`, from
/// the same sum as [`norm_ratio_log2`]: with `s ≥ ŝ·(1 − c) ≥ ŝ/2` (the rounding
/// bound there read the other way, `ŝ ≤ s·(1 + γ_ℓ) + len·η ≤ s·(1 + c)`),
/// `s ≥ 2^(⌊log₂ ŝ⌋ − 1)` and `l = ⌊(⌊log₂ ŝ⌋ − 1)/2⌋`. `0` (the ratio is at
/// least `1`) where the sum is not formed.
pub(crate) fn norm_ratio_floor_log2<T: RealScalar>(values: &[T], largest: T) -> i32 {
    ratio_sum(values, largest).map_or(0, |(sum, _)| {
        (sum.binary_exponent().unwrap_or(0) - 1).max(0) / 2
    })
}

/// The pairwise sum `ŝ` of `(vᵢ/largest)²` and the power of two `c` bounding
/// its relative rounding, or `None` where `c ≥ ½` or the sum leaves the
/// range ([`norm_ratio_log2`]).
fn ratio_sum<T: RealScalar>(values: &[T], largest: T) -> Option<(T, T)> {
    let log2_len = thresholds::ceil_log2_count(values.len());
    let roundings = usize::try_from(log2_len + 2).expect("invariant: a bit count is non-negative");
    let log2_eps = thresholds::ceil_log2(thresholds::machine_epsilon::<T>());
    let rounding_log2 = log2_eps - 1 + thresholds::ceil_log2_count(4 * roundings + 4);
    let underflow_log2 = thresholds::ceil_log2(thresholds::safe_min::<T>())
        + log2_eps
        + thresholds::ceil_log2_count(2 * values.len());
    let slack = T::ONE.scale_binary(rounding_log2.max(underflow_log2) + 1);
    if slack.scale_binary(1) >= T::ONE {
        return None;
    }
    let sum = pairwise_squares(values, largest);
    (sum.is_finite() && sum >= T::ONE).then_some((sum, slack))
}

/// `Σ (vᵢ/largest)²` by pairwise summation: halves recursively, each half
/// summed the same way, so every term passes through at most
/// `⌈log₂ len⌉` additions besides its division and square.
fn pairwise_squares<T: RealScalar>(values: &[T], largest: T) -> T {
    match values {
        [] => T::ZERO,
        [value] => {
            let ratio = value.div(largest);
            ratio.mul(ratio)
        }
        _ => {
            let (left, right) = values.split_at(values.len() / 2);
            pairwise_squares(left, largest).add(pairwise_squares(right, largest))
        }
    }
}

/// The minimal-magnitude integer `k` (either sign) with `largest·2⁻ᵏ`
/// inside `[rmin, rmax]`. `largest` is assumed already outside it (the
/// caller checks); `rmin < rmax`, both positive and finite.
fn minimal_exponent_to_range<T: RealScalar>(largest: T, rmin: T, rmax: T) -> i32 {
    if largest > rmax {
        // Scale down: the smallest k > 0 with `largest / 2^k <= rmax`.
        let mut k = largest.binary_exponent().unwrap_or(0) - rmax.binary_exponent().unwrap_or(0);
        if k < 1 {
            k = 1;
        }
        while largest.scale_binary(-k) > rmax {
            k += 1;
        }
        while k > 1 && largest.scale_binary(-(k - 1)) <= rmax {
            k -= 1;
        }
        k
    } else {
        // largest < rmin: scale up (k negative), the smallest |k| with
        // `largest / 2^k >= rmin`.
        let mut k = largest.binary_exponent().unwrap_or(0) - rmin.binary_exponent().unwrap_or(0);
        if k > -1 {
            k = -1;
        }
        while largest.scale_binary(-k) < rmin {
            k -= 1;
        }
        while k < -1 && largest.scale_binary(-(k + 1)) >= rmin {
            k += 1;
        }
        k
    }
}

/// The minimal-move exponent `k` bringing `largest` into `range`, or `None`
/// when it already lies inside (the caller factors its input unscaled).
pub(crate) fn balancing_exponent<T: RealScalar>(largest: T, range: (T, T)) -> Option<i32> {
    let (rmin, rmax) = range;
    if largest >= rmin && largest <= rmax {
        return None;
    }
    Some(minimal_exponent_to_range(largest, rmin, rmax))
}

/// A kernel's representable window for its local magnitude `m` (the
/// largest operand it reads): the kernel tier's gate, computed once per
/// routine call from [`thresholds::kernel_window`] and passed down to the
/// hot kernels.
#[derive(Clone, Copy)]
pub(crate) struct KernelWindow<T> {
    low: T,
    high: T,
}

impl<T: RealScalar> KernelWindow<T> {
    /// The window keeping the kernel's smallest relied-upon product (degree
    /// `lower_degree`) normal and its largest (degree `upper_degree`, bounded
    /// by `2^factor_log2·m^dᵤ`) finite.
    pub(crate) fn new(lower_degree: u32, upper_degree: u32, factor_log2: i32) -> Self {
        let (low, high) = thresholds::kernel_window::<T>(lower_degree, upper_degree, factor_log2);
        Self { low, high }
    }

    /// `0` when the largest magnitude among `operands` lies in the window (or
    /// is zero or non-finite — nothing to rescale), otherwise its binary
    /// exponent `k`, so the operands divided by `2ᵏ` have their largest
    /// magnitude in `[1, 2)`.
    pub(crate) fn exponent(self, operands: &[T]) -> i32 {
        let largest = operands
            .iter()
            .fold(T::ZERO, |acc, v| if v.abs() > acc { v.abs() } else { acc });
        if largest == T::ZERO || (largest >= self.low && largest <= self.high) {
            return 0;
        }
        largest.binary_exponent().unwrap_or(0)
    }
}

/// `√(x² + y²)` without the overflow or underflow of squaring the larger
/// operand — LAPACK `dlapy2`: the larger magnitude times
/// `√(1 + (smaller/larger)²)`.
#[inline]
pub(crate) fn hypot<T: RealScalar>(x: T, y: T) -> T {
    let (x, y) = (x.abs(), y.abs());
    let (large, small) = if x >= y { (x, y) } else { (y, x) };
    if large == T::ZERO {
        return T::ZERO;
    }
    let ratio = small.div(large);
    large.mul(T::ONE.add(ratio.mul(ratio)).sqrt())
}

/// Multiply every entry of `values` by `2^exponent` (exact while the results
/// stay normal).
pub(crate) fn scale_by_power_of_two<T: RealScalar>(values: &mut [T], exponent: i32) {
    for value in values {
        *value = value.scale_binary(exponent);
    }
}

/// A gate's power-of-two bounds (see
/// [`thresholds::homogeneous_safe_range`]): `2^factor_log2` bounds the
/// relied-upon intermediate from above; `2^floor_log2·safmin` is the
/// routine's absolute deflation threshold divided by `2^l ≤ ‖A‖_F/‖A‖_max`
/// ([`norm_ratio_floor_log2`]), so that `floor ≤ ε·2^l·‖A‖_max ≤ ε·‖A‖_F`
/// is what the lower end secures (`0` when it has none beyond `safmin`).
#[derive(Clone, Copy)]
pub(crate) struct GateBound {
    pub(crate) factor_log2: i32,
    pub(crate) floor_log2: i32,
}

impl GateBound {
    /// A bound with no deflation floor.
    pub(crate) fn factor(factor_log2: i32) -> Self {
        Self {
            factor_log2,
            floor_log2: 0,
        }
    }
}

/// The matrix-tier gate over the entries `values`: `Ok(Some(k))`, the
/// minimal move, when `‖values‖_max` lies outside
/// [`thresholds::homogeneous_safe_range`] for `degree` and
/// `bound(values, ‖values‖_max)`; `Ok(None)` when it lies inside or `values`
/// is empty, all zero, or holds a non-finite entry.
///
/// # Errors
///
/// [`LetoError::Overflow`] when the range is empty
/// ([`thresholds::EmptyRange`]): the bound factor exceeds the format's
/// `Ω/smlnum`, so the routine's intermediates cannot be kept finite at this
/// order in `T` whatever the scaling, or no scaling keeps the routine's
/// deflation floor at or below `ε·‖A‖_F` (through its `2^l ≤ ‖A‖_F/‖A‖_max`).
pub(crate) fn gate_exponent<T: RealScalar>(
    values: &[T],
    degree: u32,
    bound: impl FnOnce(&[T], T) -> GateBound,
) -> Result<Option<i32>> {
    let Some(largest) = largest_magnitude(values.iter().copied()) else {
        return Ok(None);
    };
    let GateBound {
        factor_log2,
        floor_log2,
    } = bound(values, largest);
    let range = thresholds::homogeneous_safe_range::<T>(degree, factor_log2, floor_log2).map_err(
        |empty| LetoError::Overflow {
            reason: match empty {
                thresholds::EmptyRange::Intermediates => "the matrix order exceeds the scalar's exponent range: its bounded intermediates cannot be kept finite",
                thresholds::EmptyRange::DeflationFloor => "the matrix order exceeds the scalar's exponent range: no scaling keeps the absolute deflation floor 2^ceil(log2 n)*safmin at or below eps*||A||_F",
            },
        },
    )?;
    Ok(balancing_exponent(largest, range))
}

/// `matrix` multiplied by `2⁻ᵏ` as an owned contiguous copy, `k` from
/// [`gate_exponent`]; `Ok(None)` when no scaling applies, in which case the
/// caller factors `matrix` itself, unscaled.
///
/// # Errors
///
/// As [`gate_exponent`].
pub(crate) fn balanced<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    degree: u32,
    bound: impl FnOnce(&[T], T) -> GateBound,
) -> Result<Option<(Array2<T>, i32)>> {
    let mut values = match matrix.as_slice() {
        Some(slice) => slice.to_vec(),
        None => matrix.iter().copied().collect(),
    };
    let Some(exponent) = gate_exponent(&values, degree, bound)? else {
        return Ok(None);
    };
    scale_by_power_of_two(&mut values, -exponent);
    let array = Array2::from_shape_vec(matrix.shape(), values)
        .expect("invariant: a copy of a view keeps its shape and length");
    Ok(Some((array, exponent)))
}

/// Multiply scale-carrying results back by `2ᵏ`.
///
/// # Errors
///
/// [`LetoError::Overflow`] when a result exceeds the range of `T`: the true
/// value is not representable, so returning it as infinity would be a wrong
/// `Ok`.
pub(crate) fn restore<T: RealScalar>(
    values: &mut [T],
    exponent: i32,
    what: &'static str,
) -> Result<()> {
    scale_by_power_of_two(values, exponent);
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(LetoError::Overflow { reason: what })
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test scope: a failed precondition is a test failure"
)]
mod tests {
    use super::{
        gate_exponent, norm_ratio_log2, restore, scale_by_power_of_two, GateBound, KernelWindow,
    };
    use crate::application::linalg::thresholds::homogeneous_safe_range;
    use leto::LetoError;

    fn no_factor(_: &[f64], _: f64) -> GateBound {
        GateBound::factor(0)
    }

    #[test]
    fn in_range_values_are_not_scaled() {
        // Degree 2, bound 2⁰: f64's range is [√(2⁻⁹⁷⁰), √Ω) ≈ [2⁻⁴⁸⁵, 2⁵¹²).
        let (rmin, rmax) = homogeneous_safe_range::<f64>(2, 0, 0).unwrap();
        assert!(
            rmin > 2.0_f64.powi(-486) && rmin < 2.0_f64.powi(-484),
            "{rmin}"
        );
        assert!(
            rmax > 2.0_f64.powi(511) && rmax < 2.0_f64.powi(512),
            "{rmax}"
        );
        for value in [1.0_f64, 3.99, 4.0, 0.5, 0.2, 1e100, 1e-100, 1e150, 1e-145] {
            assert_eq!(
                gate_exponent(&[value, -value / 3.0], 2, no_factor).unwrap(),
                None,
                "{value} should factor unscaled"
            );
        }
        assert_eq!(gate_exponent(&[0.0_f64, -0.0], 2, no_factor).unwrap(), None);
        assert_eq!(
            gate_exponent(&[1.0_f64, f64::INFINITY], 2, no_factor).unwrap(),
            None
        );
        assert_eq!(
            gate_exponent(&Vec::<f64>::new(), 2, no_factor).unwrap(),
            None
        );
    }

    #[test]
    fn degree_one_upper_end_is_the_overflow_threshold() {
        // Degree 1 takes no root: the upper end is exactly Ω·2⁻ᶠ, so
        // `1.5·2¹⁰²²` (≤ Ω/2) is in range at bound 2¹ and out at 2².
        let value = 1.5 * 2.0_f64.powi(1022);
        assert_eq!(
            gate_exponent(&[value], 1, |_, _| GateBound::factor(1)).unwrap(),
            None
        );
        assert_eq!(
            gate_exponent(&[value], 1, |_, _| GateBound::factor(2)).unwrap(),
            Some(1)
        );
        let (_, rmax) = homogeneous_safe_range::<f64>(1, 1, 0).unwrap();
        assert_eq!(rmax, f64::MAX / 2.0);
    }

    #[test]
    fn out_of_range_values_scale_by_the_minimal_move() {
        let (rmin, rmax) = homogeneous_safe_range::<f64>(2, 0, 0).unwrap();
        let k = gate_exponent(&[1e300_f64], 2, no_factor)
            .unwrap()
            .expect("1e300 is out of range");
        let mut scaled = [1e300_f64];
        scale_by_power_of_two(&mut scaled, -k);
        assert!(scaled[0] <= rmax, "{} > {rmax}", scaled[0]);
        let mut one_less = [1e300_f64];
        scale_by_power_of_two(&mut one_less, -(k - 1));
        assert!(one_less[0] > rmax, "{k} was not minimal");

        let tiny = f64::from_bits(1);
        let k = gate_exponent(&[tiny], 2, no_factor)
            .unwrap()
            .expect("subnormal is out of range");
        assert!(k < 0);
        let mut scaled = [tiny];
        scale_by_power_of_two(&mut scaled, -k);
        assert!(scaled[0] >= rmin, "{} < {rmin}", scaled[0]);
        let mut one_less = [tiny];
        scale_by_power_of_two(&mut one_less, -(k + 1));
        assert!(one_less[0] < rmin, "{k} was not minimal");
    }

    #[test]
    fn bound_factor_narrows_only_the_upper_end() {
        let (rmin0, rmax0) = homogeneous_safe_range::<f64>(4, 0, 0).unwrap();
        let (rmin, rmax) = homogeneous_safe_range::<f64>(4, 1000, 0).unwrap();
        assert_eq!(rmin, rmin0);
        assert!(rmax < rmax0);
        assert_eq!(gate_exponent(&[100.0_f64], 4, no_factor).unwrap(), None);
        assert!(matches!(
            gate_exponent(&[100.0_f64], 4, |_, _| GateBound::factor(1000)),
            Ok(Some(_))
        ));
    }

    #[test]
    fn an_empty_range_is_a_typed_overflow() {
        // F16: Ω/smlnum = 65504·2⁴ < 2²⁰, so a degree-1 bound factor of 2²⁰ leaves
        // no scaling that keeps the intermediates finite.
        let values = [eunomia::F16::from_f64(1.0)];
        assert!(matches!(
            gate_exponent(&values, 1, |_, _| GateBound::factor(20)),
            Err(LetoError::Overflow { .. })
        ));
    }

    #[test]
    fn an_unfittable_deflation_floor_is_a_typed_overflow() {
        // F16, degree 2, no bound factor: upper end √Ω < 2⁸; a floor needing
        // ‖A‖_max ≥ 2^13·smlnum = 2⁹ cannot be met by any scaling.
        let one = eunomia::F16::from_f64(1.0);
        let floor = |_: &[eunomia::F16], _| GateBound {
            factor_log2: 0,
            floor_log2: 13,
        };
        match gate_exponent(&[one], 2, floor) {
            Err(LetoError::Overflow { reason }) => {
                assert!(reason.contains("deflation floor"), "{reason}");
            }
            other => panic!("expected the deflation-floor overflow, got {other:?}"),
        }
    }

    #[test]
    fn deflation_floor_raises_only_the_lower_end() {
        // Degree 1: the lower end is smlnum = 2⁻⁹⁷⁰, raised to 2⁶·smlnum by a
        // 2⁶·safmin deflation floor so that floor is ε·‖A‖_max at most.
        let (rmin, rmax) = homogeneous_safe_range::<f64>(1, 0, 0).unwrap();
        let (floor_rmin, floor_rmax) = homogeneous_safe_range::<f64>(1, 0, 6).unwrap();
        assert_eq!(rmin, 2.0_f64.powi(-970));
        assert_eq!(floor_rmin, 2.0_f64.powi(-964));
        assert_eq!(floor_rmax, rmax);
        assert!(64.0 * f64::MIN_POSITIVE <= f64::EPSILON * floor_rmin);
    }

    #[test]
    fn window_exponent_rescales_only_outside_the_window() {
        let window = KernelWindow::<f64>::new(2, 2, 1);
        assert_eq!(window.exponent(&[3.0, -1e-300]), 0);
        assert_eq!(window.exponent(&[0.0, -0.0]), 0);
        assert_eq!(window.exponent(&[1.0, -(2.0_f64.powi(600))]), 600);
        assert_eq!(window.exponent(&[1.5 * 2.0_f64.powi(-600)]), -600);
    }

    #[test]
    fn norm_ratio_is_tight_for_a_diagonal_and_bounded_by_the_dimension() {
        assert_eq!(norm_ratio_log2(&[1e300_f64, 0.0, 0.0, 1e-300], 1e300), 0);
        // All-ones 4×4: ‖A‖_F/‖A‖_max = 4 = 2².
        assert_eq!(norm_ratio_log2(&[1.0_f64; 16], 1.0), 2);
        // All-ones 3×3: 3, rounded up to 2².
        assert_eq!(norm_ratio_log2(&[1.0_f64; 9], 1.0), 2);
        // 128² unit entries in F16: ‖A‖_F/‖A‖_max = 2⁷. A recursive sum
        // stagnates at 2048 (2048 + 1 rounds to 2048) and charged r = 6.
        let one = eunomia::F16::from_f64(1.0);
        assert_eq!(norm_ratio_log2(&vec![one; 128 * 128], one), 7);
        // A computed sum just below a power of two may hide a true one above
        // it: 64 entries, one 1 and fifteen (1 − 2⁻⁶), sum 1 + 15·0.96875 =
        // 15.53 inside the F16 band (16·(1 − 2⁻⁴), 16), are charged
        // ⌈log₂ √32⌉ = 3 rather than ⌈log₂ √15.53⌉ = 2.
        let near = eunomia::F16::from_f64(1.0 - 2.0_f64.powi(-6));
        let zero = eunomia::F16::from_f64(0.0);
        let mut band = vec![zero; 64];
        band[0] = one;
        band[1..16].fill(near);
        assert_eq!(norm_ratio_log2(&band, one), 3);
    }

    #[test]
    fn restore_reports_an_unrepresentable_result() {
        let mut values = [1.5_f64, 3.0];
        assert!(matches!(
            restore(&mut values, 1023, "probe"),
            Err(LetoError::Overflow { reason: "probe" })
        ));
        let mut values = [1.5_f64, -3.0];
        restore(&mut values, 4, "probe").expect("in range");
        assert_eq!(values, [24.0, -48.0]);
    }
}
