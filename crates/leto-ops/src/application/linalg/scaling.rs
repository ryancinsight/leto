//! Exact power-of-two scaling of dense factorization inputs, in two tiers.
//!
//! **Kernel tier.** The Francis double-shift and Golub–Kahan bidiagonal QR
//! kernels form their products scale-safely: each local quantity that is a
//! product of entries (a Givens norm, a reflector norm, a shift's first
//! column, a 2×2 block's discriminant) is formed unscaled while its local
//! magnitude lies in the kernel's representable window
//! ([`thresholds::kernel_window`](super::thresholds::kernel_window)) and, only
//! outside it, from operands rescaled by the power of two bringing the local
//! magnitude into `[1, 2)` ([`window_exponent`]) — the LAPACK `dlartg` /
//! `dnrm2` / `dlahqr` pattern. Inside the window the kernel's arithmetic is
//! bit-for-bit the unscaled arithmetic.
//!
//! **Matrix tier.** What the kernels cannot rescale locally — an intermediate
//! that is a product of entries accumulated across the whole matrix
//! (a pivoted-QR column norm, the QL chase's `e₁·eₗ`) or a degree-1 sum whose
//! bound grows with the dimension — is guarded by the gate: each routine
//! states the degree `d` and a power-of-two bound `2^f` with
//! `|intermediate| ≤ 2^f·‖A‖_max^d`, derived from its formulas at the call
//! site, and [`thresholds::homogeneous_safe_range`](super::thresholds::homogeneous_safe_range)
//! turns that into the range of `‖A‖_max` factored unscaled. **An input
//! inside it is factored completely unscaled**, bit-for-bit the unscaled
//! algorithm. Outside it the routine factors `2⁻ᵏ·A`, `k` the minimal move
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

/// `⌈log₂(‖A‖_F / ‖A‖_max)⌉` for the entries `values` whose largest magnitude
/// is `largest`: the power-of-two bound on the ratio that lets a gate state
/// `‖A‖₂ ≤ ‖A‖_F ≤ 2^r·‖A‖_max` tightly (a diagonal-dominated matrix has
/// `r = 0`, where the dimension bound `√(mn)` would charge up to `log₂ n`).
/// Falls back to the dimension bound `⌈⌈log₂ len⌉/2⌉` when the ratio's sum of
/// squares itself leaves the range of `T` (a narrow format at large `len`).
pub(crate) fn norm_ratio_log2<T: RealScalar>(values: &[T], largest: T) -> i32 {
    let sum = values.iter().fold(T::ZERO, |acc, &value| {
        let ratio = value.div(largest);
        acc.add(ratio.mul(ratio))
    });
    let ratio = sum.sqrt();
    if ratio.is_finite() && ratio >= T::ONE {
        thresholds::ceil_log2(ratio)
    } else {
        (thresholds::ceil_log2_count(values.len()) + 1) / 2
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

/// Multiply every entry of `values` by `2^exponent` (exact while the results
/// stay normal).
pub(crate) fn scale_by_power_of_two<T: RealScalar>(values: &mut [T], exponent: i32) {
    for value in values {
        *value = value.scale_binary(exponent);
    }
}

/// The matrix-tier gate over the entries `values`: `Some(k)`, the minimal
/// move, when `‖values‖_max` lies outside
/// [`thresholds::homogeneous_safe_range`]`(degree, f)` with `f =
/// factor_log2(values, ‖values‖_max)`; `None` when it lies inside or `values`
/// is empty, all zero, or holds a non-finite entry.
pub(crate) fn gate_exponent<T: RealScalar>(
    values: &[T],
    degree: u32,
    factor_log2: impl FnOnce(&[T], T) -> i32,
) -> Option<i32> {
    let largest = largest_magnitude(values.iter().copied())?;
    let factor = factor_log2(values, largest);
    balancing_exponent(
        largest,
        thresholds::homogeneous_safe_range::<T>(degree, factor),
    )
}

/// `matrix` multiplied by `2⁻ᵏ` as an owned contiguous copy, `k` from
/// [`gate_exponent`]; `None` when no scaling applies, in which case the
/// caller factors `matrix` itself, bit-for-bit unscaled.
pub(crate) fn balanced<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    degree: u32,
    factor_log2: impl FnOnce(&[T], T) -> i32,
) -> Option<(Array2<T>, i32)> {
    let mut values = match matrix.as_slice() {
        Some(slice) => slice.to_vec(),
        None => matrix.iter().copied().collect(),
    };
    let exponent = gate_exponent(&values, degree, factor_log2)?;
    scale_by_power_of_two(&mut values, -exponent);
    let array = Array2::from_shape_vec(matrix.shape(), values)
        .expect("invariant: a copy of a view keeps its shape and length");
    Some((array, exponent))
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
mod tests {
    use super::{gate_exponent, norm_ratio_log2, restore, scale_by_power_of_two, KernelWindow};
    use crate::application::linalg::thresholds::homogeneous_safe_range;
    use leto::LetoError;

    fn no_factor(_: &[f64], _: f64) -> i32 {
        0
    }

    #[test]
    fn in_range_values_are_not_scaled() {
        // Degree 2, bound 2⁰: f64's range is [√(2⁻⁹⁷⁰), √Ω) ≈ [2⁻⁴⁸⁵, 2⁵¹²).
        let (rmin, rmax) = homogeneous_safe_range::<f64>(2, 0);
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
                gate_exponent(&[value, -value / 3.0], 2, no_factor),
                None,
                "{value} should factor unscaled"
            );
        }
        assert_eq!(gate_exponent(&[0.0_f64, -0.0], 2, no_factor), None);
        assert_eq!(gate_exponent(&[1.0_f64, f64::INFINITY], 2, no_factor), None);
        assert_eq!(gate_exponent(&Vec::<f64>::new(), 2, no_factor), None);
    }

    #[test]
    fn degree_one_upper_end_is_the_overflow_threshold() {
        // Degree 1 takes no root: the upper end is exactly Ω·2⁻ᶠ, so
        // `1.5·2¹⁰²²` (≤ Ω/2) is in range at bound 2¹ and out at 2².
        let value = 1.5 * 2.0_f64.powi(1022);
        assert_eq!(gate_exponent(&[value], 1, |_, _| 1), None);
        assert_eq!(gate_exponent(&[value], 1, |_, _| 2), Some(1));
        let (_, rmax) = homogeneous_safe_range::<f64>(1, 1);
        assert_eq!(rmax, f64::MAX / 2.0);
    }

    #[test]
    fn out_of_range_values_scale_by_the_minimal_move() {
        let (rmin, rmax) = homogeneous_safe_range::<f64>(2, 0);
        let k = gate_exponent(&[1e300_f64], 2, no_factor).expect("1e300 is out of range");
        let mut scaled = [1e300_f64];
        scale_by_power_of_two(&mut scaled, -k);
        assert!(scaled[0] <= rmax, "{} > {rmax}", scaled[0]);
        let mut one_less = [1e300_f64];
        scale_by_power_of_two(&mut one_less, -(k - 1));
        assert!(one_less[0] > rmax, "{k} was not minimal");

        let tiny = f64::from_bits(1);
        let k = gate_exponent(&[tiny], 2, no_factor).expect("subnormal is out of range");
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
        let (rmin0, rmax0) = homogeneous_safe_range::<f64>(4, 0);
        let (rmin, rmax) = homogeneous_safe_range::<f64>(4, 1000);
        assert_eq!(rmin, rmin0);
        assert!(rmax < rmax0);
        assert_eq!(gate_exponent(&[100.0_f64], 4, no_factor), None);
        assert!(gate_exponent(&[100.0_f64], 4, |_, _| 1000).is_some());
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
