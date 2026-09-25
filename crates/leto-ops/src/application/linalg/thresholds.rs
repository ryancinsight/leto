//! Shared relative thresholds for dense linear algebra kernels.

use crate::domain::real::RealScalar;

/// Denominator for the default rank and pivot relative threshold.
///
/// The value is expressed as `f64` at the conversion boundary because the
/// scalar contract's `from_usize` constructor cannot represent this value on a
/// 32-bit target. The resulting threshold remains in the caller's native
/// precision.
const RANK_PIVOT_DENOMINATOR: f64 = 1_000_000_000_000.0;

/// Return the default relative rank and pivot threshold (`1e-12`) in `T`.
#[inline]
pub(crate) fn rank_pivot_ratio<T: RealScalar>() -> T {
    T::ONE.div(T::from_f64(RANK_PIVOT_DENOMINATOR))
}

/// Machine epsilon of `T`: the gap between `1` and the next representable
/// value, found by halving from `1` until `1 + ε/2` rounds back to `1`.
///
/// Exact for every binary format (each candidate is a power of two), and
/// computed through `T`'s own addition, so it is the epsilon of the arithmetic
/// the kernels actually perform. `O(p)` for a `p`-bit significand.
pub(crate) fn machine_epsilon<T: RealScalar>() -> T {
    let half = T::ONE.div(T::from_usize(2));
    let mut epsilon = T::ONE;
    while T::ONE.add(epsilon.mul(half)) > T::ONE {
        epsilon = epsilon.mul(half);
    }
    epsilon
}

/// Smallest positive **normalized** value of `T` — LAPACK `SAFMIN`/Fortran
/// `TINY` (`dlamch('S')`): the point below which relative precision degrades
/// to gradual underflow.
///
/// Found by halving from `1` while the candidate one step further down still
/// resolves a full relative `ε` step (`x + x·ε ≠ x`): a normal `x` has ULP
/// spacing `x·ε`, so the probe succeeds throughout the normal range and fails
/// the instant `x·ε` itself underflows to `0` in the subnormal range, which
/// happens exactly at the smallest subnormal — one format-independent
/// criterion computed entirely through `T`'s own arithmetic (no per-format
/// exponent-width constant). `O(p)` for a `p`-bit exponent range.
pub(crate) fn safe_min<T: RealScalar>() -> T {
    let eps = machine_epsilon::<T>();
    let two = T::from_usize(2);
    let mut candidate = T::ONE;
    loop {
        let halved = candidate.div(two);
        if halved == T::ZERO || halved.add(halved.mul(eps)) == halved {
            return candidate;
        }
        candidate = halved;
    }
}

/// The largest finite value of `T`, `Ω = (2 − ε)·2^emax` (LAPACK
/// `dlamch('O')`), found by doubling from `1` to the largest finite power of
/// two — computed through `T`'s own arithmetic like [`safe_min`]. (The scalar
/// contract's `MAX_VALUE` is the reduction identity `+∞`, not this.)
pub(crate) fn overflow_threshold<T: RealScalar>() -> T {
    let two = T::from_usize(2);
    let mut power = T::ONE;
    loop {
        let next = power.mul(two);
        if !next.is_finite() {
            return power.mul(two.sub(machine_epsilon::<T>()));
        }
        power = next;
    }
}

/// Which side of the exact root [`root`] may land on.
#[derive(Clone, Copy)]
enum RootRounding {
    /// `root^degree ≤ y`: an upper bound that must not be exceeded.
    NotAbove,
    /// `root^degree ≥ y`: a lower bound that must not be undercut.
    NotBelow,
}

/// `y^(1/degree)` for `degree ∈ {1, 2, 4}` by iterated `sqrt`, moved one
/// relative `ε` to the requested side when `degree > 1`: each correctly
/// rounded `sqrt` errs by at most half an ulp, so one `ε` step covers the
/// (at most two) roundings.
fn root<T: RealScalar>(y: T, degree: u32, rounding: RootRounding) -> T {
    debug_assert!(
        degree == 1 || degree == 2 || degree == 4,
        "root: degree must be 1, 2, or 4"
    );
    let mut value = y;
    let mut remaining = degree;
    while remaining > 1 {
        value = value.sqrt();
        remaining /= 2;
    }
    if degree == 1 {
        return value;
    }
    let step = value.mul(machine_epsilon::<T>());
    match rounding {
        RootRounding::NotAbove => value.sub(step),
        RootRounding::NotBelow => value.add(step),
    }
}

/// The range of `‖A‖_max` inside which a routine factors `A` unscaled
/// (the matrix-level gate): its largest relied-upon intermediate is
/// homogeneous of degree `degree` in the entries and bounded by
/// `2^factor_log2 · ‖A‖_max^degree` (each call site derives its own bound).
///
/// - Upper end: the intermediate stays finite, `2^f·‖A‖_max^d ≤ Ω` (`Ω` the
///   overflow threshold), i.e. `‖A‖_max ≤ (Ω·2^−f)^(1/d)`. The bound factor is
///   applied by exponent arithmetic (`scale_binary`), never formed in `T`, so
///   a factor such as `n⁴` cannot overflow the narrow formats.
/// - Lower end: an intermediate of magnitude `‖A‖_max^d` keeps its
///   `ε`-relative rounding normal, `‖A‖_max^d ≥ smlnum = safmin/ε` (LAPACK
///   `dsyev`/`dgeev` `SMLNUM`), i.e. `‖A‖_max ≥ smlnum^(1/d)`. The upper-bound
///   factor plays no part here: it bounds the intermediate from above, and
///   dividing `smlnum` by it would lower `rmin` and admit exactly the inputs
///   whose intermediates underflow. Scaling *up* into this end is exact (no
///   entry can underflow), so the lower end costs no precision.
/// - Deflation floor: a routine whose convergence test also deflates below an
///   absolute threshold `2^g·safmin` raises the lower end toward `2^g·smlnum`,
///   so that threshold is at most `ε·‖A‖_max` and each such deflation stays
///   inside the backward error. Where that would pass the upper end (a narrow
///   format at a large order — `F16` has only `Ω/smlnum ≈ 2²⁰` of headroom),
///   the lower end stops at the upper end: overflow is the hard constraint,
///   and the floor then costs at most `2^g·safmin` per deflation.
///
/// `None` when even the unraised range is empty, `smlnum^(1/d) > (Ω·2^−f)^(1/d)`:
/// the bound factor `2^f` itself exceeds `Ω/smlnum`, so no power-of-two
/// scaling keeps both ends — the order is past what the format can factor
/// with these intermediates.
pub(crate) fn homogeneous_safe_range<T: RealScalar>(
    degree: u32,
    factor_log2: i32,
    floor_log2: i32,
) -> Option<(T, T)> {
    let smlnum = safe_min::<T>().div(machine_epsilon::<T>());
    let root_end = root(smlnum, degree, RootRounding::NotBelow);
    let upper = root(
        overflow_threshold::<T>().scale_binary(-factor_log2),
        degree,
        RootRounding::NotAbove,
    );
    if root_end > upper {
        return None;
    }
    let floor_end = smlnum.scale_binary(floor_log2);
    let floor_end = if floor_end < upper { floor_end } else { upper };
    let lower = if floor_end > root_end {
        floor_end
    } else {
        root_end
    };
    Some((lower, upper))
}

/// The window of a kernel-local magnitude `m` inside which a kernel forms its
/// products unscaled (the LAPACK `dlartg`/`dnrm2` "medium" range, `rtmin =
/// √safmin`, `rtmax = √(safmax/2)` for a two-term sum of squares): the
/// smallest relied-upon product, of degree `lower_degree`, stays normal,
/// `m^dₗ ≥ safmin`; the largest, of degree `upper_degree` and bounded by
/// `2^factor_log2·m^dᵤ`, stays finite. Outside it the kernel rescales its
/// local operands by a power of two (exact) before forming them.
pub(crate) fn kernel_window<T: RealScalar>(
    lower_degree: u32,
    upper_degree: u32,
    factor_log2: i32,
) -> (T, T) {
    (
        root(safe_min::<T>(), lower_degree, RootRounding::NotBelow),
        root(
            overflow_threshold::<T>().scale_binary(-factor_log2),
            upper_degree,
            RootRounding::NotAbove,
        ),
    )
}

/// `⌈log₂ x⌉` for a finite `x ≥ 1`: the exponent of the smallest power of two
/// not below `x`, found from `x`'s binary exponent (exponent arithmetic, no
/// product that could overflow).
pub(crate) fn ceil_log2<T: RealScalar>(x: T) -> i32 {
    let exponent = x.binary_exponent().unwrap_or(0);
    if T::ONE.scale_binary(exponent) == x {
        exponent
    } else {
        exponent + 1
    }
}

/// `⌈log₂ n⌉` for a count `n ≥ 1`, in integer arithmetic.
pub(crate) fn ceil_log2_count(n: usize) -> i32 {
    let n = n.max(1);
    let floor = usize::BITS - 1 - n.leading_zeros();
    let floor = i32::try_from(floor).expect("invariant: a bit index fits in i32");
    if n.is_power_of_two() {
        floor
    } else {
        floor + 1
    }
}

/// [`homogeneous_safe_range`] at `degree = 2`, `factor_log2 = 0` — the
/// LAPACK `dsyev` lower end `√smlnum` with an overflow-threshold upper end.
/// Kept as a test fixture for the derivation checks below.
#[cfg(test)]
pub(crate) fn safe_range<T: RealScalar>() -> (T, T) {
    homogeneous_safe_range::<T>(2, 0, 0).expect("invariant: degree 2, no bound factor")
}

/// `‖A‖_F` of `values` without overflow or underflow in the squares: the
/// entries are divided by their largest magnitude before squaring.
pub(crate) fn scaled_frobenius<T: RealScalar>(values: &[T]) -> T {
    let largest = values.iter().fold(T::ZERO, |acc, &v| {
        let magnitude = v.abs();
        if magnitude > acc {
            magnitude
        } else {
            acc
        }
    });
    if largest == T::ZERO {
        return T::ZERO;
    }
    let sum = values.iter().fold(T::ZERO, |acc, &v| {
        let ratio = v.div(largest);
        acc.add(ratio.mul(ratio))
    });
    largest.mul(sum.sqrt())
}

#[cfg(test)]
mod tests {
    use super::{
        ceil_log2, ceil_log2_count, homogeneous_safe_range, kernel_window, machine_epsilon,
        rank_pivot_ratio, safe_min, safe_range, scaled_frobenius,
    };
    use eunomia::{Bf16, F16};

    #[test]
    fn machine_epsilon_matches_each_format() {
        assert_eq!(machine_epsilon::<f64>(), f64::EPSILON);
        assert_eq!(machine_epsilon::<f32>(), f32::EPSILON);
        assert_eq!(machine_epsilon::<F16>().to_f32(), 2.0_f32.powi(-10));
        assert_eq!(machine_epsilon::<Bf16>().to_f32(), 2.0_f32.powi(-7));
    }

    #[test]
    fn safe_min_matches_each_format_smallest_normal() {
        assert_eq!(safe_min::<f64>(), f64::MIN_POSITIVE);
        assert_eq!(safe_min::<f32>(), f32::MIN_POSITIVE);
        assert_eq!(safe_min::<F16>().to_f32(), F16::MIN_POSITIVE.to_f32());
        assert_eq!(safe_min::<Bf16>().to_f32(), Bf16::MIN_POSITIVE.to_f32());
    }

    #[test]
    fn safe_range_ends_bound_the_intermediate_on_the_safe_side() {
        // Degree 2, bound 2⁰: rmin² ≥ smlnum = safmin/ε and rmax² ≤ Ω, each
        // within two roundings of equality (the roots are stepped one ε
        // toward the safe side).
        let (rmin, rmax) = safe_range::<f64>();
        let smlnum = f64::MIN_POSITIVE / f64::EPSILON;
        assert!(rmin * rmin >= smlnum && rmin * rmin <= smlnum * (1.0 + 4.0 * f64::EPSILON));
        assert!(rmax * rmax <= f64::MAX && rmax * rmax >= f64::MAX * (1.0 - 4.0 * f64::EPSILON));
        // F16: rmin = √(2⁻¹⁴/2⁻¹⁰) = 0.25 (stepped up one ε), rmax just below
        // √65504 ≈ 255.94.
        let (rmin, rmax) = safe_range::<F16>();
        let (rmin, rmax) = (f64::from(rmin.to_f32()), f64::from(rmax.to_f32()));
        let eps = f64::from(2.0_f32.powi(-10));
        assert!((0.25..=0.25 * (1.0 + 2.0 * eps)).contains(&rmin), "{rmin}");
        assert!(
            rmax * rmax <= 65504.0 && rmax >= 255.94 * (1.0 - 2.0 * eps),
            "{rmax}"
        );
        // The bound factor divides only the overflow side.
        let (rmin4, rmax4) = homogeneous_safe_range::<F16>(2, 4, 0).expect("non-empty");
        assert_eq!(f64::from(rmin4.to_f32()), rmin);
        assert!(f64::from(rmax4.to_f32()) * f64::from(rmax4.to_f32()) <= 65504.0 / 16.0);
        // A bound such as 128·n⁴ at n = 100 (≈ 2³³·⁶) exceeds F16's range as a
        // value but not as an exponent: the upper end is small and finite.
        // …and a factor of 2²⁰, past `Ω/smlnum = 65504·2⁴`, empties the range:
        // reported, never a lower end above the upper one.
        assert!(homogeneous_safe_range::<F16>(1, 20, 0).is_none());
        let (lower, upper) = homogeneous_safe_range::<F16>(1, 19, 0).expect("non-empty");
        assert!(lower <= upper);
        // A deflation floor past the upper end stops at it.
        let (lower, upper) = homogeneous_safe_range::<F16>(1, 12, 30).expect("non-empty");
        assert_eq!(lower.to_f32(), upper.to_f32());
    }

    #[test]
    fn kernel_window_matches_the_dlartg_thresholds() {
        // dlartg: rtmin = √safmin, rtmax = √(safmax/2) (Ω here).
        let (low, high) = kernel_window::<f64>(2, 2, 1);
        assert!(
            low >= 2.0_f64.powi(-511) && low <= 2.0_f64.powi(-511) * (1.0 + 2.0 * f64::EPSILON)
        );
        assert!(high * high * 2.0 <= f64::MAX);
        assert!(high >= (f64::MAX / 2.0).sqrt() * (1.0 - 2.0 * f64::EPSILON));
    }

    #[test]
    fn ceil_log2_rounds_up_to_the_next_power_of_two() {
        assert_eq!(ceil_log2(1.0_f64), 0);
        assert_eq!(ceil_log2(4.0_f64), 2);
        assert_eq!(ceil_log2(4.5_f64), 3);
        assert_eq!(ceil_log2_count(1), 0);
        assert_eq!(ceil_log2_count(3), 2);
        assert_eq!(ceil_log2_count(4), 2);
        assert_eq!(ceil_log2_count(5), 3);
    }

    #[test]
    fn scaled_frobenius_survives_the_range_ends() {
        // 3·(1e30)² overflows f32 and 3·(1e-30)² underflows it.
        let large = [1e30_f32, 1e30, 1e30];
        let small = [1e-30_f32, 1e-30, 1e-30];
        let root3 = 3.0_f32.sqrt();
        assert!((scaled_frobenius(&large) / 1e30 - root3).abs() <= 4.0 * f32::EPSILON);
        assert!((scaled_frobenius(&small) / 1e-30 - root3).abs() <= 4.0 * f32::EPSILON);
        assert_eq!(scaled_frobenius(&[0.0_f64; 4]), 0.0);
    }

    #[test]
    fn rank_pivot_ratio_matches_floating_point_contract() {
        assert_eq!(rank_pivot_ratio::<f64>(), 1.0e-12);
        assert_eq!(rank_pivot_ratio::<f32>(), 1.0e-12_f32);
    }
}
