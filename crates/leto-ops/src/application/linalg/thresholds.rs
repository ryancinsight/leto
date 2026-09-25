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

/// The generalized LAPACK safe range for a routine whose largest relied-upon
/// intermediate is homogeneous of degree `degree` in the input entries and
/// bounded above by `dimension_factor · ‖A‖_max^degree` (a derived bound —
/// every call site cites its own derivation): the range of `‖A‖_max` keeping
/// that intermediate between `safmin/ε` (below which relative precision
/// degrades to gradual underflow) and `ε/safmin` (above which it risks
/// overflow).
///
/// [`safe_range`]'s `rmin = √smlnum, rmax = √bignum` is the `degree = 2`,
/// `dimension_factor = 1` case (`dsyev.f`'s `ISCALE` block: a single entry
/// product, no extra dimension factor). This generalizes it: an intermediate
/// bounded by `c·A_max^d` needs `smlnum ≤ c·A_max^d ≤ bignum`, i.e.
/// `A_max ∈ [(smlnum/c)^(1/d), (bignum/c)^(1/d)]`.
///
/// `degree` must be a power of two (`1`, `2`, or `4` — every routine here
/// needs one of those), computed by iterated `sqrt` so every step stays
/// exact for values that are themselves exact (as `smlnum`/`bignum`/`c` are
/// for the integer or power-of-two `dimension_factor`s used here).
pub(crate) fn homogeneous_safe_range<T: RealScalar>(degree: u32, dimension_factor: T) -> (T, T) {
    debug_assert!(
        degree == 1 || degree == 2 || degree == 4,
        "homogeneous_safe_range: degree must be 1, 2, or 4"
    );
    let smlnum = safe_min::<T>().div(machine_epsilon::<T>());
    let bignum = T::ONE.div(smlnum);
    let mut rmin = smlnum.div(dimension_factor);
    let mut rmax = bignum.div(dimension_factor);
    let mut remaining = degree;
    while remaining > 1 {
        rmin = rmin.sqrt();
        rmax = rmax.sqrt();
        remaining /= 2;
    }
    (rmin, rmax)
}

/// [`homogeneous_safe_range`] at `degree = 2`, `dimension_factor = 1` —
/// LAPACK `dsyev`/`dsteqr`'s own range, `rmin = √smlnum`, `rmax = √bignum`
/// (`smlnum = safmin/ε`, `bignum = 1/smlnum`). Kept as a test fixture for the
/// symmetry/derivation checks below; production call sites derive their own
/// `(degree, dimension_factor)` from their actual formulas instead of
/// assuming this reference case applies.
#[cfg(test)]
pub(crate) fn safe_range<T: RealScalar>() -> (T, T) {
    homogeneous_safe_range::<T>(2, T::ONE)
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
    use super::{machine_epsilon, rank_pivot_ratio, safe_min, safe_range, scaled_frobenius};
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
    fn safe_range_is_symmetric_around_one_in_log_space() {
        // rmin·rmax = √(smlnum·bignum) = √(smlnum · ε/smlnum) = 1 exactly:
        // `rmax = 1/rmin` (checked to a few ULP against the derived product).
        let (rmin, rmax) = safe_range::<f64>();
        assert!(rmin > 0.0 && rmin < 1.0, "{rmin}");
        assert!(rmax > 1.0, "{rmax}");
        assert!(
            (rmin * rmax - 1.0).abs() <= 8.0 * f64::EPSILON,
            "{rmin} * {rmax}"
        );
        let (rmin, rmax) = safe_range::<f32>();
        assert!(rmin > 0.0 && rmin < 1.0, "{rmin}");
        assert!(rmax > 1.0, "{rmax}");
        assert!(
            (f64::from(rmin) * f64::from(rmax) - 1.0).abs() <= 8.0 * f64::from(f32::EPSILON),
            "{rmin} * {rmax}"
        );
        // F16's 5-bit exponent gives the narrowest safe range of the four
        // shipped scalars: rmin = √(safmin/ε) = √(2⁻¹⁴/2⁻¹⁰) = √(2⁻⁴) = 0.25,
        // rmax = √(ε/safmin) = √(2⁴) = 4.
        let (rmin, rmax) = safe_range::<F16>();
        assert_eq!(rmin.to_f32(), 0.25);
        assert_eq!(rmax.to_f32(), 4.0);
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
