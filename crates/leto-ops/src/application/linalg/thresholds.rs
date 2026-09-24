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
    use super::{machine_epsilon, rank_pivot_ratio, scaled_frobenius};
    use eunomia::{Bf16, F16};

    #[test]
    fn machine_epsilon_matches_each_format() {
        assert_eq!(machine_epsilon::<f64>(), f64::EPSILON);
        assert_eq!(machine_epsilon::<f32>(), f32::EPSILON);
        assert_eq!(machine_epsilon::<F16>().to_f32(), 2.0_f32.powi(-10));
        assert_eq!(machine_epsilon::<Bf16>().to_f32(), 2.0_f32.powi(-7));
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
