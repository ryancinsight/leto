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

#[cfg(test)]
mod tests {
    use super::rank_pivot_ratio;

    #[test]
    fn rank_pivot_ratio_matches_floating_point_contract() {
        assert_eq!(rank_pivot_ratio::<f64>(), 1.0e-12);
        assert_eq!(rank_pivot_ratio::<f32>(), 1.0e-12_f32);
    }
}
