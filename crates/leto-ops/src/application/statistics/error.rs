//! Root-mean-square error and its normalizations.

use eunomia::RealField;

/// RMSE of `b` relative to `a`, normalised by `‖a‖₂`.
///
/// Measures error energy relative to the reference signal energy.
/// Returns `T::ZERO` when `a` is the zero vector.
#[must_use]
pub fn normalized_rmse<T: RealField>(a: &[T], b: &[T]) -> T {
    let norm = a.iter().copied().fold(T::ZERO, |acc, v| acc + v * v).sqrt();
    if norm == T::ZERO {
        return T::ZERO;
    }
    let err = a
        .iter()
        .zip(b)
        .map(|(&av, &bv)| {
            let d = av - bv;
            d * d
        })
        .fold(T::ZERO, |acc, v| acc + v)
        .sqrt();
    err / norm
}

/// RMSE of `b` relative to `a`, normalised by the dynamic range of `a`.
///
/// Measures error relative to the signal's peak-to-peak span.
/// Returns `T::ZERO` when slices differ in length, are empty, or `a` has zero range.
#[must_use]
pub fn nrmse<T: RealField>(a: &[T], b: &[T]) -> T {
    if a.len() != b.len() || a.is_empty() {
        return T::ZERO;
    }
    let mse = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).powi(2))
        .fold(T::ZERO, |acc, v| acc + v)
        / T::from_count(a.len());
    let max_a = a
        .iter()
        .copied()
        .fold(T::neg_infinity(), |acc, v| acc.max_scalar(v));
    let min_a = a
        .iter()
        .copied()
        .fold(T::infinity(), |acc, v| acc.min_scalar(v));
    let span = (max_a - min_a).abs();
    let floor = T::from_f64(1.0e-12);
    mse.sqrt() / span.max_scalar(floor)
}

/// Absolute root-mean-square error between `a` and `b`,
/// `RMSE = √(mean((aᵢ − bᵢ)²))`.
///
/// Returns `T::ZERO` when the slices differ in length or are empty.
#[must_use]
pub fn rmse<T: RealField>(a: &[T], b: &[T]) -> T {
    if a.len() != b.len() || a.is_empty() {
        return T::ZERO;
    }
    let mse = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).powi(2))
        .fold(T::ZERO, |acc, v| acc + v)
        / T::from_count(a.len());
    mse.sqrt()
}

#[cfg(test)]
mod tests {
    use super::{normalized_rmse, nrmse, rmse};
    use crate::application::statistics::accumulation_tolerance;

    #[test]
    fn normalized_rmse_perfect_match() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert_eq!(normalized_rmse(&a, &a), 0.0);
    }

    #[test]
    fn normalized_rmse_zero_reference() {
        let a: Vec<f64> = vec![0.0, 0.0, 0.0];
        let b: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert_eq!(normalized_rmse(&a, &b), 0.0);
    }

    #[test]
    fn nrmse_perfect_match() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert_eq!(nrmse(&a, &a), 0.0);
    }

    #[test]
    fn rmse_matches_definition() {
        // diffs [0, 0, 1] ⇒ MSE = 1/3 ⇒ RMSE = √(1/3).
        let a = [1.0_f64, 2.0, 3.0];
        let b = [1.0_f64, 2.0, 4.0];
        assert!((rmse(&a, &b) - (1.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(rmse(&a, &a), 0.0_f64);
        assert_eq!(rmse(&[1.0_f64, 2.0], &[1.0_f64]), 0.0_f64); // length mismatch
        assert_eq!(rmse::<f64>(&[], &[]), 0.0_f64);
    }

    // ── scalar-type genericity ──────────────────────────────────────────────
    //
    // Each contract is written once over `T: RealField` and instantiated
    // across every scalar the stack ships, so a newly admitted type
    // inherits the suite instead of needing the assertions copied again.
    // Tolerances derive from `T::EPSILON` rather than being fixed per
    // type: the algorithms accumulate over `n` elements, so a few
    // multiples of the type's epsilon is the bound, and the same
    // derivation holds at every width.

    fn rmse_contract<T: eunomia::RealField>() {
        // One unit of error in one of three samples: rmse = sqrt(1/3).
        let a: Vec<T> = (1..=3).map(|i| T::from_f64(f64::from(i))).collect();
        let mut b = a.clone();
        b[2] += T::ONE;
        let expected = (T::from_f64(1.0 / 3.0)).sqrt();
        let deviation = (rmse(&a, &b) - expected).abs();
        assert!(
            deviation < accumulation_tolerance::<T>(a.len()),
            "rmse deviated by {deviation:?} at this width"
        );
    }

    #[test]
    fn rmse_is_generic_over_scalar() {
        rmse_contract::<f32>();
        rmse_contract::<f64>();
    }
}
