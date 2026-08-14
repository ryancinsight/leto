//! Statistical quality metrics and distribution summaries.
//!
//! Pure functions with no domain dependencies. Both solver and clinical layers
//! import from here; neither depends on the other.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use eunomia::RealField;

/// Pearson product-moment correlation coefficient.
///
/// Returns `T::ZERO` when either slice has zero variance, when the slices have
/// different lengths, or when fewer than 2 samples are provided.
#[must_use]
pub fn pearson<T: RealField>(a: &[T], b: &[T]) -> T {
    if a.len() != b.len() || a.len() < 2 {
        return T::ZERO;
    }
    let n = T::from_f64(a.len() as f64);
    let ma = a.iter().copied().fold(T::ZERO, |acc, v| acc + v) / n;
    let mb = b.iter().copied().fold(T::ZERO, |acc, v| acc + v) / n;
    let mut num = T::ZERO;
    let mut da = T::ZERO;
    let mut db = T::ZERO;
    for (&av, &bv) in a.iter().zip(b) {
        let xa = av - ma;
        let xb = bv - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if da > T::ZERO && db > T::ZERO {
        num / (da.sqrt() * db.sqrt())
    } else {
        T::ZERO
    }
}

/// Pearson correlation of same-frequency sinusoids separated by phase.
///
/// For `A(t) = sin(omega t)` and `B(t) = sin(omega t + phi)`, the Pearson
/// coefficient is `cos(phi)`. Inputs are phase offsets in radians.
///
/// # Errors
///
/// Returns an error when any phase value is non-finite.
pub fn phase_shift_correlation_curve<T: RealField>(phase_rad: &[T]) -> Result<Vec<T>, String> {
    if let Some((index, value)) = phase_rad
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!("phase_rad[{index}] must be finite, got {value:?}"));
    }

    Ok(phase_rad.iter().copied().map(|phase| phase.cos()).collect())
}

/// Phase error in degrees for a target same-frequency sinusoid correlation.
///
/// This is the inverse of `r(phi) = cos(phi)` over `0 <= phi <= pi`.
///
/// # Errors
///
/// Returns an error when `correlation` is non-finite or outside `[-1, 1]`.
pub fn phase_error_degrees_for_correlation<T: RealField>(correlation: T) -> Result<T, String> {
    if !correlation.is_finite() {
        return Err(format!("correlation must be finite, got {correlation:?}"));
    }
    if correlation < -T::ONE || correlation > T::ONE {
        return Err(format!(
            "correlation must be in [-1, 1], got {correlation:?}"
        ));
    }

    Ok(correlation.acos().to_degrees())
}

/// PSNR in dB from relative RMSE values.
///
/// For `relative_rmse = RMSE / MAX`, `PSNR = -20 * log10(relative_rmse)`.
///
/// # Errors
///
/// Returns an error when any relative RMSE is non-finite or non-positive.
pub fn validation_psnr_from_relative_rmse<T: RealField>(
    relative_rmse: &[T],
) -> Result<Vec<T>, String> {
    if let Some((index, value)) = relative_rmse
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value <= T::ZERO)
    {
        return Err(format!(
            "relative_rmse[{index}] must be finite and positive, got {value:?}"
        ));
    }

    let scale = T::from_f64(-20.0);
    Ok(relative_rmse
        .iter()
        .copied()
        .map(|value| scale * value.log10())
        .collect())
}

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
        / T::from_f64(a.len() as f64);
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
        / T::from_f64(a.len() as f64);
    mse.sqrt()
}

/// Peak signal-to-noise ratio in dB between simulation `a` and reference `b`
/// (Chapter 19 §19.3):
///
/// `PSNR = 20·log₁₀(MAX_B / RMSE(a, b))`, where `MAX_B = max|bᵢ|` is the peak
/// magnitude of the reference `b` (Chapter 19 §19.3 — the absolute value handles
/// bipolar signals such as acoustic pressure).
///
/// Returns `T::INFINITY` for an exact match (`RMSE = 0`, infinite fidelity) and
/// `T::ZERO` for degenerate inputs (length mismatch, empty, or an all-zero reference
/// for which the dB ratio is undefined).
#[must_use]
pub fn psnr<T: RealField>(a: &[T], b: &[T]) -> T {
    if a.len() != b.len() || a.is_empty() {
        return T::ZERO;
    }
    let max_b = b
        .iter()
        .copied()
        .fold(T::ZERO, |m, v| m.max_scalar(v.abs()));
    if max_b <= T::ZERO {
        return T::ZERO;
    }
    let err = rmse(a, b);
    if err == T::ZERO {
        return T::infinity();
    }
    T::from_f64(20.0) * (max_b / err).log10()
}

/// Inter-percentile range P95 − P05 of a value distribution.
///
/// Returns `T::ZERO` for fewer than 2 samples.
#[must_use]
pub fn percentile_range<T: RealField>(mut values: Vec<T>) -> T {
    if values.len() < 2 {
        return T::ZERO;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let last = values.len() - 1;
    let p05 = values[(0.05 * last as f64).round() as usize];
    let p95 = values[(0.95 * last as f64).round() as usize];
    p95 - p05
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_perfect_correlation() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson(&a, &b) - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn pearson_anticorrelated() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let b: Vec<f64> = vec![3.0, 2.0, 1.0];
        assert!((pearson(&a, &b) + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn pearson_constant_returns_zero() {
        let a: Vec<f64> = vec![1.0, 1.0, 1.0];
        let b: Vec<f64> = vec![2.0, 3.0, 4.0];
        assert_eq!(pearson(&a, &b), 0.0);
    }

    #[test]
    fn pearson_length_mismatch_returns_zero() {
        assert_eq!(pearson(&[1.0_f64, 2.0], &[1.0_f64]), 0.0_f64);
    }

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

    #[test]
    fn psnr_matches_definition_and_limits() {
        // MAX_B = 4, RMSE = √(1/3) ⇒ PSNR = 20·log₁₀(4/√(1/3)).
        let a = [1.0_f64, 2.0, 3.0];
        let b = [1.0_f64, 2.0, 4.0];
        let expected = 20.0 * (4.0 / (1.0_f64 / 3.0).sqrt()).log10();
        assert!(
            (psnr(&a, &b) - expected).abs() < 1e-12,
            "psnr = {}",
            psnr(&a, &b)
        );
        // Exact match ⇒ infinite fidelity.
        assert!(psnr(&a, &a).is_infinite());
        // A 40 dB target corresponds to RMSE = MAX_B / 100: peak 1.0, err 0.01.
        let r: Vec<f64> = vec![1.0; 100];
        let mut s = r.clone();
        s[0] = 1.0 - 0.01 * (100.0_f64).sqrt(); // single-sample error giving RMSE = 0.01
        assert!(
            (psnr(&s, &r) - 40.0).abs() < 1e-9,
            "psnr = {}",
            psnr(&s, &r)
        );
        // MAX_B uses peak magnitude, so a bipolar reference works: peak |−4| = 4.
        let bi_ref = [1.0_f64, -4.0, 2.0];
        let bi_sim = [1.0_f64, -4.0, 2.0];
        assert!(psnr(&bi_sim, &bi_ref).is_infinite());
        let bi_err = [1.0_f64, -3.0, 2.0]; // err only at the |−4| sample
        assert!(psnr(&bi_err, &bi_ref) > 0.0 && psnr(&bi_err, &bi_ref).is_finite());
        // Degenerate guards.
        assert_eq!(psnr(&[1.0_f64, 2.0], &[1.0_f64]), 0.0_f64); // length mismatch
        assert_eq!(psnr(&[0.0_f64, 0.0], &[0.0_f64, 0.0]), 0.0_f64); // all-zero reference
    }

    /// §19.2 phase-sensitivity theorem: for A = sin(kx), B = sin(kx + φ) the
    /// Pearson correlation equals cos(φ).
    #[test]
    fn pearson_equals_cosine_of_phase_shift() {
        use std::f64::consts::PI;
        let n = 2000;
        let k = 2.0 * PI / n as f64; // one period over the window
        for &phi in &[0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI] {
            let a: Vec<f64> = (0..n).map(|i| (k * i as f64).sin()).collect();
            let b: Vec<f64> = (0..n).map(|i| (k * i as f64 + phi).sin()).collect();
            assert!(
                (pearson(&a, &b) - phi.cos()).abs() < 1e-3,
                "phase {phi}: r={} vs cos={}",
                pearson(&a, &b),
                phi.cos()
            );
        }
    }

    #[test]
    fn phase_shift_correlation_curve_matches_theorem_samples() {
        use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4};
        let phase = [0.0, FRAC_PI_4, FRAC_PI_2];
        let observed = phase_shift_correlation_curve(&phase).unwrap();
        let expected = [1.0, FRAC_1_SQRT_2, 0.0];

        for (actual, expected) in observed.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 2.0 * f64::EPSILON,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn phase_shift_correlation_curve_rejects_nonfinite_phase() {
        let err = phase_shift_correlation_curve(&[0.0, f64::NAN]).unwrap_err();
        assert!(err.contains("phase_rad[1] must be finite"));
    }

    #[test]
    fn phase_error_degrees_for_correlation_matches_inverse_theorem() {
        let observed = phase_error_degrees_for_correlation(0.99).unwrap();
        let expected = 0.99_f64.acos().to_degrees();

        assert!((observed - expected).abs() <= f64::EPSILON);
        assert_eq!(phase_error_degrees_for_correlation(1.0).unwrap(), 0.0);
        assert_eq!(phase_error_degrees_for_correlation(-1.0).unwrap(), 180.0);
    }

    #[test]
    fn phase_error_degrees_for_correlation_rejects_invalid_correlation() {
        let high = phase_error_degrees_for_correlation(1.01).unwrap_err();
        assert!(high.contains("correlation must be in [-1, 1]"));

        let nonfinite = phase_error_degrees_for_correlation(f64::INFINITY).unwrap_err();
        assert!(nonfinite.contains("correlation must be finite"));
    }

    #[test]
    fn validation_psnr_from_relative_rmse_matches_definition() {
        let observed: Vec<f64> =
            validation_psnr_from_relative_rmse(&[1.0_f64, 0.1, 0.01, 0.001]).unwrap();
        let expected = [0.0_f64, 20.0, 40.0, 60.0];

        for (actual, expected) in observed.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 8.0 * f64::EPSILON,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn validation_psnr_from_relative_rmse_rejects_invalid_error() {
        let zero = validation_psnr_from_relative_rmse(&[0.0_f64]).unwrap_err();
        assert!(zero.contains("relative_rmse[0] must be finite and positive"));

        let nonfinite = validation_psnr_from_relative_rmse(&[f64::NAN]).unwrap_err();
        assert!(nonfinite.contains("relative_rmse[0] must be finite and positive"));
    }

    #[test]
    fn percentile_range_monotone() {
        let v: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let r = percentile_range(v);
        assert!(r > 0.0 && r <= 9.0);
    }

    #[test]
    fn percentile_range_single_element() {
        assert_eq!(percentile_range(vec![42.0_f64]), 0.0_f64);
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

    /// Elementwise tolerance for an `n`-element accumulation in `T`:
    /// `c(n)·ε` with a small constant, per the standard summation bound.
    fn accumulation_tolerance<T: RealField>(n: usize) -> T {
        T::from_f64(4.0 * n as f64) * T::EPSILON
    }

    fn pearson_contract<T: RealField>() {
        // Exactly proportional series: the correlation is 1 by definition.
        let a: Vec<T> = (1..=5).map(|i| T::from_f64(f64::from(i))).collect();
        let b: Vec<T> = (1..=5).map(|i| T::from_f64(f64::from(2 * i))).collect();
        let deviation = (pearson(&a, &b) - T::ONE).abs();
        assert!(
            deviation < accumulation_tolerance::<T>(a.len()),
            "pearson deviated by {deviation:?} at this width"
        );
    }

    fn rmse_contract<T: RealField>() {
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

    fn psnr_contract<T: RealField>() {
        // Identical signals carry no error, so PSNR is unbounded (infinite).
        let reference: Vec<T> = vec![T::from_f64(1.0); 4];
        assert!(!psnr(&reference, &reference).is_finite());
    }

    #[test]
    fn pearson_is_generic_over_scalar() {
        pearson_contract::<f32>();
        pearson_contract::<f64>();
    }

    #[test]
    fn rmse_is_generic_over_scalar() {
        rmse_contract::<f32>();
        rmse_contract::<f64>();
    }

    #[test]
    fn psnr_is_generic_over_scalar() {
        psnr_contract::<f32>();
        psnr_contract::<f64>();
    }
}
