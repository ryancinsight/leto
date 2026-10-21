//! Phase-shift correlation and its inverse.

use eunomia::RealField;

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

#[cfg(test)]
mod tests {
    use super::{phase_error_degrees_for_correlation, phase_shift_correlation_curve};

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
}
