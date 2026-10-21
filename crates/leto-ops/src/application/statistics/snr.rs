//! Peak signal-to-noise ratio.

use eunomia::RealField;

use super::error::rmse;

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

#[cfg(test)]
mod tests {
    use super::{psnr, validation_psnr_from_relative_rmse};

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

    // ── scalar-type genericity ──────────────────────────────────────────────
    //
    // Each contract is written once over `T: RealField` and instantiated
    // across every scalar the stack ships, so a newly admitted type
    // inherits the suite instead of needing the assertions copied again.

    #[test]
    fn psnr_is_generic_over_scalar() {
        // Identical signals carry no error, so PSNR is unbounded (infinite).
        fn contract<T: eunomia::RealField>() {
            let reference: Vec<T> = vec![T::from_f64(1.0); 4];
            assert!(!psnr(&reference, &reference).is_finite());
        }
        contract::<f32>();
        contract::<f64>();
    }
}
