//! Pearson product-moment correlation.

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
    let n = T::from_count(a.len());
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

#[cfg(test)]
mod tests {
    use super::pearson;
    use crate::application::statistics::accumulation_tolerance;

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

    /// §19.2 phase-sensitivity theorem: for A = sin(kx), B = sin(kx + φ) the
    /// Pearson correlation equals cos(φ).
    #[test]
    fn pearson_equals_cosine_of_phase_shift() {
        use std::f64::consts::PI;
        let n = 2000;
        let k = 2.0 * PI / <f64 as eunomia::FloatElement>::from_count(n); // one period over the window
        for &phi in &[0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI] {
            let a: Vec<f64> = (0..n)
                .map(|i| (k * <f64 as eunomia::FloatElement>::from_count(i)).sin())
                .collect();
            let b: Vec<f64> = (0..n)
                .map(|i| (k * <f64 as eunomia::FloatElement>::from_count(i) + phi).sin())
                .collect();
            assert!(
                (pearson(&a, &b) - phi.cos()).abs() < 1e-3,
                "phase {phi}: r={} vs cos={}",
                pearson(&a, &b),
                phi.cos()
            );
        }
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

    fn pearson_contract<T: eunomia::RealField>() {
        // Exactly proportional series: the correlation is 1 by definition.
        let a: Vec<T> = (1..=5).map(|i| T::from_f64(f64::from(i))).collect();
        let b: Vec<T> = (1..=5).map(|i| T::from_f64(f64::from(2 * i))).collect();
        let deviation = (pearson(&a, &b) - T::ONE).abs();
        assert!(
            deviation < accumulation_tolerance::<T>(a.len()),
            "pearson deviated by {deviation:?} at this width"
        );
    }

    #[test]
    fn pearson_is_generic_over_scalar() {
        pearson_contract::<f32>();
        pearson_contract::<f64>();
    }
}
