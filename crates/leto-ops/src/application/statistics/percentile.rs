//! Distribution percentile ranges.

use eunomia::{convert::IntegerTarget, FloatElement, RealField};

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
    // Nearest-rank indices: round in `f64` (half away from zero, as before),
    // then convert the already-integral value exactly.
    let last_f = f64::from_count(last);
    let p05 = values[usize::from_truncated((0.05 * last_f).round())];
    let p95 = values[usize::from_truncated((0.95 * last_f).round())];
    p95 - p05
}

#[cfg(test)]
mod tests {
    use super::percentile_range;

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
}
