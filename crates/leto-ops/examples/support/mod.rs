//! Shared numerical evidence for the runnable parity examples.

/// One measured differential and its admissible numerical bound.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Observation {
    pub(crate) name: &'static str,
    pub(crate) error: f64,
    pub(crate) bound: f64,
}

impl Observation {
    pub(crate) const fn new(name: &'static str, error: f64, bound: f64) -> Self {
        Self { name, error, bound }
    }

    pub(crate) fn assert_within_bound(self) {
        assert!(
            self.error <= self.bound,
            "{} error {:.6e} exceeds derived bound {:.6e}",
            self.name,
            self.error,
            self.bound
        );
    }
}

/// Maximum elementwise absolute difference.
///
/// Length equality is part of the value contract. Checking it before `zip`
/// prevents a missing result tail from being reported as parity.
pub(crate) fn max_abs_diff(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(
        actual.len(),
        expected.len(),
        "parity operands have different lengths"
    );
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max)
}

/// Higham's standard floating-point accumulation factor
/// `γₙ = nε / (1 - nε)` for round-to-nearest arithmetic.
pub(crate) fn gamma(terms: usize) -> f64 {
    let scaled_epsilon = terms as f64 * f64::EPSILON;
    assert!(
        scaled_epsilon < 1.0,
        "roundoff bound requires terms * epsilon < 1"
    );
    scaled_epsilon / (1.0 - scaled_epsilon)
}
