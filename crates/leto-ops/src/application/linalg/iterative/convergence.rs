//! Convergence monitoring for iterative linear solvers.

use eunomia::{FloatElement, NumericElement, RealField};

#[inline]
fn from_f64<T: FloatElement>(v: f64) -> T {
    <T as FloatElement>::from_f64(v)
}

/// Records per-iteration residual history and provides convergence analysis helpers.
///
/// The monitor is returned from every successful solver call so that callers can inspect
/// the convergence trajectory, estimate the condition number, and validate against
/// theoretical bounds without incurring overhead when monitoring is not needed.
#[derive(Debug, Clone)]
pub struct ConvergenceMonitor<T: RealField + Copy> {
    /// Residual norm at iteration 0 (before the first update).
    pub initial_residual: T,
    /// Current iteration count.
    pub iteration: usize,
    /// Residual norm at each iteration (index 0 = initial residual).
    pub residual_history: Vec<T>,
    /// Optional theoretical convergence-rate bound (e.g. CG's Chebyshev bound).
    pub theoretical_bound: Option<T>,
    /// Optional condition-number estimate derived from the Lanczos recurrence.
    pub condition_number_estimate: Option<f64>,
}

impl<T: RealField + Copy> ConvergenceMonitor<T> {
    /// Create a fresh monitor, recording `initial_residual` as iteration 0.
    pub fn new(initial_residual: T) -> Self {
        Self {
            initial_residual,
            iteration: 0,
            residual_history: vec![initial_residual],
            theoretical_bound: None,
            condition_number_estimate: None,
        }
    }

    /// Push a residual measurement for the completed iteration.
    pub fn record_residual(&mut self, residual: T) {
        self.iteration += 1;
        self.residual_history.push(residual);
    }

    /// Attach a theoretical per-iteration reduction bound.
    pub fn set_theoretical_bound(&mut self, bound: T) {
        self.theoretical_bound = Some(bound);
    }

    /// Attach a condition-number estimate.
    pub fn set_condition_number_estimate(&mut self, kappa: f64) {
        self.condition_number_estimate = Some(kappa);
    }

    /// Geometric mean of the per-iteration residual reduction factor, or `None` if fewer
    /// than two data points are recorded.
    pub fn convergence_factor(&self) -> Option<T>
    where
        T: FloatElement,
    {
        if self.residual_history.len() < 2 {
            return None;
        }
        let r_final = *self.residual_history.last()?;
        let ratio = r_final / self.initial_residual;
        Some(FloatElement::powf(
            ratio,
            from_f64(1.0 / self.iteration as f64),
        ))
    }

    /// CG per-iteration upper bound on the A-norm error reduction:
    ///
    /// ```text
    /// factor = (√κ − 1) / (√κ + 1)
    /// ```
    ///
    /// where `κ` is the spectral condition number.  Returns `ONE` for degenerate
    /// or non-finite `kappa` values.
    pub fn cg_theoretical_bound(&self, kappa: f64) -> T
    where
        T: FloatElement,
    {
        if !kappa.is_finite() || kappa < 1.0 {
            return <T as NumericElement>::ONE;
        }
        let sqrt_kappa = kappa.sqrt();
        from_f64((sqrt_kappa - 1.0) / (sqrt_kappa + 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_mean_reduction_factor() {
        let mut m = ConvergenceMonitor::new(16.0_f64);
        m.record_residual(4.0);
        m.record_residual(1.0);
        assert_eq!(m.convergence_factor(), Some(0.25));
    }

    #[test]
    fn cg_bound_matches_formula() {
        let m = ConvergenceMonitor::new(1.0_f64);
        assert_eq!(m.cg_theoretical_bound(9.0), 0.5);
        assert_eq!(m.cg_theoretical_bound(0.5), 1.0);
        assert_eq!(m.cg_theoretical_bound(f64::INFINITY), 1.0);
    }
}
