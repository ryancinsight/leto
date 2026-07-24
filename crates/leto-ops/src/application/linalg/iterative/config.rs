//! Configuration for iterative linear solvers.

use eunomia::{FloatElement, RealField};
use serde::{Deserialize, Serialize};

#[inline]
fn from_f64<T: FloatElement>(v: f64) -> T {
    <T as FloatElement>::from_f64(v)
}

/// Configuration shared by all iterative solvers (CG, BiCGSTAB, GMRES, LSQR).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IterativeSolverConfig<T: RealField + Copy> {
    /// Maximum number of iterations before declaring non-convergence.
    pub max_iterations: usize,
    /// Absolute residual norm tolerance for convergence.
    pub tolerance: T,
    /// Whether the caller has attached a preconditioner.
    pub use_preconditioner: bool,
}

impl<T: RealField + Copy> IterativeSolverConfig<T> {
    /// Create a configuration with the given convergence tolerance.
    pub fn new(tolerance: T) -> Self {
        Self {
            max_iterations: 1000,
            tolerance,
            use_preconditioner: false,
        }
    }

    /// Override the maximum iteration count.
    #[must_use]
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Enable preconditioning flag (informational; solver still requires caller to pass `P`).
    #[must_use]
    pub fn with_preconditioner(mut self) -> Self {
        self.use_preconditioner = true;
        self
    }
}

impl<T: RealField + Copy + FloatElement> Default for IterativeSolverConfig<T> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: from_f64(1e-6),
            use_preconditioner: false,
        }
    }
}
