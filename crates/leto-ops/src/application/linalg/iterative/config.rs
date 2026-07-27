//! Configuration for iterative linear solvers.

use eunomia::{FloatElement, NumericElement, RealField};
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
    /// Relative residual tolerance, measured against `‖b‖₂`.
    ///
    /// Zero (the default) reduces [`Self::threshold`] to the purely absolute
    /// test.  A non-zero value makes convergence invariant under a uniform
    /// rescaling of `A·x = b`, which the absolute test alone is not: scaling
    /// the system by `α` scales every residual by `α`, so a fixed absolute
    /// tolerance becomes unreachable for `α ≫ 1` and is satisfied by `x = 0`
    /// for `α ≪ 1`.
    pub relative_tolerance: T,
    /// Whether the caller has attached a preconditioner.
    pub use_preconditioner: bool,
}

impl<T: RealField + Copy> IterativeSolverConfig<T> {
    /// Create a configuration with the given absolute convergence tolerance.
    pub fn new(tolerance: T) -> Self {
        Self {
            max_iterations: 1000,
            tolerance,
            relative_tolerance: <T as NumericElement>::ZERO,
            use_preconditioner: false,
        }
    }

    /// Convergence threshold for a right-hand side of norm `rhs_norm`:
    ///
    /// ```text
    /// threshold = max(tolerance, relative_tolerance · ‖b‖₂)
    /// ```
    ///
    /// A solver has converged when `‖b − A·x‖₂ ≤ threshold`.
    #[must_use]
    pub fn threshold(&self, rhs_norm: T) -> T {
        let relative = self.relative_tolerance * rhs_norm;
        if relative > self.tolerance {
            relative
        } else {
            self.tolerance
        }
    }

    /// Override the maximum iteration count.
    #[must_use]
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set the relative residual tolerance (see [`Self::relative_tolerance`]).
    #[must_use]
    pub fn with_relative_tolerance(mut self, relative_tolerance: T) -> Self {
        self.relative_tolerance = relative_tolerance;
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
            relative_tolerance: <T as NumericElement>::ZERO,
            use_preconditioner: false,
        }
    }
}
