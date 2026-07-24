//! Core traits for iterative linear solvers.
//!
//! These traits form the SSOT abstraction boundary for all iterative solver
//! implementations inside `leto-ops` and consumed by downstream crates
//! (`cfd-math`, `kwavers-math`, etc.).

use super::config::IterativeSolverConfig;
use super::convergence::ConvergenceMonitor;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, LetoError, Result};

/// A linear operator A that computes `y ← A·x`.
///
/// Implementations may wrap a sparse/dense matrix or be entirely matrix-free
/// (e.g. a discrete Laplacian stencil).  The trait is `Send + Sync` so it
/// can be shared across threads when `moirai` parallelism is enabled.
pub trait LinearOperator<T: RealField + Copy>: Send + Sync {
    /// Apply the operator: `y ← A·x`.
    ///
    /// # Errors
    /// Returns [`LetoError::InvalidInput`] if dimensions are inconsistent.
    fn apply(&self, x: &Array1<T>, y: &mut Array1<T>) -> Result<()>;

    /// Logical size of the operator (number of rows/columns for square operators).
    /// Returning `0` signals "unchecked" — the solver skips dimension validation.
    fn size(&self) -> usize;

    /// Whether the operator is symmetric (A = Aᵀ).  Defaults to `false`.
    fn is_symmetric(&self) -> bool {
        false
    }

    /// Optional estimate of ‖A‖₂.
    fn norm_estimate(&self) -> Option<T> {
        None
    }

    /// Apply the transpose operator: `y ← Aᵀ·x`.
    ///
    /// Defaults to returning [`LetoError::InvalidInput`] so callers know that
    /// transpose application is unsupported.
    fn apply_transpose(&self, _x: &Array1<T>, _y: &mut Array1<T>) -> Result<()> {
        Err(LetoError::InvalidInput(
            "transpose application is unsupported for this operator".into(),
        ))
    }
}

/// A preconditioner M that approximates A⁻¹ via `z ← M⁻¹·r`.
///
/// Implementations must be `Send + Sync`.
pub trait Preconditioner<T: RealField + Copy>: Send + Sync {
    /// Apply the preconditioner: stores the result in `z`.
    ///
    /// Making `z` an explicit output keeps memory management visible at the
    /// call-site and avoids hidden allocations.
    ///
    /// # Errors
    /// Returns [`LetoError::InvalidInput`] on dimension mismatch, or
    /// [`LetoError::NumericalBreakdown`] if the preconditioner is singular.
    fn apply_to(&self, r: &Array1<T>, z: &mut Array1<T>) -> Result<()>;
}

/// Configurable solver (exposes its configuration for inspection).
pub trait Configurable<T: RealField + Copy> {
    /// The configuration type.
    type Config;
    /// Borrow the solver configuration.
    fn config(&self) -> &Self::Config;
}

/// Object-safe trait for direct or iterative solvers: returns the solution vector.
pub trait LinearSolver<T: RealField + Copy + FloatElement + NumericElement>: Send + Sync {
    /// Solve `A·x = b`, returning a new solution vector.
    ///
    /// # Errors
    /// Returns an error on convergence failure or numerical breakdown.
    fn solve_system(
        &self,
        a: &dyn LinearOperator<T>,
        b: &Array1<T>,
        x0: Option<&Array1<T>>,
    ) -> Result<Array1<T>>;
}

/// In-place iterative solver trait.
///
/// Uses pre-allocated vectors to avoid repeated heap allocation inside
/// tight time-stepping loops.
pub trait IterativeLinearSolver<T: RealField + Copy>:
    Send + Sync + Configurable<T, Config = IterativeSolverConfig<T>>
{
    /// Solve `A·x = b` in place, optionally using preconditioner `P`.
    ///
    /// `x` serves as both initial guess and output solution.
    ///
    /// # Errors
    /// Returns [`LetoError::ConvergenceError`] when `max_iterations` is exhausted,
    /// [`LetoError::NumericalBreakdown`] on stagnation/NaN, or
    /// [`LetoError::NotPositiveDefinite`] if the operator violates SPD assumptions.
    fn solve<Op: LinearOperator<T> + ?Sized, P: Preconditioner<T>>(
        &self,
        a: &Op,
        b: &Array1<T>,
        x: &mut Array1<T>,
        preconditioner: Option<&P>,
    ) -> Result<ConvergenceMonitor<T>>;
}
