//! Iterative linear solvers for symmetric, non-symmetric, and least-squares
//! systems, together with preconditioner implementations.
//!
//! ## Solvers
//!
//! | Type | Struct | System type |
//! |------|--------|-------------|
//! | Conjugate Gradient | [`ConjugateGradient`] | SPD |
//! | BiCGSTAB | [`BiCGSTAB`] | Non-symmetric |
//! | GMRES(m) | [`GMRES`] | Non-symmetric |
//! | LSQR | [`LsqrSolver`] | Least-squares (rectangular A) |
//!
//! ## Preconditioners
//!
//! | Type | Struct |
//! |------|--------|
//! | Identity (no-op) | [`IdentityPreconditioner`] |
//! | Jacobi (diagonal) | [`JacobiPreconditioner`] |
//! | ILU(0) | [`ILUPreconditioner`] |
//!
//! ## Core traits
//!
//! - [`LinearOperator`] — apply A or Aᵀ to a vector.
//! - [`Preconditioner`] — apply M⁻¹ to a vector.
//! - [`IterativeLinearSolver`] — in-place solve returning a [`ConvergenceMonitor`].
//! - [`LinearSolver`] — object-safe solve returning the solution vector.
//! - [`Configurable`] — access solver configuration.
//!
//! ## Error handling
//!
//! All solver methods return `leto::Result<_>`, using:
//! - [`LetoError::ConvergenceError`] — maximum iterations exceeded.
//! - [`LetoError::NumericalBreakdown`] — stagnation / near-zero denominator.
//! - [`LetoError::NotPositiveDefinite`] — CG SPD violation.
//! - [`LetoError::InvalidInput`] — dimension mismatch / bad parameters.

pub mod bicgstab;
pub mod cg;
pub mod config;
pub mod convergence;
pub mod gmres;
pub mod lsqr;
pub(super) mod ops;
pub mod preconditioners;
pub mod traits;

pub use bicgstab::BiCGSTAB;
pub use cg::ConjugateGradient;
pub use config::IterativeSolverConfig;
pub use convergence::ConvergenceMonitor;
pub use gmres::GMRES;
pub use lsqr::{LsqrConfig, LsqrResult, LsqrSolver, LsqrStopReason};
pub use preconditioners::{ILUPreconditioner, IdentityPreconditioner, JacobiPreconditioner};
pub use traits::{
    Configurable, IterativeLinearSolver, LinearOperator, LinearSolver, Preconditioner,
};
