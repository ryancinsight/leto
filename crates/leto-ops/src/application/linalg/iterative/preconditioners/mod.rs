//! Preconditioner implementations for iterative linear solvers.
//!
//! - [`IdentityPreconditioner`] — no-op baseline.
//! - [`JacobiPreconditioner`] — diagonal scaling.
//! - [`ILUPreconditioner`] — ILU(0) factorisation with zero fill-in.

pub mod identity;
pub mod ilu;
pub mod jacobi;

pub use identity::IdentityPreconditioner;
pub use ilu::ILUPreconditioner;
pub use jacobi::JacobiPreconditioner;
