//! Preconditioner implementations for iterative linear solvers.
//!
//! - [`IdentityPreconditioner`] — no-op baseline.
//! - [`JacobiPreconditioner`] — diagonal scaling.
//! - [`ILUPreconditioner`] — ILU(0) factorisation with zero fill-in.
//! - [`SORPreconditioner`] — forward SOR sweep.
//! - [`SSORPreconditioner`] — symmetric SOR sweep for SPD-like systems.

pub mod identity;
pub mod ilu;
pub mod jacobi;
pub mod sor;
pub mod ssor;

pub use identity::IdentityPreconditioner;
pub use ilu::ILUPreconditioner;
pub use jacobi::JacobiPreconditioner;
pub use sor::SORPreconditioner;
pub use ssor::SSORPreconditioner;
