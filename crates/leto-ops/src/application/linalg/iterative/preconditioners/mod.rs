//! Preconditioner implementations for iterative linear solvers.
//!
//! - [`IdentityPreconditioner`] — no-op baseline.
//! - [`JacobiPreconditioner`] — diagonal scaling.
//! - [`SORPreconditioner`] — forward SOR sweep preconditioner.
//! - [`ILUPreconditioner`] — ILU(0) factorisation with zero fill-in.
//! - [`SSORPreconditioner`] — symmetric SOR preconditioner for SPD-like systems.

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
