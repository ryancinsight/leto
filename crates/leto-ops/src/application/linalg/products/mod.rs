//! Structural matrix products that build a larger matrix from two operands.
//!
//! Distinct from the contracting matrix product (`matmul`): these compose
//! shapes rather than contract a shared dimension. Generic over
//! [`Scalar`](crate::Scalar).

/// Kronecker (tensor) product.
pub mod kronecker;

pub use kronecker::kron;
