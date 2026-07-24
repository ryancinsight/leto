//! Matrix functions: integer power and exponential (leto `pow` / `exp`
//! parity).
//!
//! Both reduce to repeated matrix products (and, for the exponential, one
//! inverse of a well-conditioned denominator), reusing the caller-owned
//! [`matmul`](crate::matmul) and partial-pivot LU inverse rather than
//! introducing new contraction or solve paths (SSOT). Shared dense helpers live
//! in `dense`.
//!
//! # Submodules
//! - power — [`matpow`]: `Aᵏ` by exponentiation-by-squaring (`Θ(log k)`
//!   matmuls), exact, generic over [`Scalar`](crate::domain::scalar::Scalar).
//! - exponential — [`matexp`]: `e^A` by scaling-and-squaring + diagonal Padé(6).

mod dense;
mod exponential;
mod power;

pub use exponential::matexp;
pub use power::matpow;
