//! Leto operation scalar contract: the trait, its implementations, and the
//! scalar fallback kernels behind the SIMD hooks.

mod contract;
mod fallback;
mod impls;

pub use contract::Scalar;
