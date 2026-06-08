#![warn(missing_docs)]
//! Leto Operations contains performance-tuned math and reduction kernels for Leto arrays.

/// Application-level operation entry points.
pub mod application;
/// Operation scalar and strategy contracts.
pub mod domain;
/// SIMD and parallel execution infrastructure.
pub mod infrastructure;

pub use domain::scalar::Scalar;
pub use domain::strategy::{ExecutionStrategy, ScalarStrategy};

#[cfg(feature = "simd")]
pub use domain::strategy::SimdStrategy;

#[cfg(feature = "parallel")]
pub use domain::strategy::ParallelStrategy;

pub use application::map::{add, div, matmul, mul, sub, sum};
