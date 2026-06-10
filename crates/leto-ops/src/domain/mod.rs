/// Real transcendental scalar contract.
pub mod real;
/// Numeric scalar contract.
pub mod scalar;
/// Execution strategy marker types.
pub mod strategy;

pub use real::RealScalar;
pub use scalar::Scalar;
pub use strategy::ExecutionStrategy;
pub use strategy::ScalarStrategy;

#[cfg(feature = "simd")]
pub use strategy::SimdStrategy;

#[cfg(feature = "parallel")]
pub use strategy::ParallelStrategy;
