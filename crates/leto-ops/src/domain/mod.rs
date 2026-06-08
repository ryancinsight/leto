pub mod strategy;
pub mod scalar;

pub use strategy::ExecutionStrategy;
pub use strategy::ScalarStrategy;
pub use scalar::Scalar;

#[cfg(feature = "simd")]
pub use strategy::SimdStrategy;

#[cfg(feature = "parallel")]
pub use strategy::ParallelStrategy;
