/// Complex layout operation contracts.
pub mod layout;
/// Real transcendental scalar contract.
pub mod real;
/// Deterministic seeded pseudo-random generator.
pub mod rng;
/// Numeric scalar contract.
pub mod scalar;
/// Execution strategy marker types.
pub mod strategy;

pub use real::RealScalar;
pub use rng::Xorshift64;
pub use scalar::Scalar;
pub use strategy::ExecutionStrategy;
pub use strategy::ScalarStrategy;

pub use strategy::SimdStrategy;

#[cfg(feature = "parallel")]
pub use strategy::ParallelStrategy;
