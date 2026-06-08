/// Marker trait for operation execution strategies.
pub trait ExecutionStrategy: Send + Sync + 'static {}

/// Zero-sized type marker routing operations through scalar execution loops.
pub struct ScalarStrategy;
impl ExecutionStrategy for ScalarStrategy {}

/// Zero-sized type marker routing operations through SIMD (hermes-simd) execution paths.
#[cfg(feature = "simd")]
pub struct SimdStrategy;
#[cfg(feature = "simd")]
impl ExecutionStrategy for SimdStrategy {}

/// Zero-sized type marker routing operations through multi-threaded parallel execution schedules via moirai.
#[cfg(feature = "parallel")]
pub struct ParallelStrategy;
#[cfg(feature = "parallel")]
impl ExecutionStrategy for ParallelStrategy {}
