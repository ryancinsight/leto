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

pub use application::map::{
    add, binary_map, div, matmul, mul, sub, sum, AddOp, BinaryOp, DivOp, MulOp, SubOp,
};
pub use application::reduction::{
    max_axis_into, mean_axis_into, min_axis_into, reduce_axis_into, sum_axis_into, AxisReduction,
    MaxAxis, MeanAxis, MinAxis, SumAxis,
};
pub use application::unary::{map, map_into, mapv};
pub use application::zip::zip_mut_with;
