/// Elementwise, reduction, and matrix operations.
pub mod map;
/// Axis-aware keep-dim reduction operations.
pub mod reduction;

pub use map::{add, binary_map, div, matmul, mul, sub, sum, AddOp, BinaryOp, DivOp, MulOp, SubOp};
pub use reduction::{
    max_axis_into, mean_axis_into, min_axis_into, reduce_axis_into, sum_axis_into, AxisReduction,
    MaxAxis, MeanAxis, MinAxis, SumAxis,
};
