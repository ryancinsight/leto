/// Real symmetric eigensolver operations.
pub mod eigen;
/// Shared logical-index conversion helpers.
pub(crate) mod index;
/// Elementwise binary and aggregate map operations.
pub mod map;
/// Matrix multiplication operations.
pub mod matrix;
/// Axis-aware keep-dim reduction operations.
pub mod reduction;
/// Unary map operations.
pub mod unary;
/// Rank-1 vector operations.
pub mod vector;
/// Mutable zip-map operations.
pub mod zip;

pub use eigen::{symmetric_eigen_jacobi, SymmetricEigenDecomposition};
pub use map::{
    add, binary_map, div, mul, scalar_map, scalar_map_into, sub, sum, AddOp, BinaryOp, DivOp,
    MulOp, SubOp,
};
pub use matrix::matmul;
pub use reduction::{
    max_axis, max_axis_into, mean_axis, mean_axis_into, min_axis, min_axis_into, reduce_axis,
    reduce_axis_into, sum_axis, sum_axis_into, AxisReduction, MaxAxis, MeanAxis, MinAxis, SumAxis,
};
pub use unary::{
    map, map_inplace, map_into, mapv, unary_map, unary_map_into, AbsOp, CosOp, ExpOp, LnOp, NegOp,
    PowfOp, RecipOp, SinOp, SqrtOp, UnaryOp,
};
pub use vector::dot;
pub use zip::zip_mut_with;
