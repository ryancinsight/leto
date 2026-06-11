/// Shared logical-index conversion helpers.
pub(crate) mod index;
/// Dense linear algebra: eigensolver and norms.
pub mod linalg;
/// Elementwise binary and aggregate map operations.
pub mod map;
/// Matrix multiplication operations.
pub mod matrix;
/// Deterministic seeded random array constructors.
pub mod random;
/// Axis-aware keep-dim reduction operations.
pub mod reduction;
/// Prefix/suffix scan operations.
pub mod scan;
/// Unary map operations.
pub mod unary;
/// Rank-1 vector operations.
pub mod vector;
/// Mutable zip-map operations.
pub mod zip;

pub use linalg::{
    det, inv, lu_decompose, norm, norm_l1, norm_l2, norm_max, solve, symmetric_eigen_jacobi,
    LuDecomposition, NormKind, NormL1, NormL2, NormMax, SymmetricEigenDecomposition,
};
pub use map::{
    add, binary_map, div, mul, scalar_map, scalar_map_into, sub, sum, AddOp, BinaryOp, DivOp,
    MulOp, SubOp,
};
pub use matrix::{batched_matmul, matmul};
pub use random::{normal_with_seed, uniform_with_seed};
pub use reduction::{
    max_axis, max_axis_into, mean_axis, mean_axis_into, min_axis, min_axis_into, reduce_axis,
    reduce_axis_into, sum_axis, sum_axis_into, AxisReduction, MaxAxis, MeanAxis, MinAxis, SumAxis,
};
pub use scan::{
    cumsum, cumsum_into, scan_axis, scan_axis_into, CumProdOp, CumSumOp, ScanDirection, ScanOp,
};
pub use unary::{
    map, map_inplace, map_into, mapv, unary_map, unary_map_into, AbsOp, CosOp, ExpOp, LnOp, NegOp,
    PowfOp, RecipOp, SinOp, SqrtOp, UnaryOp,
};
pub use vector::dot;
pub use zip::{zip2_mut_with, zip_mut_with};
