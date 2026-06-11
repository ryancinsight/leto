#![warn(missing_docs)]
//! Leto Operations contains performance-tuned math and reduction kernels for Leto arrays.

/// Application-level operation entry points.
pub mod application;
/// Operation scalar and strategy contracts.
pub mod domain;
/// SIMD and parallel execution infrastructure.
pub mod infrastructure;

pub use domain::real::RealScalar;
pub use domain::rng::Xorshift64;
pub use domain::scalar::Scalar;
pub use domain::strategy::{ExecutionStrategy, ScalarStrategy};

#[cfg(feature = "simd")]
pub use domain::strategy::SimdStrategy;

#[cfg(feature = "parallel")]
pub use domain::strategy::ParallelStrategy;

pub use application::linalg::{
    cholesky_decompose, qr_decompose, solve_least_squares, CholeskyDecomposition, QrDecomposition,
    det, inv, lu_decompose, norm, norm_l1, norm_l2, norm_max, solve, symmetric_eigen_jacobi,
    symmetric_eigen_jacobi_with_tolerance, LuDecomposition, NormKind, NormL1, NormL2, NormMax,
    SymmetricEigenDecomposition,
};
pub use application::map::{
    add, binary_map, div, mul, scalar_map, scalar_map_into, sub, sum, AddOp, BinaryOp, DivOp,
    MulOp, SubOp,
};
pub use application::matrix::{batched_matmul, matmul};
pub use application::random::{normal_with_seed, uniform_with_seed};
pub use application::reduction::{
    max_axis, max_axis_into, mean_axis, mean_axis_into, min_axis, min_axis_into, reduce_axis,
    reduce_axis_into, sum_axis, sum_axis_into, AxisReduction, MaxAxis, MeanAxis, MinAxis, SumAxis,
};
pub use application::scan::{
    cumsum, cumsum_into, scan_axis, scan_axis_into, CumProdOp, CumSumOp, ScanDirection, ScanOp,
};
pub use application::unary::{
    map, map_inplace, map_into, mapv, unary_map, unary_map_into, AbsOp, CosOp, ExpOp, LnOp, NegOp,
    PowfOp, RecipOp, SinOp, SqrtOp, UnaryOp,
};
pub use application::vector::dot;
pub use application::zip::{zip2_mut_with, zip_mut_with};
