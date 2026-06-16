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
/// Sparse matrices (CSR) and sparsity-exploiting kernels.
pub mod sparse;
/// Unary map operations.
pub mod unary;
/// Rank-1 vector operations.
pub mod vector;
/// Mutable zip-map operations.
pub mod zip;

pub use linalg::{
    cholesky_decompose, cholesky_det, cholesky_inv, cholesky_solve, det, inv, lu_decompose, norm,
    norm_l1, norm_l2, norm_max, qr_decompose, solve, solve_least_squares, symmetric_eigen_jacobi,
    symmetric_eigen_jacobi_with_tolerance, symmetric_eigenvalues_jacobi,
    symmetric_eigenvalues_jacobi_with_tolerance, CholeskyDecomposition, LuDecomposition, NormKind,
    NormL1, NormL2, NormMax, QrDecomposition, SymmetricEigenDecomposition,
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
pub use sparse::{spmv, spmv_into, CsrMatrix};
pub use unary::{
    map, map_inplace, map_into, mapv, unary_map, unary_map_into, AbsOp, CosOp, ExpOp, LnOp, NegOp,
    PowfOp, RecipOp, SinOp, SqrtOp, UnaryOp,
};
pub use vector::dot;
pub use zip::{indexed_zip2_mut_with, indexed_zip_mut_with, zip2_mut_with, zip_mut_with};
