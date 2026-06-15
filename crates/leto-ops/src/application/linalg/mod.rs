//! Dense linear algebra (Stage A1 of the Atlas nalgebra replacement).
//!
//! Routines are admitted with a named consumer driver, value-semantic
//! analytical fixtures, and differential contracts where an internal reference
//! path exists. Kernels are generic over `RealScalar` and run in native
//! precision per the `Scalar` contract.

/// Cholesky factorization of symmetric positive-definite matrices.
pub mod cholesky;
/// Real symmetric eigensolver (Jacobi rotations).
pub mod eigen;
/// Upper Hessenberg reduction via Householder reflectors.
pub mod hessenberg;
/// LU decomposition with partial pivoting, solve, determinant, inverse.
pub mod lu;
/// Fluent rank-2 linear-algebra trait surface over `Array2`/`ArrayView2`.
pub mod matrix;
/// Vector and matrix norms.
pub mod norms;
/// Structural matrix products (Kronecker product).
pub mod products;
/// Scalar matrix properties (trace, numerical rank).
pub mod properties;
/// Householder QR factorization and least-squares solve.
pub mod qr;
/// Thin SVD and singular values for finite matrices.
pub mod svd;

pub use cholesky::{
    cholesky_decompose, cholesky_det, cholesky_inv, cholesky_solve, CholeskyDecomposition,
};
pub use eigen::SymmetricEigenDecomposition;
pub use eigen::{
    symmetric_eigen_jacobi, symmetric_eigen_jacobi_with_tolerance, symmetric_eigenvalues_jacobi,
    symmetric_eigenvalues_jacobi_with_tolerance,
};
pub use hessenberg::{hessenberg, HessenbergDecomposition};
pub use lu::{det, inv, lu_decompose, solve, LuDecomposition};
pub use matrix::{
    AsMatrixView, MatrixDecompose, MatrixNorm, MatrixProduct, MatrixProperties, MatrixSolve,
};
pub use norms::{norm, norm_l1, norm_l2, norm_max, NormKind, NormL1, NormL2, NormMax};
pub use products::kron;
pub use properties::{matrix_rank, matrix_rank_with_tolerance, trace};
pub use qr::{qr_decompose, solve_least_squares, QrDecomposition};
pub use svd::{
    pinv, singular_values, svd_decompose, svd_decompose_with_tolerance, svd_rank_revealing,
    svd_rank_revealing_with_tolerance, SvdDecomposition,
};
