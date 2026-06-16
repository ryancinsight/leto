//! Dense linear algebra (Stage A1 of the Atlas nalgebra replacement).
//!
//! Routines are admitted with a named consumer driver, value-semantic
//! analytical fixtures, and differential contracts where an internal reference
//! path exists. Kernels are generic over `RealScalar` and run in native
//! precision per the `Scalar` contract.

/// Golub–Kahan bidiagonalization via two-sided Householder reflectors.
pub mod bidiagonal;
/// Symmetric-indefinite Bunch–Kaufman `P A Pᵀ = L D Lᵀ` factorization.
pub mod bunch_kaufman;
/// Cholesky factorization of symmetric positive-definite matrices.
pub mod cholesky;
/// QR with column pivoting: rank-revealing `A P = Q R`.
pub mod col_piv_qr;
/// Real symmetric eigensolver (Jacobi rotations).
pub mod eigen;
/// General (non-symmetric) eigenvalues via shifted complex QR.
pub mod eigenvalues;
/// LU with complete (full) pivoting: rank-revealing `P A Q = L U`.
pub mod full_piv_lu;
/// Upper Hessenberg reduction via Householder reflectors.
pub mod hessenberg;
/// Shared Householder reflector primitive (SSOT for orthogonal transforms).
pub(crate) mod householder;
/// LU decomposition with partial pivoting, solve, determinant, inverse.
pub mod lu;
/// Fluent rank-2 linear-algebra trait surface over `Array2`/`ArrayView2`.
pub mod matrix;
/// Matrix functions: integer power and exponential.
pub mod matrix_function;
/// Vector and matrix norms.
pub mod norms;
/// Structural matrix products (Kronecker product).
pub mod products;
/// Scalar matrix properties (trace, numerical rank).
pub mod properties;
/// Householder QR factorization and least-squares solve.
pub mod qr;
/// Real Schur decomposition `A = Q T Qᵀ` via Francis double-shift QR.
pub mod schur;
/// Thin SVD and singular values for finite matrices.
pub mod svd;
/// Symmetric indefinite unpivoted `U D Uᵀ` factorization.
pub mod udu;

pub use bidiagonal::{bidiagonalize, BidiagonalDecomposition};
pub use bunch_kaufman::{bunch_kaufman, BunchKaufmanDecomposition};
pub use cholesky::{
    cholesky_decompose, cholesky_det, cholesky_inv, cholesky_solve, CholeskyDecomposition,
};
pub use col_piv_qr::{col_piv_qr, ColPivQrDecomposition};
pub use eigen::SymmetricEigenDecomposition;
pub use eigen::{
    symmetric_eigen_jacobi, symmetric_eigen_jacobi_with_tolerance, symmetric_eigenvalues_jacobi,
    symmetric_eigenvalues_jacobi_with_tolerance,
};
pub use eigenvalues::eigenvalues;
pub use full_piv_lu::{full_piv_lu, FullPivLuDecomposition};
pub use hessenberg::{hessenberg, HessenbergDecomposition};
pub use lu::{det, inv, lu_decompose, solve, LuDecomposition};
pub use matrix::{
    AsMatrixView, MatrixDecompose, MatrixFunction, MatrixNorm, MatrixProduct, MatrixProperties,
    MatrixSolve,
};
pub use matrix_function::{matexp, matpow};
pub use norms::{norm, norm_l1, norm_l2, norm_max, NormKind, NormL1, NormL2, NormMax};
pub use products::kron;
pub use properties::{matrix_rank, matrix_rank_with_tolerance, trace};
pub use qr::{qr_decompose, solve_least_squares, QrDecomposition};
pub use schur::{schur, RealSchur};
pub use svd::{
    pinv, singular_values, svd_decompose, svd_decompose_with_tolerance, svd_rank_revealing,
    svd_rank_revealing_with_tolerance, svd_via_bidiagonal, SvdDecomposition,
};
pub use udu::{udu_decompose, UduDecomposition};
