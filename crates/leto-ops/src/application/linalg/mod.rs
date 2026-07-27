//! Dense linear algebra (Stage A1 of the Atlas leto replacement).
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
/// Complex linear algebra: solve and inverse for `Array2<Complex<f64>>`.
pub mod complex_linalg;
/// Real symmetric eigensolver (Jacobi rotations).
pub mod eigen;
/// General (non-symmetric) eigenvalues via shifted complex QR.
pub mod eigenvalues;
/// LU with complete (full) pivoting: rank-revealing `P A Q = L U`.
pub mod full_piv_lu;
/// Hermitian eigensolver (Jacobi and implicit QR with Wilkinson shift).
pub mod hermitian;
/// Upper Hessenberg reduction via Householder reflectors.
pub mod hessenberg;
/// Shared Householder reflector primitive (SSOT for orthogonal transforms).
pub(crate) mod householder;
/// Iterative solvers (CG, BiCGSTAB, GMRES, LSQR) and preconditioners.
pub mod iterative;
/// LU decomposition with partial pivoting, solve, determinant, inverse.
pub mod lu;
/// Batched LU decomposition over a stack of square matrices.
pub mod lu_batch;
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
/// Compact-WY block Householder reflectors (BLAS-3 reflector aggregation).
pub(crate) mod reflector_block;
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
pub use lu_batch::lu_batch;
pub use matrix::{
    AsMatrixView, MatrixDecompose, MatrixFunction, MatrixNorm, MatrixProduct, MatrixProperties,
    MatrixSolve,
};
pub use matrix_function::{matexp, matpow};
pub use norms::{
    l2_normalize, l2_normalize_into, norm, norm_l1, norm_l2, norm_max, NormKind, NormL1, NormL2,
    NormMax,
};
pub use products::kron;
pub use properties::{matrix_rank, matrix_rank_with_tolerance, trace};
pub use qr::{qr_decompose, solve_least_squares, QrDecomposition};
pub use schur::{schur, RealSchur};
pub use svd::{
    pinv, singular_values, svd_decompose, svd_decompose_with_tolerance, svd_rank_revealing,
    svd_rank_revealing_with_tolerance, svd_via_bidiagonal, SvdDecomposition,
};
pub use udu::{udu_decompose, UduDecomposition};

/// Iterative solvers (SSOT re-export).
pub use iterative::{
    BiCGSTAB, Configurable, ConjugateGradient, ConvergenceMonitor, ILUPreconditioner,
    IdentityPreconditioner, IterativeLinearSolver, IterativeSolverConfig, JacobiPreconditioner,
    LinearOperator, LinearSolver, LsqrConfig, LsqrResult, LsqrSolver, LsqrStopReason,
    Preconditioner, SORPreconditioner, SSORPreconditioner, GMRES,
};

/// Complex linear algebra (re-export).
pub use complex_linalg::{complex_inv, complex_solve};

/// Hermitian eigensolvers (re-export).
pub use hermitian::{
    hermitian_eigen_jacobi, hermitian_eigen_qr, HermitianEigenConfig, HermitianEigenResult,
};
