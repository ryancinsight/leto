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
/// LU decomposition with partial pivoting, solve, determinant, inverse.
pub mod lu;
/// Vector and matrix norms.
pub mod norms;
/// Householder QR factorization and least-squares solve.
pub mod qr;

pub use cholesky::{
    cholesky_decompose, cholesky_det, cholesky_inv, cholesky_solve, CholeskyDecomposition,
};
pub use eigen::SymmetricEigenDecomposition;
pub use eigen::{symmetric_eigen_jacobi, symmetric_eigen_jacobi_with_tolerance};
pub use lu::{det, inv, lu_decompose, solve, LuDecomposition};
pub use norms::{norm, norm_l1, norm_l2, norm_max, NormKind, NormL1, NormL2, NormMax};
pub use qr::{qr_decompose, solve_least_squares, QrDecomposition};
