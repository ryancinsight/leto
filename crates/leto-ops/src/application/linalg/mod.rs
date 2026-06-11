//! Dense linear algebra (Stage A1 of the Atlas nalgebra replacement).
//!
//! Routines are admitted with a named consumer driver and a differential
//! oracle (nalgebra as a dev-dependency), generic over `RealScalar`, and run
//! in native precision per the `Scalar` contract.

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

pub use cholesky::{cholesky_decompose, CholeskyDecomposition};
pub use eigen::SymmetricEigenDecomposition;
pub use eigen::{symmetric_eigen_jacobi, symmetric_eigen_jacobi_with_tolerance};
pub use lu::{det, inv, lu_decompose, solve, LuDecomposition};
pub use norms::{norm, norm_l1, norm_l2, norm_max, NormKind, NormL1, NormL2, NormMax};
pub use qr::{qr_decompose, solve_least_squares, QrDecomposition};
