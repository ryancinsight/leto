//! Successive Over-Relaxation (SOR) preconditioner.

use super::super::traits::Preconditioner;
use crate::CsrMatrix;
use crate::Scalar as LetoScalar;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, LetoError, Result};

#[inline]
fn from_f64<T: FloatElement>(value: f64) -> T {
    <T as FloatElement>::from_f64(value)
}

#[inline]
fn vector_len<T>(vector: &Array1<T>) -> usize {
    vector.shape()[0]
}

/// Forward-sweep SOR preconditioner.
///
/// Applies a single sweep solving `(D/ω - L) z = r`.
pub struct SORPreconditioner<T: RealField + Copy> {
    matrix: CsrMatrix<T>,
    omega: T,
}

impl<T: RealField + FloatElement + Copy + LetoScalar> SORPreconditioner<T> {
    /// Create SOR preconditioner with a caller-supplied relaxation parameter.
    ///
    /// # Errors
    /// Returns [`LetoError::InvalidInput`] when the matrix is non-square or
    /// `omega` is outside `(0, 2)`.
    pub fn new(matrix: CsrMatrix<T>, omega: T) -> Result<Self> {
        if matrix.nrows() != matrix.ncols() {
            return Err(LetoError::InvalidInput(format!(
                "SOR preconditioner requires square matrix, got {}x{}",
                matrix.nrows(),
                matrix.ncols()
            )));
        }

        let zero = <T as NumericElement>::ZERO;
        let two = from_f64(2.0);
        if omega <= zero || omega >= two {
            return Err(LetoError::InvalidInput(
                "SOR omega parameter must be in range (0, 2)".into(),
            ));
        }

        Ok(Self { matrix, omega })
    }

    /// Create SOR preconditioner with omega tuned for 1D Poisson systems.
    ///
    /// # Errors
    /// Returns [`LetoError::InvalidInput`] when the matrix does not match the
    /// expected tridiagonal nearest-neighbour structure.
    pub fn with_omega_for_1d_poisson(matrix: CsrMatrix<T>) -> Result<Self> {
        Self::validate_1d_poisson_structure(&matrix)?;

        let n = matrix.nrows() as f64;
        let omega_opt = 2.0 / (1.0 + (std::f64::consts::PI / n).sin());
        Self::new(matrix, from_f64(omega_opt))
    }

    /// Return the relaxation parameter.
    pub fn omega(&self) -> T {
        self.omega
    }

    /// Number of rows in the preconditioned system.
    pub fn nrows(&self) -> usize {
        self.matrix.nrows()
    }

    fn validate_1d_poisson_structure(matrix: &CsrMatrix<T>) -> Result<()> {
        for i in 0..matrix.nrows() {
            let row = matrix.row(i);
            if row.nnz() > 3 {
                return Err(LetoError::InvalidInput(format!(
                    "Row {} has {} non-zeros; 1D Poisson should have at most 3",
                    i,
                    row.nnz()
                )));
            }

            for &j in row.col_indices() {
                if (j as i32 - i as i32).abs() > 1 {
                    return Err(LetoError::InvalidInput(format!(
                        "Non-zero at ({i}, {j}) violates tridiagonal structure"
                    )));
                }
            }
        }

        Ok(())
    }
}

impl<T: RealField + Copy + NumericElement + LetoScalar> Preconditioner<T> for SORPreconditioner<T> {
    fn apply_to(&self, r: &Array1<T>, z: &mut Array1<T>) -> Result<()> {
        let n = self.matrix.nrows();
        let r_len = vector_len(r);
        if r_len != n {
            return Err(LetoError::InvalidInput(format!(
                "SOR residual length mismatch: expected {n}, got {r_len}"
            )));
        }

        let z_len = vector_len(z);
        if z_len != n {
            return Err(LetoError::InvalidInput(format!(
                "SOR output length mismatch: expected {n}, got {z_len}"
            )));
        }

        for idx in 0..n {
            z[idx] = <T as NumericElement>::ZERO;
        }

        for i in 0..n {
            let mut sum = <T as NumericElement>::ZERO;
            let mut diag = <T as NumericElement>::ONE;
            let row = self.matrix.row(i);

            for (&j, &val) in row.col_indices().iter().zip(row.values()) {
                if j < i {
                    sum += val * z[j];
                } else if j == i {
                    diag = val;
                }
            }

            z[i] = (r[i] + sum) * self.omega / diag;
        }

        Ok(())
    }
}
