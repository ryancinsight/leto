//! Symmetric Successive Over-Relaxation (SSOR) preconditioner.

use super::super::traits::Preconditioner;
use crate::CsrMatrix;
use crate::Scalar as LetoScalar;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, LetoError, Result};

const DEFAULT_OMEGA: f64 = 1.0;

#[inline]
fn from_f64<T: FloatElement>(value: f64) -> T {
    <T as FloatElement>::from_f64(value)
}

#[inline]
fn vector_len<T>(vector: &Array1<T>) -> usize {
    vector.shape()[0]
}

/// Symmetric SOR preconditioner.
///
/// Applies one forward and one backward SOR sweep using a shared relaxation
/// parameter `omega`.
pub struct SSORPreconditioner<T: RealField + Copy> {
    matrix: CsrMatrix<T>,
    omega: T,
}

impl<T: RealField + Copy + FloatElement + LetoScalar> SSORPreconditioner<T> {
    /// Construct with `omega = 1`.
    pub fn new(matrix: CsrMatrix<T>) -> Result<Self> {
        Self::with_omega(matrix, from_f64(DEFAULT_OMEGA))
    }

    /// Construct with a caller-supplied relaxation parameter.
    ///
    /// # Errors
    /// Returns [`LetoError::InvalidInput`] when the matrix is non-square or
    /// `omega` is outside `(0, 2)`.
    pub fn with_omega(matrix: CsrMatrix<T>, omega: T) -> Result<Self> {
        if matrix.nrows() != matrix.ncols() {
            return Err(LetoError::InvalidInput(format!(
                "SSOR preconditioner requires square matrix, got {}x{}",
                matrix.nrows(),
                matrix.ncols()
            )));
        }

        let zero = <T as NumericElement>::ZERO;
        let two = from_f64(2.0);
        if omega <= zero || omega >= two {
            return Err(LetoError::InvalidInput(
                "SSOR omega parameter must be in range (0, 2)".into(),
            ));
        }

        Ok(Self { matrix, omega })
    }

    /// Number of rows in the preconditioned system.
    pub fn nrows(&self) -> usize {
        self.matrix.nrows()
    }

    fn forward_sweep(&self, b: &Array1<T>, x: &mut Array1<T>) {
        let n = self.matrix.nrows();
        for i in 0..n {
            let mut sum = b[i];
            let mut diag = <T as NumericElement>::ONE;
            let row = self.matrix.row(i);
            for (&j, &val) in row.col_indices().iter().zip(row.values()) {
                match j.cmp(&i) {
                    std::cmp::Ordering::Less | std::cmp::Ordering::Greater => {
                        sum -= val * x[j];
                    }
                    std::cmp::Ordering::Equal => {
                        diag = val;
                    }
                }
            }
            x[i] = (<T as NumericElement>::ONE - self.omega) * x[i] + self.omega * sum / diag;
        }
    }

    fn backward_sweep(&self, b: &Array1<T>, x: &mut Array1<T>) {
        let n = self.matrix.nrows();
        for i in (0..n).rev() {
            let mut sum = b[i];
            let mut diag = <T as NumericElement>::ONE;
            let row = self.matrix.row(i);
            for (&j, &val) in row.col_indices().iter().zip(row.values()) {
                match j.cmp(&i) {
                    std::cmp::Ordering::Less | std::cmp::Ordering::Greater => {
                        sum -= val * x[j];
                    }
                    std::cmp::Ordering::Equal => {
                        diag = val;
                    }
                }
            }
            x[i] = (<T as NumericElement>::ONE - self.omega) * x[i] + self.omega * sum / diag;
        }
    }
}

impl<T: RealField + Copy + FloatElement + LetoScalar> Preconditioner<T> for SSORPreconditioner<T> {
    fn apply_to(&self, r: &Array1<T>, z: &mut Array1<T>) -> Result<()> {
        let n = self.matrix.nrows();
        let r_len = vector_len(r);
        if r_len != n {
            return Err(LetoError::InvalidInput(format!(
                "SSOR residual length mismatch: expected {n}, got {r_len}"
            )));
        }

        let z_len = vector_len(z);
        if z_len != n {
            return Err(LetoError::InvalidInput(format!(
                "SSOR output length mismatch: expected {n}, got {z_len}"
            )));
        }

        for idx in 0..z_len {
            z[idx] = <T as NumericElement>::ZERO;
        }
        self.forward_sweep(r, z);
        self.backward_sweep(r, z);
        Ok(())
    }
}
