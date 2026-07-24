//! Jacobi (diagonal) preconditioner.

use super::super::traits::Preconditioner;
use crate::CsrMatrix;
use crate::Scalar as LetoScalar;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, LetoError, Result};

/// Jacobi (diagonal scaling) preconditioner: `z ← D⁻¹ · r`.
///
/// Built once from the diagonal of a CSR matrix; the inverse diagonal is
/// stored so the apply step is a single pass of divisions.  Near-zero
/// diagonal entries are clamped to a small epsilon to avoid blow-up.
pub struct JacobiPreconditioner<T: RealField + Copy> {
    inv_diagonal: Array1<T>,
}

#[inline]
fn diagonal_epsilon<T: FloatElement>() -> T {
    <T as FloatElement>::from_f64(1e-14)
}

impl<T: RealField + FloatElement + Copy + LetoScalar> JacobiPreconditioner<T> {
    /// Construct from the diagonal of a CSR matrix.
    ///
    /// Missing diagonal entries (structurally zero) are treated as `epsilon`
    /// rather than zero so the preconditioner is always well-defined.
    pub fn from_matrix(matrix: &CsrMatrix<T>) -> Self {
        let n = matrix.nrows();
        let mut inv_diag = Array1::zeros([n]);
        for row in 0..n {
            let mut d = diagonal_epsilon::<T>();
            let start = matrix.row_ptr()[row];
            let end = matrix.row_ptr()[row + 1];
            for k in start..end {
                if matrix.col_indices()[k] == row {
                    let diag_val = matrix.values()[k];
                    if diag_val.abs() > diagonal_epsilon::<T>() {
                        d = diag_val;
                    }
                    break;
                }
            }
            inv_diag[row] = <T as NumericElement>::ONE / d;
        }
        Self { inv_diagonal: inv_diag }
    }

    /// Construct from an explicit inverse-diagonal vector (zero-copy path).
    pub fn from_inv_diagonal(inv_diag: Array1<T>) -> Self {
        Self { inv_diagonal: inv_diag }
    }
}

impl<T: RealField + FloatElement + Copy + LetoScalar> Preconditioner<T> for JacobiPreconditioner<T> {
    fn apply_to(&self, r: &Array1<T>, z: &mut Array1<T>) -> Result<()> {
        let n = r.shape()[0];
        if z.shape()[0] != n {
            return Err(LetoError::InvalidInput(format!(
                "Jacobi preconditioner output length {}, expected {n}",
                z.shape()[0]
            )));
        }
        for i in 0..n {
            z[i] = r[i] * self.inv_diagonal[i];
        }
        Ok(())
    }
}
