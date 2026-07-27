//! [`LinearOperator`] over [`CsrMatrix`].
//!
//! This is the seam impl that lets every iterative solver in this module run
//! directly against the Leto sparse storage type, without each caller writing
//! an adapter for the most common operator by far.

use super::super::traits::LinearOperator;
use crate::application::sparse::spmv_into;
use crate::domain::scalar::Scalar as LetoScalar;
use crate::CsrMatrix;
use eunomia::{NumericElement, RealField};
use leto::{Array1, LetoError, Result};

/// Non-contiguous output buffers would force a materialise-and-scatter pass on
/// every operator application; solvers in this module always own contiguous
/// vectors, so the case is rejected rather than silently paid for.
#[inline]
fn output_slice<T>(y: &mut Array1<T>) -> Result<&mut [T]> {
    y.as_slice_mut().ok_or_else(|| {
        LetoError::InvalidInput("CsrMatrix operator requires a contiguous output vector".into())
    })
}

impl<T> LinearOperator<T> for CsrMatrix<T>
where
    T: RealField + Copy + LetoScalar + Send + Sync,
{
    fn apply(&self, x: &Array1<T>, y: &mut Array1<T>) -> Result<()> {
        let view = x.view();
        spmv_into(self, &view, output_slice(y)?)
    }

    /// `nrows` for a square matrix, `0` (unchecked) otherwise, so rectangular
    /// operators reach the least-squares solvers through [`Self::nrows`] and
    /// [`Self::ncols`] without tripping the square-system dimension guard.
    fn size(&self) -> usize {
        let (rows, columns) = self.shape();
        if rows == columns { rows } else { 0 }
    }

    fn nrows(&self) -> usize {
        CsrMatrix::nrows(self)
    }

    fn ncols(&self) -> usize {
        CsrMatrix::ncols(self)
    }

    /// `y ← Aᵀ·x`, accumulated by scattering each stored row into `y`.
    ///
    /// Transposing the matrix first would cost an `O(nnz)` allocation and a
    /// full rebuild per application; the scatter reads the same CSR arrays the
    /// forward product does.
    fn apply_transpose(&self, x: &Array1<T>, y: &mut Array1<T>) -> Result<()> {
        let (rows, columns) = self.shape();
        if x.shape() != [rows] {
            return Err(LetoError::ShapeMismatch {
                lhs: x.shape().to_vec(),
                rhs: vec![rows],
            });
        }
        let out = output_slice(y)?;
        if out.len() != columns {
            return Err(LetoError::ShapeMismatch {
                lhs: vec![out.len()],
                rhs: vec![columns],
            });
        }

        out.fill(<T as NumericElement>::ZERO);
        let row_ptr = self.row_ptr();
        let col_indices = self.col_indices();
        let values = self.values();
        for row in 0..rows {
            let scale = x[row];
            for entry in row_ptr[row]..row_ptr[row + 1] {
                out[col_indices[entry]] += values[entry] * scale;
            }
        }
        Ok(())
    }
}
