//! In-place value and diagonal-scaling mutation for
//! [`CscMatrix`](super::CscMatrix).

use super::types::CscMatrix;
use crate::domain::scalar::Scalar;
use leto::{LetoError, Result};

impl<T: Scalar> CscMatrix<T> {
    /// Mutably borrow the stored nonzero values.
    #[must_use]
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Scale every stored nonzero by `factor`.
    pub fn scale_values(&mut self, factor: T) {
        for value in &mut self.values {
            *value *= factor;
        }
    }

    /// Scale each row by the corresponding `scaling[row]`.
    ///
    /// # Errors
    /// [`LetoError::ShapeMismatch`] if `scaling.len() != self.nrows()`.
    pub fn scale_rows(&mut self, scaling: &[T]) -> Result<()> {
        if scaling.len() != self.nrows {
            return Err(LetoError::ShapeMismatch {
                lhs: vec![self.nrows],
                rhs: vec![scaling.len()],
            });
        }

        for (value, &row) in self.values.iter_mut().zip(self.row_indices.iter()) {
            *value *= scaling[row];
        }

        Ok(())
    }

    /// Scale each column by the corresponding `scaling[column]`.
    ///
    /// # Errors
    /// [`LetoError::ShapeMismatch`] if `scaling.len() != self.ncols()`.
    pub fn scale_columns(&mut self, scaling: &[T]) -> Result<()> {
        if scaling.len() != self.ncols {
            return Err(LetoError::ShapeMismatch {
                lhs: vec![self.ncols],
                rhs: vec![scaling.len()],
            });
        }

        for (j, scaling_val) in scaling.iter().enumerate() {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                self.values[p] *= *scaling_val;
            }
        }
        Ok(())
    }
}
