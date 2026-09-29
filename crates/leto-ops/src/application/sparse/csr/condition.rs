//! Diagonal-dominance conditioning estimate for [`CsrMatrix`].
//!
//! Split out of the storage module ([`super`]) so the `RealScalar`-only
//! heuristic does not bloat the CSR format definition.

use super::CsrMatrix;
use crate::domain::real::RealScalar;
use eunomia::FloatElement;
use leto::{LetoError, Result};

impl<T: RealScalar> CsrMatrix<T> {
    /// Estimate a square matrix's conditioning from diagonal dominance.
    ///
    /// This inexpensive structural heuristic returns
    /// `max_i((|a_ii| + sum_{j != i}|a_ij|) / |a_ii|)`, or `T::INFINITY`
    /// when any diagonal magnitude is below `1e-12`.
    ///
    /// # Errors
    /// [`LetoError::ShapeMismatch`] if the matrix is not square.
    #[must_use = "condition_estimate reports the computed estimate or a shape error"]
    pub fn condition_estimate(&self) -> Result<T> {
        if self.nrows != self.ncols {
            return Err(LetoError::ShapeMismatch {
                lhs: vec![self.nrows, self.nrows],
                rhs: vec![self.nrows, self.ncols],
            });
        }

        let mut max_ratio = T::ONE;
        let near_zero = <T as FloatElement>::from_f64(1.0e-12);

        for row in 0..self.nrows {
            let csr_row = self.row(row);
            let mut diagonal = T::ZERO;
            let mut off_diagonal_sum = T::ZERO;

            for (&column, &value) in csr_row.col_indices.iter().zip(csr_row.values.iter()) {
                let magnitude = value.abs();
                if column == row {
                    diagonal = magnitude;
                } else {
                    off_diagonal_sum += magnitude;
                }
            }

            if diagonal < near_zero {
                return Ok(T::INFINITY);
            }

            let ratio = (off_diagonal_sum + diagonal) / diagonal;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }

        Ok(max_ratio)
    }
}
