//! Norm, diagonal-dominance, and conditioning-estimate properties for
//! [`CscMatrix`](super::CscMatrix).

use super::types::CscMatrix;
use crate::domain::real::RealScalar;
use crate::domain::scalar::Scalar;
use eunomia::FloatElement;
use leto::{LetoError, Result};

impl<T: Scalar> CscMatrix<T> {
    /// Frobenius norm `sqrt(sum_ij A_ij^2)`.
    #[must_use]
    pub fn frobenius_norm(&self) -> T {
        self.values
            .iter()
            .copied()
            .fold(T::ZERO, |acc, value| acc + value * value)
            .sqrt()
    }

    /// Return whether every row is strictly diagonally dominant by absolute row
    /// sum: `|a_ii| > sum_{j != i} |a_ij|`.
    #[must_use]
    pub fn is_strictly_diagonally_dominant(&self) -> bool {
        if self.nrows != self.ncols {
            return false;
        }

        for row in 0..self.nrows {
            let mut diagonal = T::ZERO;
            let mut off_diagonal_sum = T::ZERO;

            for col in 0..self.ncols {
                for p in self.col_ptr[col]..self.col_ptr[col + 1] {
                    if self.row_indices[p] == row {
                        let magnitude = self.values[p].abs();
                        if col == row {
                            diagonal = magnitude;
                        } else {
                            off_diagonal_sum += magnitude;
                        }
                    }
                }
            }

            if diagonal <= off_diagonal_sum {
                return false;
            }
        }

        true
    }
}

impl<T: RealScalar> CscMatrix<T> {
    /// Estimate a square matrix's conditioning from diagonal dominance.
    ///
    /// Returns `max_i((|a_ii| + sum_{j != i}|a_ij|) / |a_ii|)`, or `T::INFINITY`
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
            let mut diagonal = T::ZERO;
            let mut off_diagonal_sum = T::ZERO;

            for col in 0..self.ncols {
                for p in self.col_ptr[col]..self.col_ptr[col + 1] {
                    if self.row_indices[p] == row {
                        let magnitude = self.values[p].abs();
                        if col == row {
                            diagonal = magnitude;
                        } else {
                            off_diagonal_sum += magnitude;
                        }
                    }
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
