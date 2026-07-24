//! ILU(0) preconditioner (incomplete LU factorisation with zero fill-in).

use super::super::traits::Preconditioner;
use crate::CsrMatrix;
use crate::Scalar as LetoScalar;
use eunomia::{FloatElement, RealField};
use leto::{Array1, LetoError, Result};

/// ILU(0) preconditioner for a CSR matrix.
///
/// The factorisation keeps **the same sparsity pattern** as the original
/// matrix (zero fill-in).  The result is stored as a single CSR matrix
/// that encodes both L (strictly lower triangular part) and U (upper
/// triangular + diagonal).
///
/// ## Algorithm (IKJ form)
///
/// For each row `i`, for each column `k < i` in the pattern of row `i`:
/// ```text
/// a[i,k] /= a[k,k]
/// for j in row_pattern(i) ∩ col_pattern(k) with j > k:
///     a[i,j] -= a[i,k] * a[k,j]
/// ```
///
/// ## Solve step
///
/// Forward sweep (L·z = r) followed by back-substitution (U·z = z).
pub struct ILUPreconditioner<T: RealField + Copy> {
    // Combined L+U stored in CSR form (L strictly lower, U including diagonal).
    lu_values: Vec<T>,
    col_indices: Vec<usize>,
    row_offsets: Vec<usize>,
    diag_positions: Vec<usize>,
    n: usize,
}

impl<T: RealField + FloatElement + Copy + LetoScalar> ILUPreconditioner<T> {
    /// Factor a CSR matrix A with ILU(0).
    ///
    /// # Errors
    /// Returns [`LetoError::NumericalBreakdown`] if a zero pivot is encountered.
    pub fn factor(matrix: &CsrMatrix<T>) -> Result<Self> {
        let n = matrix.nrows();
        let nnz = matrix.values().len();

        let mut values: Vec<T> = matrix.values().to_vec();
        let col_indices: Vec<usize> = matrix.col_indices().to_vec();
        let row_offsets: Vec<usize> = matrix.row_ptr().to_vec();

        // Build diagonal-position lookup.
        let mut diag_positions = vec![0usize; n];
        for (row, &start) in row_offsets.iter().enumerate().take(n) {
            let end = row_offsets[row + 1];
            if let Some(pos) = col_indices[start..end].iter().position(|&c| c == row) {
                diag_positions[row] = start + pos;
            }
        }

        // IKJ ILU(0) sweep.
        for i in 1..n {
            let row_start = row_offsets[i];
            let row_end = row_offsets[i + 1];

            for ptr_k in row_start..row_end {
                let k = col_indices[ptr_k];
                if k >= i {
                    break;
                }
                let diag_k = values[diag_positions[k]];
                if diag_k.abs() < <T as FloatElement>::from_f64(1e-300) {
                    return Err(LetoError::NumericalBreakdown(format!(
                        "ILU(0): zero pivot at column {k}"
                    )));
                }
                let mult = values[ptr_k] / diag_k;
                values[ptr_k] = mult;

                // Update remaining entries in row i that share a column with row k.
                for ptr_j in ptr_k + 1..row_end {
                    let j = col_indices[ptr_j];
                    // Find a[k, j] in row k.
                    let k_start = row_offsets[k];
                    let k_end = row_offsets[k + 1];
                    for ptr_kj in k_start..k_end {
                        if col_indices[ptr_kj] == j {
                            values[ptr_j] = values[ptr_j] - mult * values[ptr_kj];
                            break;
                        }
                    }
                }
            }
        }

        let _ = nnz; // acknowledged
        Ok(Self {
            lu_values: values,
            col_indices,
            row_offsets,
            diag_positions,
            n,
        })
    }
}

impl<T: RealField + FloatElement + Copy + LetoScalar> Preconditioner<T> for ILUPreconditioner<T> {
    fn apply_to(&self, r: &Array1<T>, z: &mut Array1<T>) -> Result<()> {
        let n = self.n;
        if r.shape()[0] != n || z.shape()[0] != n {
            return Err(LetoError::InvalidInput(format!(
                "ILU preconditioner dimension {n} != r({}) or z({})",
                r.shape()[0],
                z.shape()[0]
            )));
        }

        // Forward substitution: L · y = r  (L has unit diagonal implicitly).
        for i in 0..n {
            let mut s = r[i];
            let start = self.row_offsets[i];
            let diag_pos = self.diag_positions[i];
            for k in start..diag_pos {
                s -= self.lu_values[k] * z[self.col_indices[k]];
            }
            z[i] = s;
        }

        // Back substitution: U · z = y.
        for i in (0..n).rev() {
            let mut s = z[i];
            let diag_pos = self.diag_positions[i];
            let end = self.row_offsets[i + 1];
            for k in diag_pos + 1..end {
                s -= self.lu_values[k] * z[self.col_indices[k]];
            }
            let d = self.lu_values[diag_pos];
            if d.abs() < <T as FloatElement>::from_f64(1e-300) {
                return Err(LetoError::NumericalBreakdown(format!(
                    "ILU solve: zero diagonal at row {i}"
                )));
            }
            z[i] = s / d;
        }
        Ok(())
    }
}
