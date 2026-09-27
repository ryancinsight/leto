//! Conversion between [`CscMatrix`](super::CscMatrix), [`CsrMatrix`], and
//! dense arrays.

use super::types::CscMatrix;
use crate::application::sparse::CsrMatrix;
use crate::domain::scalar::Scalar;
use leto::Array2;

impl<T: Scalar> CscMatrix<T> {
    /// Return the CSC transpose `A^T` as a [`CsrMatrix`].
    ///
    /// CSC(A)^T = CSR(A) by duality.  Counting nonzeros per output row (one per
    /// source column), prefix-scanning into `row_ptr`, then scattering each
    /// source entry `(i, j, value)` into output row `j` with column `i`.
    #[must_use = "transpose returns the transposed sparse matrix (CSR)"]
    pub fn transpose(&self) -> CsrMatrix<T> {
        // row_counts per output CSR row = per source CSC column.
        let mut row_counts = vec![0usize; self.ncols];
        for (j, count) in row_counts.iter_mut().enumerate() {
            *count = self.col_ptr[j + 1] - self.col_ptr[j];
        }

        let mut row_ptr = Vec::with_capacity(self.ncols + 1);
        row_ptr.push(0);
        for count in &row_counts {
            row_ptr.push(row_ptr.last().copied().expect("row_ptr has seed") + count);
        }

        let mut next = row_ptr[..self.ncols].to_vec();
        let nnz = self.values.len();
        let mut values = vec![T::ZERO; nnz];
        let mut col_indices = vec![0usize; nnz];

        for source_col in 0..self.ncols {
            for p in self.col_ptr[source_col]..self.col_ptr[source_col + 1] {
                let out_row = source_col;
                let out_col = self.row_indices[p];
                let target = next[out_row];
                values[target] = self.values[p];
                col_indices[target] = out_col;
                next[out_row] += 1;
            }
        }

        CsrMatrix::from_parts(values, col_indices, row_ptr, self.ncols, self.nrows)
            .expect("CSC transpose invariants are preserved")
    }

    /// Convert CSC to CSR by transposition (delegates to [`transpose`](Self::transpose)).
    #[must_use = "to_csr returns the row-compressed matrix"]
    pub fn to_csr(&self) -> CsrMatrix<T> {
        self.transpose()
    }

    /// Reconstruct the dense matrix (`O(n·m)`; inverse of [`from_dense`](Self::from_dense)).
    #[must_use]
    pub fn to_dense(&self) -> Array2<T> {
        let mut dense = vec![T::ZERO; self.nrows * self.ncols];
        for j in 0..self.ncols {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                dense[self.row_indices[p] * self.ncols + j] = self.values[p];
            }
        }
        Array2::from_shape_vec([self.nrows, self.ncols], dense).expect("CSC dense shape is valid")
    }
}
