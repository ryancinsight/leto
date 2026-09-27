//! Construction of [`CscMatrix`](super::CscMatrix) from dense arrays, raw
//! parts, CSR transposition, and the all-zero matrix.

use super::types::CscMatrix;
use crate::application::sparse::CsrMatrix;
use crate::domain::scalar::Scalar;
use leto::{ArrayView2, LetoError, Result};

impl<T: Scalar> CscMatrix<T> {
    /// Compress a dense matrix view into CSC, dropping every exact-zero entry.
    #[must_use = "from_dense returns the compressed matrix"]
    pub fn from_dense(matrix: &ArrayView2<'_, T>) -> Self {
        let [nrows, ncols] = matrix.shape();
        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_ptr = Vec::with_capacity(ncols + 1);
        col_ptr.push(0);

        if let Some(dense) = matrix.as_slice() {
            for j in 0..ncols {
                for i in 0..nrows {
                    let value = dense[i * ncols + j];
                    if value != T::ZERO {
                        values.push(value);
                        row_indices.push(i);
                    }
                }
                col_ptr.push(values.len());
            }
        } else {
            let strides = matrix.strides();
            let data = matrix.data();
            for j in 0..ncols {
                if nrows == 0 {
                    col_ptr.push(values.len());
                    continue;
                }
                let mut offset = matrix
                    .layout()
                    .offset_of([0, j])
                    .expect("column start is in bounds") as isize;
                for _ in 0..nrows {
                    let value = data[offset as usize];
                    if value != T::ZERO {
                        values.push(value);
                        row_indices.push(offset as usize / ncols);
                    }
                    offset += strides[0];
                }
                col_ptr.push(values.len());
            }
        }

        Self {
            values,
            row_indices,
            col_ptr,
            nrows,
            ncols,
        }
    }

    /// Construct CSC from raw parts, validating the structural invariants.
    ///
    /// # Errors
    /// [`LetoError::StorageError`] if any CSC invariant is violated.
    pub fn from_parts(
        values: Vec<T>,
        row_indices: Vec<usize>,
        col_ptr: Vec<usize>,
        nrows: usize,
        ncols: usize,
    ) -> Result<Self> {
        let bad = |reason: &str| LetoError::StorageError {
            reason: format!("invalid CSC: {reason}"),
        };
        let expected_col_ptr_len = ncols
            .checked_add(1)
            .ok_or_else(|| LetoError::StorageError {
                reason: "invalid CSC: ncols + 1 overflows usize".to_string(),
            })?;
        if col_ptr.len() != expected_col_ptr_len {
            return Err(bad("col_ptr length must be ncols + 1"));
        }
        if row_indices.len() != values.len() {
            return Err(bad("row_indices and values lengths differ"));
        }
        if col_ptr[0] != 0 || *col_ptr.last().expect("ncols+1 >= 1") != values.len() {
            return Err(bad("col_ptr must start at 0 and end at nnz"));
        }
        if col_ptr.windows(2).any(|w| w[0] > w[1]) {
            return Err(bad("col_ptr must be non-decreasing"));
        }
        if row_indices.iter().any(|&i| i >= nrows) {
            return Err(bad("row index out of range"));
        }
        for window in col_ptr.windows(2) {
            let col_rows = &row_indices[window[0]..window[1]];
            if col_rows.windows(2).any(|rows| rows[0] >= rows[1]) {
                return Err(bad(
                    "row indices in each column must be strictly increasing",
                ));
            }
        }
        Ok(Self {
            values,
            row_indices,
            col_ptr,
            nrows,
            ncols,
        })
    }

    /// Build CSC from CSR via transpose (O(nnz)).
    ///
    /// CSR row i corresponds to CSC column i in A^T.
    #[must_use = "from_csr returns the column-compressed matrix"]
    pub fn from_csr(csr: &CsrMatrix<T>) -> Self {
        let nrows = csr.nrows();
        let ncols = csr.ncols();
        let mut col_counts = vec![0usize; ncols];
        for &col in csr.col_indices() {
            col_counts[col] += 1;
        }

        let mut col_ptr = Vec::with_capacity(ncols + 1);
        col_ptr.push(0);
        for count in col_counts {
            col_ptr.push(col_ptr.last().copied().expect("col_ptr has seed") + count);
        }

        let mut next = col_ptr[..ncols].to_vec();
        let mut values = vec![T::ZERO; csr.nnz()];
        let mut row_indices = vec![0usize; csr.nnz()];

        for row in 0..nrows {
            let (src_vals, src_cols, src_ptr) = csr.as_parts();
            for p in src_ptr[row]..src_ptr[row + 1] {
                let col = src_cols[p];
                let target = next[col];
                values[target] = src_vals[p];
                row_indices[target] = row;
                next[col] += 1;
            }
        }

        Self {
            values,
            row_indices,
            col_ptr,
            nrows,
            ncols,
        }
    }

    /// Construct an all-zero CSC matrix with shape `(nrows, ncols)`.
    #[must_use = "zeros returns the constructed sparse matrix"]
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            values: Vec::new(),
            row_indices: Vec::new(),
            col_ptr: vec![0; ncols.saturating_add(1)],
            nrows,
            ncols,
        }
    }
}
