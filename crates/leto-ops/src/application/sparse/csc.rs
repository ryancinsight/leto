//! Compressed Sparse Column (CSC) storage — the column-major sparse format.
//!
//! CSC is the column-major analogue of CSR: columns are stored as contiguous
//! runs in `(values, row_indices)`, with `col_ptr[j]` marking the first index
//! of column `j`. CSC is preferred when column-wise access patterns dominate
//! (column extraction, matrix constructed from column vectors, certain
//! element-assembly loops, and transposed SpMV where CSR would need a
//! gather).
//!
//! The canonical pipeline is *assemble in COO →
//! [`to_csc`](CooMatrix::to_csc) → CSC kernels*.  Conversion to CSR via
//! [`to_csr`](Self::to_csr) (which is
//! [`transpose`](Self::transpose) of the CSR ↔ CSC duality) bridges to the
//! existing CSR solver surface.

use super::CsrMatrix;
use crate::domain::real::RealScalar;
use crate::domain::scalar::Scalar;
use eunomia::FloatElement;
use leto::{Array2, ArrayView2, LetoError, Result};

/// Compressed Sparse Column matrix: stores only the `nnz` nonzero entries
/// in column-major order.
///
/// Invariants (established by [`CscMatrix::from_dense`] and required by
/// [`CscMatrix::from_parts`]): `col_ptr.len() == ncols + 1`, `col_ptr` is
/// non-decreasing with `col_ptr[0] == 0` and `col_ptr[ncols] == values.len()`,
/// `row_indices.len() == values.len()`, every `row_indices[p] < nrows`, and
/// row indices are strictly increasing within each column.
#[derive(Debug, Clone, PartialEq)]
pub struct CscMatrix<T> {
    values: Vec<T>,
    row_indices: Vec<usize>,
    col_ptr: Vec<usize>,
    nrows: usize,
    ncols: usize,
}

/// Borrowed view over one CSC column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CscColumn<'a, T> {
    row_indices: &'a [usize],
    values: &'a [T],
}

impl<'a, T> CscColumn<'a, T> {
    /// Borrow the column row indices.
    #[must_use]
    #[inline]
    pub fn row_indices(&self) -> &'a [usize] {
        self.row_indices
    }

    /// Borrow the column values.
    #[must_use]
    #[inline]
    pub fn values(&self) -> &'a [T] {
        self.values
    }

    /// Number of stored entries in this column.
    #[must_use]
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

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

    /// `(nrows, ncols)`.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Number of stored nonzero entries.
    #[must_use]
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Number of rows.
    #[must_use]
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[must_use]
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Fraction of entries that are nonzero, in `[0, 1]`.
    #[must_use]
    #[inline]
    pub fn density(&self) -> f64 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }

    /// Borrowed CSC arrays `(values, row_indices, col_ptr)` — zero-copy.
    #[must_use]
    #[inline]
    pub fn as_parts(&self) -> (&[T], &[usize], &[usize]) {
        (&self.values, &self.row_indices, &self.col_ptr)
    }

    /// CSC column-pointer array.
    #[must_use]
    #[inline]
    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }

    /// CSC row-index array.
    #[must_use]
    #[inline]
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// CSC nonzero value array.
    #[must_use]
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Borrow one matrix column in CSC form.
    #[must_use]
    #[inline]
    pub fn column(&self, col: usize) -> CscColumn<'_, T> {
        let start = self.col_ptr[col];
        let end = self.col_ptr[col + 1];
        CscColumn {
            row_indices: &self.row_indices[start..end],
            values: &self.values[start..end],
        }
    }

    /// Extract the diagonal as a dense vector.
    #[must_use]
    pub fn diagonal(&self) -> Vec<T> {
        let mut diag = vec![T::ZERO; self.nrows.min(self.ncols)];
        for (col, d) in diag.iter_mut().enumerate() {
            for p in self.col_ptr[col]..self.col_ptr[col + 1] {
                if self.row_indices[p] == col {
                    *d = self.values[p];
                    break;
                }
            }
        }
        diag
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

#[cfg(test)]
mod tests {
    use super::{CscMatrix, CsrMatrix};

    #[test]
    fn from_dense_round_trips() {
        let dense =
            leto::Array2::from_shape_vec([3, 3], vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0])
                .unwrap();
        let csc = CscMatrix::from_dense(&dense.view());
        assert_eq!(csc.shape(), (3, 3));
        assert_eq!(csc.nnz(), 5);
        let round = csc.to_dense();
        assert_eq!(round, dense);
    }

    #[test]
    fn csc_identity_matrix() {
        let dense =
            leto::Array2::from_shape_vec([3, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
                .unwrap();
        let csc = CscMatrix::from_dense(&dense.view());
        assert_eq!(csc.shape(), (3, 3));
        assert_eq!(csc.nnz(), 3);
        let (values, row_indices, col_ptr) = csc.as_parts();
        assert_eq!(values, &[1.0, 1.0, 1.0]);
        assert_eq!(row_indices, &[0, 1, 2]);
        assert_eq!(col_ptr, &[0, 1, 2, 3]);
    }

    #[test]
    fn csc_column_access() {
        let dense =
            leto::Array2::from_shape_vec([3, 2], vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0]).unwrap();
        let csc = CscMatrix::from_dense(&dense.view());
        let col0 = csc.column(0);
        assert_eq!(col0.values(), &[1.0, 3.0]);
        assert_eq!(col0.row_indices(), &[0, 2]);
        let col1 = csc.column(1);
        assert_eq!(col1.values(), &[2.0]);
        assert_eq!(col1.row_indices(), &[1]);
    }

    #[test]
    fn csc_transpose_is_csr_of_transpose() {
        let dense =
            leto::Array2::from_shape_vec([2, 3], vec![1.0, 0.0, 2.0, 3.0, 4.0, 0.0]).unwrap();
        let csc = CscMatrix::from_dense(&dense.view());
        let csr_via_transpose = csc.transpose();
        let expected_csr = CsrMatrix::from_dense(&dense.view().transpose([1, 0]).unwrap());
        assert_eq!(csr_via_transpose, expected_csr);
    }

    #[test]
    fn csc_from_csr_round_trips() {
        let dense =
            leto::Array2::from_shape_vec([3, 3], vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0])
                .unwrap();
        let csr = CsrMatrix::from_dense(&dense.view());
        let csc = CscMatrix::from_csr(&csr);
        assert_eq!(csc.to_dense(), dense);
    }

    #[test]
    fn csc_zeros() {
        let csc = CscMatrix::<f64>::zeros(3, 4);
        assert_eq!(csc.shape(), (3, 4));
        assert_eq!(csc.nnz(), 0);
    }

    #[test]
    fn csc_diagonal() {
        let dense =
            leto::Array2::from_shape_vec([3, 3], vec![1.0, 0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 5.0])
                .unwrap();
        let csc = CscMatrix::from_dense(&dense.view());
        assert_eq!(csc.diagonal(), vec![1.0, 4.0, 5.0]);
    }

    #[test]
    fn csc_scale_values() {
        let dense =
            leto::Array2::from_shape_vec([3, 2], vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0]).unwrap();
        let mut csc = CscMatrix::from_dense(&dense.view());
        csc.scale_values(2.0);
        let expected =
            leto::Array2::from_shape_vec([3, 2], vec![2.0, 0.0, 0.0, 4.0, 6.0, 0.0]).unwrap();
        assert_eq!(csc.to_dense(), expected);
    }

    #[test]
    fn csc_frobenius_norm() {
        let dense = leto::Array2::from_shape_vec([2, 2], vec![3.0f64, 0.0, 0.0, 4.0]).unwrap();
        let csc = CscMatrix::from_dense(&dense.view());
        assert!((csc.frobenius_norm() - 5.0f64).abs() < 1e-12);
    }

    #[test]
    fn csc_diagonally_dominant() {
        let dd =
            leto::Array2::from_shape_vec([3, 3], vec![4.0, 1.0, 0.0, 1.0, 5.0, 2.0, 0.0, 2.0, 6.0])
                .unwrap();
        let csc = CscMatrix::from_dense(&dd.view());
        assert!(csc.is_strictly_diagonally_dominant());

        let non_dd = leto::Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 1.0]).unwrap();
        let csc2 = CscMatrix::from_dense(&non_dd.view());
        assert!(!csc2.is_strictly_diagonally_dominant());
    }

    #[test]
    fn from_parts_validates_invariants() {
        assert!(
            CscMatrix::<f64>::from_parts(
                vec![1.0, 2.0, 3.0],
                vec![0, 1, 2],
                vec![0, 2, 2, 3],
                3,
                3
            )
            .is_ok()
        );
        assert!(CscMatrix::<f64>::from_parts(vec![1.0], vec![0, 1], vec![0, 1], 2, 1).is_err());
        assert!(CscMatrix::<f64>::from_parts(vec![1.0], vec![5], vec![0, 1], 2, 1).is_err());
        assert!(CscMatrix::<f64>::from_parts(vec![1.0], vec![0], vec![1, 1], 2, 1).is_err());
    }
}
