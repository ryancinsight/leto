//! Compressed Sparse Row (CSR) storage — the solve/kernel-facing sparse format.
//!
//! See the [module theorem](super) for the representation identity and the
//! `O(nnz)` kernel-complexity argument. CSR is the format [`spmv`](super::spmv)
//! and [`spmm`](super::spmm) consume; assembly happens in
//! [`CooMatrix`](super::CooMatrix), which converts here via
//! [`CooMatrix::to_csr`](super::CooMatrix::to_csr).

use crate::domain::real::RealScalar;
use crate::domain::scalar::Scalar;
use eunomia::FloatElement;
use leto::{Array2, ArrayView2, LetoError, Result};

/// Compressed Sparse Row matrix: stores only the `nnz` nonzero entries.
///
/// Invariants (established by [`CsrMatrix::from_dense`] and required by
/// [`CsrMatrix::from_parts`]): `row_ptr.len() == nrows + 1`, `row_ptr` is
/// non-decreasing with `row_ptr[0] == 0` and `row_ptr[nrows] == values.len()`,
/// `col_indices.len() == values.len()`, every `col_indices[p] < ncols`, and
/// column indices are strictly increasing within each row.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix<T> {
    values: Vec<T>,
    col_indices: Vec<usize>,
    row_ptr: Vec<usize>,
    nrows: usize,
    ncols: usize,
}

/// Borrowed view over one CSR row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CsrRow<'a, T> {
    col_indices: &'a [usize],
    values: &'a [T],
}

impl<'a, T> CsrRow<'a, T> {
    /// Borrow the row column indices.
    #[must_use]
    #[inline]
    pub fn col_indices(&self) -> &'a [usize] {
        self.col_indices
    }

    /// Borrow the row values.
    #[must_use]
    #[inline]
    pub fn values(&self) -> &'a [T] {
        self.values
    }

    /// Number of stored entries in this row.
    #[must_use]
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

impl<T: Scalar> CsrMatrix<T> {
    /// Compress a dense matrix view into CSR, dropping every exact-zero entry.
    ///
    /// This is the automatic dense→sparse path: it scans the matrix once
    /// (`O(n·m)`) and retains only nonzeros, after which every kernel over the
    /// result is `O(nnz)`. Worthwhile when [`density`](Self::density) is small.
    #[must_use = "from_dense returns the compressed matrix"]
    pub fn from_dense(matrix: &ArrayView2<'_, T>) -> Self {
        let [nrows, ncols] = matrix.shape();
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        row_ptr.push(0);

        if let Some(dense) = matrix.as_slice() {
            for i in 0..nrows {
                let row = &dense[i * ncols..i * ncols + ncols];
                for (j, &value) in row.iter().enumerate() {
                    if value != T::ZERO {
                        values.push(value);
                        col_indices.push(j);
                    }
                }
                row_ptr.push(values.len());
            }
        } else {
            let strides = matrix.strides();
            let data = matrix.data();
            for i in 0..nrows {
                if ncols == 0 {
                    row_ptr.push(values.len());
                    continue;
                }
                let mut offset = matrix
                    .layout()
                    .offset_of([i, 0])
                    .expect("row start is in bounds") as isize;
                for j in 0..ncols {
                    let value = data[offset as usize];
                    if value != T::ZERO {
                        values.push(value);
                        col_indices.push(j);
                    }
                    offset += strides[1];
                }
                row_ptr.push(values.len());
            }
        }

        Self {
            values,
            col_indices,
            row_ptr,
            nrows,
            ncols,
        }
    }

    /// Construct CSR from raw parts, validating the structural invariants.
    ///
    /// # Errors
    /// [`LetoError::StorageError`] if any CSR invariant is violated (mismatched
    /// lengths, non-monotone `row_ptr`, or an out-of-range column index).
    pub fn from_parts(
        values: Vec<T>,
        col_indices: Vec<usize>,
        row_ptr: Vec<usize>,
        nrows: usize,
        ncols: usize,
    ) -> Result<Self> {
        let bad = |reason: &str| LetoError::StorageError {
            reason: format!("invalid CSR: {reason}"),
        };
        let expected_row_ptr_len = nrows
            .checked_add(1)
            .ok_or_else(|| LetoError::StorageError {
                reason: "invalid CSR: nrows + 1 overflows usize".to_string(),
            })?;
        if row_ptr.len() != expected_row_ptr_len {
            return Err(bad("row_ptr length must be nrows + 1"));
        }
        if col_indices.len() != values.len() {
            return Err(bad("col_indices and values lengths differ"));
        }
        if row_ptr[0] != 0 || *row_ptr.last().expect("nrows+1 >= 1") != values.len() {
            return Err(bad("row_ptr must start at 0 and end at nnz"));
        }
        if row_ptr.windows(2).any(|w| w[0] > w[1]) {
            return Err(bad("row_ptr must be non-decreasing"));
        }
        if col_indices.iter().any(|&j| j >= ncols) {
            return Err(bad("column index out of range"));
        }
        for window in row_ptr.windows(2) {
            let row_cols = &col_indices[window[0]..window[1]];
            if row_cols.windows(2).any(|cols| cols[0] >= cols[1]) {
                return Err(bad(
                    "column indices in each row must be strictly increasing",
                ));
            }
        }
        Ok(Self {
            values,
            col_indices,
            row_ptr,
            nrows,
            ncols,
        })
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

    /// Fraction of entries that are nonzero, in `[0, 1]` (`0` for an empty matrix).
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

    /// Borrowed CSR arrays `(values, col_indices, row_ptr)` — zero-copy, for
    /// kernels (e.g. [`spmv`](super::spmv)) and for bridging to an external SpMV
    /// backend.
    #[must_use]
    #[inline]
    pub fn as_parts(&self) -> (&[T], &[usize], &[usize]) {
        (&self.values, &self.col_indices, &self.row_ptr)
    }

    /// CSR row-offset array.
    #[must_use]
    #[inline]
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// CSR column-index array.
    #[must_use]
    #[inline]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// CSR nonzero value array.
    #[must_use]
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Borrow one matrix row in CSR form.
    #[must_use]
    #[inline]
    pub fn row(&self, row: usize) -> CsrRow<'_, T> {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        CsrRow {
            col_indices: &self.col_indices[start..end],
            values: &self.values[start..end],
        }
    }

    /// Extract the diagonal as a dense vector.
    #[must_use]
    pub fn diagonal(&self) -> Vec<T> {
        let mut diag = vec![T::ZERO; self.nrows];
        for (row, d) in diag.iter_mut().enumerate().take(self.nrows) {
            for p in self.row_ptr[row]..self.row_ptr[row + 1] {
                if self.col_indices[p] == row {
                    *d = self.values[p];
                    break;
                }
            }
        }
        diag
    }

    /// Construct an all-zero CSR matrix with shape `(nrows, ncols)`.
    #[must_use = "zeros returns the constructed sparse matrix"]
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptr: vec![0; nrows.saturating_add(1)],
            nrows,
            ncols,
        }
    }

    /// Mutably borrow the stored nonzero values.
    ///
    /// This preserves CSR structural invariants because row pointers and column
    /// indices remain immutable. Use it for value-only transforms such as
    /// scaling or replacing assembled coefficients.
    #[must_use]
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Scale every stored nonzero by `factor`.
    ///
    /// The operation mutates values only; row pointers and column indices are
    /// unchanged, so all CSR structural invariants are preserved.
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

        for (row, &scale) in scaling.iter().enumerate() {
            for index in self.row_ptr[row]..self.row_ptr[row + 1] {
                self.values[index] *= scale;
            }
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

        for (value, &column) in self.values.iter_mut().zip(self.col_indices.iter()) {
            *value *= scaling[column];
        }

        Ok(())
    }

    /// Frobenius norm `sqrt(sum_ij A_ij^2)`.
    ///
    /// The reduction executes in `T`'s native precision through
    /// [`eunomia::NumericElement`]; no hidden wider accumulator is introduced.
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
    ///
    /// Non-square matrices are not diagonally dominant under this contract.
    #[must_use]
    pub fn is_strictly_diagonally_dominant(&self) -> bool {
        if self.nrows != self.ncols {
            return false;
        }

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

            if diagonal <= off_diagonal_sum {
                return false;
            }
        }

        true
    }

    /// Return the CSR transpose `A^T`.
    ///
    /// The implementation counts nonzeros per output row, prefix-scans those
    /// counts into the transposed `row_ptr`, then scatters each source entry
    /// `(i, j, value)` into output row `j` with column `i`. Source rows are
    /// traversed in ascending order, so each transposed row receives strictly
    /// increasing column indices and preserves the [`from_parts`](Self::from_parts)
    /// CSR invariant.
    #[must_use = "transpose returns the transposed sparse matrix"]
    pub fn transpose(&self) -> Self {
        let mut row_counts = vec![0usize; self.ncols];
        for &col in &self.col_indices {
            row_counts[col] += 1;
        }

        let mut row_ptr = Vec::with_capacity(self.ncols + 1);
        row_ptr.push(0);
        for count in row_counts {
            row_ptr.push(row_ptr.last().copied().expect("row_ptr has seed") + count);
        }

        let mut next = row_ptr[..self.ncols].to_vec();
        let mut values = vec![T::ZERO; self.values.len()];
        let mut col_indices = vec![0usize; self.col_indices.len()];

        for source_row in 0..self.nrows {
            for source_index in self.row_ptr[source_row]..self.row_ptr[source_row + 1] {
                let transposed_row = self.col_indices[source_index];
                let target_index = next[transposed_row];
                values[target_index] = self.values[source_index];
                col_indices[target_index] = source_row;
                next[transposed_row] += 1;
            }
        }

        Self {
            values,
            col_indices,
            row_ptr,
            nrows: self.ncols,
            ncols: self.nrows,
        }
    }

    /// Reconstruct the dense matrix (`O(n·m)`; inverse of [`from_dense`](Self::from_dense)).
    #[must_use]
    pub fn to_dense(&self) -> Array2<T> {
        let mut dense = vec![T::ZERO; self.nrows * self.ncols];
        for i in 0..self.nrows {
            for p in self.row_ptr[i]..self.row_ptr[i + 1] {
                dense[i * self.ncols + self.col_indices[p]] = self.values[p];
            }
        }
        Array2::from_shape_vec([self.nrows, self.ncols], dense).expect("CSR dense shape is valid")
    }
}

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
