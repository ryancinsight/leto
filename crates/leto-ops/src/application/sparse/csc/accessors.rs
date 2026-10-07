//! Shape, storage-array, column-view, and diagonal accessors for
//! [`CscMatrix`](super::CscMatrix).

use super::types::{CscColumn, CscMatrix, CscView};
use crate::domain::scalar::Scalar;

impl<T: Scalar> CscMatrix<T> {
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

    /// Reborrow as a [`CscView`]: zero-cost, aliases the owned arrays.
    ///
    /// Infallible: every constructor establishes the [`CscView::from_slices`]
    /// invariants, so no re-validation is needed.
    #[must_use]
    #[inline]
    pub fn as_view(&self) -> CscView<'_, T> {
        CscView {
            values: &self.values,
            row_indices: &self.row_indices,
            col_ptr: &self.col_ptr,
            nrows: self.nrows,
            ncols: self.ncols,
        }
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
}
