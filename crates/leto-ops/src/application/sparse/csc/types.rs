//! [`CscMatrix`] and [`CscColumn`] storage layout.
//!
//! Fields are `pub(super)`: every operation-family submodule in
//! `application::sparse::csc` constructs or destructures them directly, but
//! nothing outside this module may.

use super::construction::validate_csc_parts;
use leto::Result;

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
    pub(super) values: Vec<T>,
    pub(super) row_indices: Vec<usize>,
    pub(super) col_ptr: Vec<usize>,
    pub(super) nrows: usize,
    pub(super) ncols: usize,
}

/// Borrowed view over one CSC column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CscColumn<'a, T> {
    pub(super) row_indices: &'a [usize],
    pub(super) values: &'a [T],
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

/// Borrowed CSC matrix view: the validated [`CscMatrix`] triple without owning it.
///
/// Construct with [`CscView::from_slices`] (validates the [`CscMatrix`]
/// invariants over borrowed slices — zero-copy, the view aliases its inputs)
/// or reborrow an owned matrix with [`CscMatrix::as_view`]. The CSC SpMV
/// kernel consumes the view
/// ([`csc_spmv_view_into`](super::super::csc_spmv_view_into)); the
/// owned-matrix entry keeps its signature and delegates through
/// [`CscMatrix::as_view`]. Structural and mutating operations stay on
/// [`CscMatrix`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CscView<'a, T> {
    pub(super) values: &'a [T],
    pub(super) row_indices: &'a [usize],
    pub(super) col_ptr: &'a [usize],
    pub(super) nrows: usize,
    pub(super) ncols: usize,
}

impl<'a, T> CscView<'a, T> {
    /// Borrow CSC parts after validating the [`CscMatrix`] invariants.
    ///
    /// # Errors
    /// [`LetoError::StorageError`](leto::LetoError::StorageError) on the same
    /// conditions as [`CscMatrix::from_parts`] (shared validator).
    pub fn from_slices(
        values: &'a [T],
        row_indices: &'a [usize],
        col_ptr: &'a [usize],
        nrows: usize,
        ncols: usize,
    ) -> Result<Self> {
        validate_csc_parts(values.len(), row_indices, col_ptr, nrows, ncols)?;
        Ok(Self {
            values,
            row_indices,
            col_ptr,
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

    /// Borrowed CSC arrays `(values, row_indices, col_ptr)` for kernels.
    #[must_use]
    #[inline]
    pub fn as_parts(&self) -> (&[T], &[usize], &[usize]) {
        (self.values, self.row_indices, self.col_ptr)
    }
}
