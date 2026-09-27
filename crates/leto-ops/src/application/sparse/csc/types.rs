//! [`CscMatrix`] and [`CscColumn`] storage layout.
//!
//! Fields are `pub(super)`: every operation-family submodule in
//! `application::sparse::csc` constructs or destructures them directly, but
//! nothing outside this module may.

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
