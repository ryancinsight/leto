use super::structure::validate_csr_parts;
use leto::Result;

/// Borrowed view over one CSR row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CsrRow<'a, T> {
    pub(super) col_indices: &'a [usize],
    pub(super) values: &'a [T],
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

/// Borrowed CSR matrix view: the validated [`CsrMatrix`](super::CsrMatrix) triple without owning it.
///
/// Construct with [`CsrView::from_slices`] (validates the [`CsrMatrix`](super::CsrMatrix)
/// invariants over borrowed slices — zero-copy, the view aliases its inputs)
/// or reborrow an owned matrix with [`CsrMatrix::as_view`](super::CsrMatrix::as_view). The SpMV/SpMM
/// kernels consume the view ([`spmv_view_into`](super::super::spmv_view_into),
/// [`spmm_view_into`](super::super::spmm_view_into)); the owned-matrix `_into`
/// entries keep their signatures and delegate through [`CsrMatrix::as_view`](super::CsrMatrix::as_view).
/// Structural and mutating operations stay on [`CsrMatrix`](super::CsrMatrix).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CsrView<'a, T> {
    pub(super) values: &'a [T],
    pub(super) col_indices: &'a [usize],
    pub(super) row_ptr: &'a [usize],
    pub(super) nrows: usize,
    pub(super) ncols: usize,
}

impl<'a, T> CsrView<'a, T> {
    /// Borrow CSR parts after validating the [`CsrMatrix`](super::CsrMatrix) invariants.
    ///
    /// # Errors
    /// [`LetoError::StorageError`](leto::LetoError::StorageError) on the same conditions as
    /// [`CsrMatrix::from_parts`](super::CsrMatrix::from_parts) (shared validator).
    pub fn from_slices(
        values: &'a [T],
        col_indices: &'a [usize],
        row_ptr: &'a [usize],
        nrows: usize,
        ncols: usize,
    ) -> Result<Self> {
        validate_csr_parts(values.len(), col_indices, row_ptr, nrows, ncols)?;
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

    /// Borrow the column indices.
    #[must_use]
    #[inline]
    pub fn col_indices(&self) -> &'a [usize] {
        self.col_indices
    }

    /// Borrowed CSR arrays `(values, col_indices, row_ptr)` for kernels.
    #[must_use]
    #[inline]
    pub fn as_parts(&self) -> (&[T], &[usize], &[usize]) {
        (self.values, self.col_indices, self.row_ptr)
    }
}
