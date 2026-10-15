use super::policy::{Canonical, CsrIndex, CsrOrdering};
use super::structure::validate_csr_parts;
use core::marker::PhantomData;
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
pub struct CsrView<'a, T, I: CsrIndex = usize, O: CsrOrdering = Canonical> {
    pub(super) values: &'a [T],
    pub(super) col_indices: &'a [I],
    pub(super) row_ptr: &'a [I],
    pub(super) nrows: usize,
    pub(super) ncols: usize,
    pub(super) order: PhantomData<O>,
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
        Self::builder(values, col_indices, row_ptr, nrows, ncols).build()
    }
}

impl<'a, T, I: CsrIndex> CsrView<'a, T, I> {
    /// Start checked construction, inferring the original index representation.
    /// Canonical ordering is the default; select another policy with
    /// [`CsrViewBuilder::with_order`]. No input is copied or converted.
    #[inline]
    pub fn builder(
        values: &'a [T],
        col_indices: &'a [I],
        row_ptr: &'a [I],
        nrows: usize,
        ncols: usize,
    ) -> CsrViewBuilder<'a, T, I> {
        CsrViewBuilder {
            values,
            col_indices,
            row_ptr,
            nrows,
            ncols,
            order: PhantomData,
        }
    }
}

/// Unvalidated borrowed CSR parts. Only [`Self::build`] creates a usable view.
#[derive(Debug)]
pub struct CsrViewBuilder<'a, T, I: CsrIndex, O: CsrOrdering = Canonical> {
    values: &'a [T],
    col_indices: &'a [I],
    row_ptr: &'a [I],
    nrows: usize,
    ncols: usize,
    order: PhantomData<O>,
}

impl<'a, T, I: CsrIndex, O: CsrOrdering> CsrViewBuilder<'a, T, I, O> {
    /// Select the structural ordering invariant without changing input data.
    #[inline]
    pub fn with_order<P: CsrOrdering>(self) -> CsrViewBuilder<'a, T, I, P> {
        CsrViewBuilder {
            values: self.values,
            col_indices: self.col_indices,
            row_ptr: self.row_ptr,
            nrows: self.nrows,
            ncols: self.ncols,
            order: PhantomData,
        }
    }

    /// Validate lengths, target-width representability, offsets, columns and order.
    ///
    /// # Errors
    /// Returns [`leto::LetoError::StorageError`] for any invalid CSR invariant.
    #[inline]
    pub fn build(self) -> Result<CsrView<'a, T, I, O>> {
        let Self {
            values,
            col_indices,
            row_ptr,
            nrows,
            ncols,
            ..
        } = self;
        validate_csr_parts::<I, O>(values.len(), col_indices, row_ptr, nrows, ncols)?;
        Ok(CsrView {
            values,
            col_indices,
            row_ptr,
            nrows,
            ncols,
            order: PhantomData,
        })
    }
}

impl<'a, T, I: CsrIndex, O: CsrOrdering> CsrView<'a, T, I, O> {
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
    pub fn col_indices(&self) -> &'a [I] {
        self.col_indices
    }

    /// Borrowed CSR arrays `(values, col_indices, row_ptr)` for kernels.
    #[must_use]
    #[inline]
    pub fn as_parts(&self) -> (&'a [T], &'a [I], &'a [I]) {
        (self.values, self.col_indices, self.row_ptr)
    }
}
