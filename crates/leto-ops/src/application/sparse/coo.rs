//! Coordinate-list (COO) storage — the **assembly-facing** sparse format.
//!
//! COO holds an unordered list of `(row, col, value)` triplets and is the
//! natural target for matrix assembly (finite-element/finite-volume stiffness
//! and mass matrices, graph adjacency): each contribution is a single
//! [`push`](CooMatrix::push), duplicates at the same `(i, j)` are permitted and
//! **accumulate**. Once assembly is complete, [`to_csr`](CooMatrix::to_csr)
//! compresses to the solve/kernel-facing [`CsrMatrix`](super::CsrMatrix) that
//! [`spmv`](super::spmv)/[`spmm`](super::spmm) consume.
//!
//! # Theorem (COO→CSR preserves the assembled matrix)
//! Let the triplets define `A[i,j] = Σ_{p : (rowᵢ,colⱼ)=(i,j)} valueₚ` (the
//! duplicate-accumulation semantics). [`to_csr`](CooMatrix::to_csr) sorts the
//! triplets by `(row, col)`, sums each maximal equal-`(i,j)` run into one entry,
//! and drops entries whose sum is exactly zero. The result stores, for each
//! `(i,j)` with `A[i,j] ≠ 0`, exactly that value with strictly increasing column
//! indices per row — i.e. a valid CSR equal to `A`. Sorting is `O(nnz log nnz)`,
//! the accumulation pass `O(nnz)`. ∎

use super::{CscMatrix, CsrMatrix};
use crate::domain::scalar::Scalar;
use leto::{Array2, LetoError, Result};

/// Coordinate-list sparse matrix: an unordered, duplicate-tolerant triplet list
/// for assembly. Convert to [`CsrMatrix`] with [`to_csr`](Self::to_csr) before
/// running kernels.
#[derive(Debug, Clone, PartialEq)]
pub struct CooMatrix<T> {
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<T>,
    nrows: usize,
    ncols: usize,
}

impl<T: Scalar> CooMatrix<T> {
    /// An empty `nrows × ncols` matrix ready for assembly.
    #[must_use]
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            nrows,
            ncols,
        }
    }

    /// An empty matrix pre-sized for `cap` triplets (assembly without reallocation).
    #[must_use]
    pub fn with_capacity(nrows: usize, ncols: usize, cap: usize) -> Self {
        Self {
            row_indices: Vec::with_capacity(cap),
            col_indices: Vec::with_capacity(cap),
            values: Vec::with_capacity(cap),
            nrows,
            ncols,
        }
    }

    /// Append a `(row, col, value)` contribution. Duplicates accumulate in
    /// [`to_csr`](Self::to_csr).
    ///
    /// # Panics
    /// If `row >= nrows` or `col >= ncols` — an assembly logic error (the indices
    /// are caller-computed, not external input).
    #[inline]
    pub fn push(&mut self, row: usize, col: usize, value: T) {
        assert!(
            row < self.nrows && col < self.ncols,
            "COO index ({row}, {col}) out of bounds for {}x{} matrix",
            self.nrows,
            self.ncols
        );
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(value);
    }

    /// Build from parallel triplet arrays, validating bounds.
    ///
    /// # Errors
    /// [`LetoError::StorageError`] if the three arrays differ in length or any
    /// index is out of range.
    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self> {
        let bad = |reason: &str| LetoError::StorageError {
            reason: format!("invalid COO: {reason}"),
        };
        if row_indices.len() != values.len() || col_indices.len() != values.len() {
            return Err(bad("row/col/value arrays must have equal length"));
        }
        if row_indices.iter().any(|&i| i >= nrows) {
            return Err(bad("row index out of range"));
        }
        if col_indices.iter().any(|&j| j >= ncols) {
            return Err(bad("column index out of range"));
        }
        Ok(Self {
            row_indices,
            col_indices,
            values,
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

    /// Number of stored triplets (before duplicate accumulation; an upper bound
    /// on the CSR `nnz`).
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether no triplets have been pushed.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Compress to CSR, summing duplicate `(i, j)` entries and dropping exact
    /// zeros (`O(nnz log nnz)`; see the [module theorem](crate::application::sparse)).
    #[must_use = "to_csr returns the compressed matrix"]
    pub fn to_csr(&self) -> CsrMatrix<T> {
        let nnz_in = self.values.len();
        let mut order: Vec<usize> = (0..nnz_in).collect();
        order.sort_unstable_by_key(|&p| (self.row_indices[p], self.col_indices[p]));

        let mut values: Vec<T> = Vec::with_capacity(nnz_in);
        let mut col_indices: Vec<usize> = Vec::with_capacity(nnz_in);
        let mut row_ptr = vec![0usize; self.nrows + 1];

        let mut k = 0;
        while k < nnz_in {
            let p = order[k];
            let (i, j) = (self.row_indices[p], self.col_indices[p]);
            let mut sum = self.values[p];
            let mut k2 = k + 1;
            while k2 < nnz_in {
                let q = order[k2];
                if self.row_indices[q] == i && self.col_indices[q] == j {
                    sum = sum.add(self.values[q]);
                    k2 += 1;
                } else {
                    break;
                }
            }
            if sum != T::ZERO {
                values.push(sum);
                col_indices.push(j);
                row_ptr[i + 1] += 1;
            }
            k = k2;
        }
        for i in 0..self.nrows {
            row_ptr[i + 1] += row_ptr[i];
        }

        CsrMatrix::from_parts(values, col_indices, row_ptr, self.nrows, self.ncols)
            .expect("invariant: COO→CSR emits sorted, deduplicated, in-range entries")
    }

    /// Compress to CSC, summing duplicate `(i, j)` entries and dropping exact
    /// zeros (`O(nnz log nnz)`).
    #[must_use = "to_csc returns the column-compressed matrix"]
    pub fn to_csc(&self) -> CscMatrix<T> {
        let nnz_in = self.values.len();
        let mut order: Vec<usize> = (0..nnz_in).collect();
        order.sort_unstable_by_key(|&p| (self.col_indices[p], self.row_indices[p]));

        let mut values: Vec<T> = Vec::with_capacity(nnz_in);
        let mut row_indices: Vec<usize> = Vec::with_capacity(nnz_in);
        let mut col_ptr = vec![0usize; self.ncols + 1];

        let mut k = 0;
        while k < nnz_in {
            let p = order[k];
            let (i, j) = (self.row_indices[p], self.col_indices[p]);
            let mut sum = self.values[p];
            let mut k2 = k + 1;
            while k2 < nnz_in {
                let q = order[k2];
                if self.row_indices[q] == i && self.col_indices[q] == j {
                    sum = sum.add(self.values[q]);
                    k2 += 1;
                } else {
                    break;
                }
            }
            if sum != T::ZERO {
                values.push(sum);
                row_indices.push(i);
                col_ptr[j + 1] += 1;
            }
            k = k2;
        }
        for j in 0..self.ncols {
            col_ptr[j + 1] += col_ptr[j];
        }

        CscMatrix::from_parts(values, row_indices, col_ptr, self.nrows, self.ncols)
            .expect("invariant: COO→CSC emits sorted, deduplicated, in-range entries")
    }

    /// Reconstruct the dense matrix by accumulating triplets (`O(nnz + n·m)`;
    /// for testing/inspection).
    #[must_use]
    pub fn to_dense(&self) -> Array2<T> {
        let mut dense = vec![T::ZERO; self.nrows * self.ncols];
        for p in 0..self.values.len() {
            let idx = self.row_indices[p] * self.ncols + self.col_indices[p];
            dense[idx] = dense[idx].add(self.values[p]);
        }
        Array2::from_shape_vec([self.nrows, self.ncols], dense).expect("COO dense shape is valid")
    }
}

#[cfg(test)]
mod tests {
    use super::{CooMatrix, CsrMatrix};

    #[test]
    fn coo_to_csc_round_trips() {
        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.push(0, 0, 2.0);
        coo.push(0, 0, 3.0); // accumulates to 5.0
        coo.push(1, 2, -1.0);
        coo.push(2, 1, 4.0);
        let csc = coo.to_csc();
        assert_eq!(csc.shape(), (3, 3));
        assert_eq!(csc.nnz(), 3);
        // CSC should match dense round-trip.
        assert_eq!(csc.to_dense(), coo.to_dense());
    }

    #[test]
    fn coo_to_csc_matches_csr_via_transpose() {
        let mut coo = CooMatrix::<f64>::new(3, 4);
        coo.push(0, 1, 2.0);
        coo.push(1, 0, 3.0);
        coo.push(2, 3, -1.0);
        coo.push(0, 0, 1.0);
        let csc = coo.to_csc();
        assert_eq!(csc.to_dense(), coo.to_dense());
        // CSC → CSR gives CSR(A^T); verify the round-trip is valid.
        assert_eq!(csc.to_csr().nnz(), csc.nnz());
        assert_eq!(csc.to_csr().nrows(), csc.ncols());
        assert_eq!(csc.to_csr().ncols(), csc.nrows());
    }

    #[test]
    fn to_dense_round_trips_through_csr() {
        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.push(0, 0, 2.0);
        coo.push(0, 0, 3.0); // accumulates to 5.0 in to_dense
        coo.push(1, 2, -1.0);
        // COO → dense → CSR must equal COO → CSR directly (both sum dups, drop zeros).
        let via_dense = CsrMatrix::from_dense(&coo.to_dense().view());
        assert_eq!(via_dense, coo.to_csr());
    }

    #[test]
    fn push_accumulates_duplicates_into_csr() {
        // 3x3 with a duplicate at (0,0): 2 + 3 = 5, and an entry that cancels.
        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.push(0, 0, 2.0);
        coo.push(0, 0, 3.0); // duplicate -> 5.0
        coo.push(1, 2, -1.0);
        coo.push(2, 1, 4.0);
        coo.push(2, 1, -4.0); // cancels -> dropped
        let csr = coo.to_csr();
        assert_eq!(csr.shape(), (3, 3));
        // (0,0)=5, (1,2)=-1; (2,1) cancelled => nnz = 2
        assert_eq!(csr.nnz(), 2);
        let (values, cols, row_ptr) = csr.as_parts();
        assert_eq!(values, &[5.0, -1.0]); // row-major nonzero order
        assert_eq!(cols, &[0, 2]);
        assert_eq!(row_ptr, &[0, 1, 2, 2]); // row 0: 1 nnz, row 1: 1 nnz, row 2: empty
    }

    #[test]
    fn coo_to_csr_sorts_and_sums_duplicates() {
        let mut coo = CooMatrix::<f64>::new(2, 4);
        // (0,1) appears twice -> 1.5 + 0.5 = 2.0; entries arrive out of column order.
        for &(i, j, v) in &[(0, 3, 2.0), (0, 1, 1.5), (1, 0, -3.0), (0, 1, 0.5)] {
            coo.push(i, j, v);
        }
        let csr = coo.to_csr();
        let (values, cols, row_ptr) = csr.as_parts();
        // row 0: cols sorted ascending (1, 3) with the duplicate summed; row 1: (0)
        assert_eq!(values, &[2.0, 2.0, -3.0]);
        assert_eq!(cols, &[1, 3, 0]);
        assert_eq!(row_ptr, &[0, 2, 3]);
    }

    #[test]
    fn from_triplets_validates_bounds() {
        assert!(CooMatrix::from_triplets(2, 2, vec![0, 5], vec![0, 1], vec![1.0, 2.0]).is_err());
        assert!(CooMatrix::from_triplets(2, 2, vec![0], vec![0, 1], vec![1.0]).is_err());
        assert!(CooMatrix::from_triplets(2, 2, vec![0, 1], vec![0, 1], vec![1.0, 2.0]).is_ok());
    }
}
