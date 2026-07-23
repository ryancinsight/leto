//! Zero-copy format conversions between sparse matrix formats.
//!
//! This module provides efficient conversion functions between different sparse formats,
//! minimizing data copying where possible.

use crate::infrastructure::sparse::coo::CooArray;
use crate::infrastructure::sparse::csc::CscArray;
use crate::infrastructure::sparse::csr::CsrArray;
use crate::infrastructure::sparse::traits::{SparseStorage, SparseStorageMut};
use eunomia::NumericElement;

/// Converts a CSR matrix to CSC format (transpose operation).
///
/// This is a true transpose operation, not just a format conversion.
pub fn csr_to_csc_transpose<T: NumericElement>(csr: &CsrArray<T>) -> CscArray<T> {
    let mut coo = CooArray::with_capacity(csr.ncols(), csr.nrows(), csr.nnz());

    for row in 0..csr.nrows() {
        for (col, value) in csr.row_entries(row) {
            coo.add(col, row, *value);
        }
    }

    CscArray::from_coo(coo)
}

/// Converts a CSC matrix to CSR format (transpose operation).
///
/// This is a true transpose operation, not just a format conversion.
pub fn csc_to_csr_transpose<T: NumericElement>(csc: &CscArray<T>) -> CsrArray<T> {
    let mut coo = CooArray::with_capacity(csc.ncols(), csc.nrows(), csc.nnz());

    for col in 0..csc.ncols() {
        for (row, value) in csc.col_entries(col) {
            coo.add(col, row, *value);
        }
    }

    coo.sort_by_row_column();
    CsrArray::from_coo(coo)
}

/// Converts COO to CSR with duplicate handling.
///
/// # Arguments
/// * `coo` - COO matrix
/// * `sum_duplicates` - If true, duplicate entries are summed; if false, the last value is kept
pub fn coo_to_csr_with_duplicates<T: NumericElement>(
    mut coo: CooArray<T>,
    sum_duplicates: bool,
) -> CsrArray<T> {
    coo.sort_by_row_column();

    let mut compact = CooArray::with_capacity(coo.nrows(), coo.ncols(), coo.nnz());
    let mut pending: Option<(usize, usize, T)> = None;

    for (row, col, value) in coo.entries() {
        match pending.as_mut() {
            Some((pending_row, pending_col, pending_value))
                if *pending_row == row && *pending_col == col =>
            {
                *pending_value = if sum_duplicates {
                    *pending_value + *value
                } else {
                    *value
                };
            }
            Some(_) => {
                let (pending_row, pending_col, pending_value) = pending
                    .replace((row, col, *value))
                    .expect("invariant: match arm proves the pending entry exists");
                if !sum_duplicates || pending_value != T::ZERO {
                    compact.add(pending_row, pending_col, pending_value);
                }
            }
            None => pending = Some((row, col, *value)),
        }
    }

    if let Some((row, col, value)) = pending {
        if !sum_duplicates || value != T::ZERO {
            compact.add(row, col, value);
        }
    }

    CsrArray::from_coo(compact)
}

#[cfg(test)]
mod duplicate_contract {
    use super::*;

    #[test]
    fn duplicate_policy_is_order_independent_between_coordinates() {
        let coo =
            CooArray::from_triplets(2, 2, [(1, 1, 3.0), (0, 0, 1.0), (0, 0, 2.0), (1, 1, -3.0)]);

        let summed = coo_to_csr_with_duplicates(coo.clone(), true);
        assert_eq!(summed.nnz(), 1);
        assert_eq!(summed.get(0, 0), Some(3.0));
        assert_eq!(summed.get(1, 1), None);

        let kept = coo_to_csr_with_duplicates(coo, false);
        assert_eq!(kept.nnz(), 2);
        assert_eq!(kept.get(0, 0), Some(2.0));
        assert_eq!(kept.get(1, 1), Some(-3.0));
    }
}

/// Converts CSR to COO (zero-copy view where possible).
///
/// This creates a COO matrix that shares data with the CSR matrix
/// when the underlying storage allows it.
pub fn csr_to_coo_view<T: NumericElement>(csr: &CsrArray<T>) -> CooArray<T> {
    let mut triplets = Vec::with_capacity(csr.nnz());

    for row in 0..csr.nrows() {
        for (col, value) in csr.row_entries(row) {
            triplets.push((row, col, *value));
        }
    }

    CooArray::from_triplets(csr.nrows(), csr.ncols(), triplets)
}

/// Converts CSC to COO (zero-copy view where possible).
///
/// This creates a COO matrix that shares data with the CSC matrix
/// when the underlying storage allows it.
pub fn csc_to_coo_view<T: NumericElement>(csc: &CscArray<T>) -> CooArray<T> {
    let mut triplets = Vec::with_capacity(csc.nnz());

    for col in 0..csc.ncols() {
        for (row, value) in csc.col_entries(col) {
            triplets.push((row, col, *value));
        }
    }

    CooArray::from_triplets(csc.nrows(), csc.ncols(), triplets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_to_csc_transpose() {
        let triplets = vec![(0, 1, 1.0), (1, 0, 2.0)];
        let mut coo = CooArray::from_triplets(2, 2, triplets);
        coo.sort_by_row_column();

        let csr = CsrArray::from_coo(coo);
        let csc = csr_to_csc_transpose(&csr);

        // After transpose, (0,1) becomes (1,0) and (1,0) becomes (0,1)
        assert_eq!(csc.get(1, 0), Some(1.0));
        assert_eq!(csc.get(0, 1), Some(2.0));
    }

    #[test]
    fn test_coo_to_csr_sum_duplicates() {
        let triplets = vec![(0, 0, 1.0), (0, 0, 2.0), (1, 1, 3.0)];
        let mut coo = CooArray::from_triplets(2, 2, triplets);
        coo.sort_by_row_column();

        let csr = coo_to_csr_with_duplicates(coo, true);
        assert_eq!(csr.get(0, 0), Some(3.0)); // 1.0 + 2.0
        assert_eq!(csr.get(1, 1), Some(3.0));
    }

    #[test]
    fn test_coo_to_csr_keep_last() {
        let triplets = vec![(0, 0, 1.0), (0, 0, 2.0), (1, 1, 3.0)];
        let mut coo = CooArray::from_triplets(2, 2, triplets);
        coo.sort_by_row_column();

        let csr = coo_to_csr_with_duplicates(coo, false);
        assert_eq!(csr.get(0, 0), Some(2.0)); // Keep last value
        assert_eq!(csr.get(1, 1), Some(3.0));
    }
}
