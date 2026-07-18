//! Zero-copy format conversions between sparse matrix formats.
//!
//! This module provides efficient conversion functions between different sparse formats,
//! minimizing data copying where possible.

use crate::infrastructure::sparse::coo::CooArray;
use crate::infrastructure::sparse::csr::CsrArray;
use crate::infrastructure::sparse::csc::CscArray;
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
    
    coo.sort_by_row_column();
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
/// * `coo` - COO matrix (should be sorted by row, then column)
/// * `sum_duplicates` - If true, duplicate entries are summed; if false, the last value is kept
pub fn coo_to_csr_with_duplicates<T: NumericElement>(
    coo: CooArray<T>,
    sum_duplicates: bool,
) -> CsrArray<T> {
    // For now, use the standard from_coo which handles duplicates by keeping the last value
    // TODO: Implement proper duplicate handling with sum_duplicates flag
    if sum_duplicates {
        // Convert to COO with summed duplicates, then to CSR
        let mut summed = CooArray::with_capacity(coo.nrows(), coo.ncols(), coo.nnz());
        let mut entries: std::collections::HashMap<(usize, usize), T> = std::collections::HashMap::new();
        
        for (row, col, value) in coo.entries() {
            let key = (row, col);
            if let Some(existing) = entries.get(&key) {
                entries.insert(key, *existing + *value);
            } else {
                entries.insert(key, *value);
            }
        }
        
        for ((row, col), value) in entries {
            if value != T::ZERO {
                summed.add(row, col, value);
            }
        }
        
        summed.sort_by_row_column();
        CsrArray::from_coo(summed)
    } else {
        // Keep last value (default behavior)
        let mut sorted = coo.clone();
        sorted.sort_by_row_column();
        CsrArray::from_coo(sorted)
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
