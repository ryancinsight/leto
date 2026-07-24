//! Compressed Sparse Column (CSC) sparse matrix format.
//!
//! CSC format stores non-zero entries compressed by column.
//! This format is efficient for column-wise operations and transpose operations.

use crate::infrastructure::sparse::coo::CooArray;
use crate::infrastructure::sparse::traits::{SparseFormat, SparseStorage, SparseStorageMut};
use eunomia::NumericElement;

// Forward declarations
use super::CsrArray;

/// Compressed Sparse Column (CSC) format sparse matrix.
///
/// Stores non-zero entries compressed by column using three arrays:
/// - `data`: Non-zero values
/// - `row_indices`: Row indices for each value
/// - `col_ptr`: Column pointers indicating where each column starts in data/row_indices
///
/// Efficient for column-wise operations and transpose operations.
#[derive(Debug, Clone)]
pub struct CscArray<T: NumericElement> {
    /// Number of rows in the matrix.
    nrows: usize,
    /// Number of columns in the matrix.
    ncols: usize,
    /// Non-zero values in column-major order.
    data: Vec<T>,
    /// Row indices corresponding to values in `data`.
    row_indices: Vec<usize>,
    /// Column pointers: `col_ptr[j]` is the start index of column j in `data` and `row_indices`.
    /// `col_ptr[ncols]` equals `nnz`.
    col_ptr: Vec<usize>,
}

impl<T: NumericElement> CscArray<T> {
    /// Creates a new CSC matrix with the given dimensions (empty).
    ///
    /// # Arguments
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            data: Vec::new(),
            row_indices: Vec::new(),
            col_ptr: vec![0; ncols + 1],
        }
    }

    /// Creates a CSC matrix from COO entries.
    ///
    /// Entries are normalized to column-major order at this boundary.
    pub fn from_coo(coo: CooArray<T>) -> Self {
        let nrows = coo.nrows();
        let ncols = coo.ncols();
        let nnz = coo.nnz();
        let mut entries: Vec<_> = coo
            .entries()
            .map(|(row, col, value)| (row, col, *value))
            .collect();
        entries.sort_by_key(|&(row, col, _)| (col, row));

        let mut data = Vec::with_capacity(nnz);
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_ptr = vec![0usize; ncols + 1];

        let mut current_col = 0;
        for (row, col, value) in entries {
            // Update column pointers for skipped columns
            while current_col < col {
                current_col += 1;
                col_ptr[current_col] = data.len();
            }

            data.push(value);
            row_indices.push(row);
        }

        // Fill remaining column pointers
        while current_col < ncols {
            current_col += 1;
            col_ptr[current_col] = data.len();
        }

        Self {
            nrows,
            ncols,
            data,
            row_indices,
            col_ptr,
        }
    }

    /// Returns an iterator over the non-zero entries in a specific column.
    ///
    /// # Arguments
    /// * `col` - Column index
    pub fn col_entries(&self, col: usize) -> impl Iterator<Item = (usize, &T)> {
        let start = self.col_ptr[col];
        let end = self.col_ptr[col + 1];
        self.row_indices[start..end]
            .iter()
            .zip(self.data[start..end].iter())
            .map(|(&row, value)| (row, value))
    }

    /// Returns the number of non-zero entries in a specific column.
    pub fn col_nnz(&self, col: usize) -> usize {
        self.col_ptr[col + 1] - self.col_ptr[col]
    }

    /// Converts to CSR format (transpose operation).
    pub fn to_csr(&self) -> CsrArray<T> {
        let mut coo = CooArray::with_capacity(self.ncols, self.nrows, self.nnz());
        for col in 0..self.ncols {
            for (row, value) in self.col_entries(col) {
                coo.add(col, row, *value);
            }
        }
        coo.sort_by_row_column();
        CsrArray::from_coo(coo)
    }

    /// Converts to COO format.
    pub fn to_coo(&self) -> CooArray<T> {
        let mut triplets = Vec::with_capacity(self.nnz());
        for col in 0..self.ncols {
            for (row, value) in self.col_entries(col) {
                triplets.push((row, col, *value));
            }
        }
        CooArray::from_triplets(self.nrows, self.ncols, triplets)
    }
}

impl<T: NumericElement> SparseStorage<T> for CscArray<T> {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }

    fn nnz(&self) -> usize {
        self.data.len()
    }

    fn format(&self) -> SparseFormat {
        SparseFormat::Csc
    }

    fn get(&self, row: usize, col: usize) -> Option<T> {
        if row >= self.nrows || col >= self.ncols {
            return None;
        }

        for (r, value) in self.col_entries(col) {
            if r == row {
                return Some(*value);
            }
            if r > row {
                break; // Rows are sorted within each column
            }
        }
        None
    }
}

impl<T: NumericElement> SparseStorageMut<T> for CscArray<T> {
    fn set(&mut self, row: usize, col: usize, value: T) {
        // For simplicity, convert to COO, modify, and convert back
        // In production, this would be optimized for in-place modification
        let mut coo = self.to_coo();
        coo.set(row, col, value);
        *self = CscArray::from_coo(coo);
    }

    fn add(&mut self, row: usize, col: usize, delta: T) {
        let mut coo = self.to_coo();
        coo.add(row, col, delta);
        *self = CscArray::from_coo(coo);
    }

    fn remove(&mut self, row: usize, col: usize) {
        let mut coo = self.to_coo();
        coo.remove(row, col);
        *self = CscArray::from_coo(coo);
    }

    fn reserve(&mut self, _additional: usize) {
        // CSC doesn't support direct reservation after construction
        // This is a no-op for the current implementation
    }

    fn clear(&mut self) {
        self.data.clear();
        self.row_indices.clear();
        for ptr in &mut self.col_ptr {
            *ptr = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csc_from_coo() {
        let triplets = vec![(1, 1, 3.0), (1, 0, 2.0), (0, 0, 1.0)];
        let coo = CooArray::from_triplets(2, 2, triplets);
        let csc = CscArray::from_coo(coo);
        assert_eq!(csc.nrows(), 2);
        assert_eq!(csc.ncols(), 2);
        assert_eq!(csc.nnz(), 3);
        assert_eq!(csc.get(0, 0), Some(1.0));
        assert_eq!(csc.get(1, 0), Some(2.0));
        assert_eq!(csc.get(1, 1), Some(3.0));
    }

    #[test]
    fn test_csc_col_entries() {
        let triplets = vec![(0, 0, 1.0), (1, 0, 2.0), (1, 1, 3.0)];
        let coo = CooArray::from_triplets(2, 2, triplets);
        let csc = CscArray::from_coo(coo);

        let col0: Vec<_> = csc
            .col_entries(0)
            .map(|(row, value)| (row, *value))
            .collect();
        let col1: Vec<_> = csc
            .col_entries(1)
            .map(|(row, value)| (row, *value))
            .collect();
        assert_eq!(col0, [(0, 1.0), (1, 2.0)]);
        assert_eq!(col1, [(1, 3.0)]);
    }

    #[test]
    fn test_csc_get() {
        let triplets = vec![(0, 0, 1.0), (1, 0, 2.0), (1, 1, 3.0)];
        let coo = CooArray::from_triplets(2, 2, triplets);
        let csc = CscArray::from_coo(coo);
        assert_eq!(csc.get(0, 0), Some(1.0));
        assert_eq!(csc.get(1, 0), Some(2.0));
        assert_eq!(csc.get(1, 1), Some(3.0));
        assert_eq!(csc.get(0, 1), None);
    }
}
