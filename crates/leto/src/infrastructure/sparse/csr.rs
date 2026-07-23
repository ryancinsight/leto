//! Compressed Sparse Row (CSR) sparse matrix format.
//!
//! CSR format stores non-zero entries compressed by row.
//! This format is efficient for row-wise operations and sparse matrix-vector multiplication.

use crate::infrastructure::sparse::coo::CooArray;
use crate::infrastructure::sparse::traits::{SparseFormat, SparseStorage, SparseStorageMut};
use eunomia::NumericElement;

// Forward declarations
use super::CscArray;

/// Compressed Sparse Row (CSR) format sparse matrix.
///
/// Stores non-zero entries compressed by row using three arrays:
/// - `data`: Non-zero values
/// - `col_indices`: Column indices for each value
/// - `row_ptr`: Row pointers indicating where each row starts in data/col_indices
///
/// Efficient for row-wise operations and sparse matrix-vector multiplication.
#[derive(Debug, Clone)]
pub struct CsrArray<T: NumericElement> {
    /// Number of rows in the matrix.
    nrows: usize,
    /// Number of columns in the matrix.
    ncols: usize,
    /// Non-zero values in row-major order.
    data: Vec<T>,
    /// Column indices corresponding to values in `data`.
    col_indices: Vec<usize>,
    /// Row pointers: `row_ptr[i]` is the start index of row i in `data` and `col_indices`.
    /// `row_ptr[nrows]` equals `nnz`.
    row_ptr: Vec<usize>,
}

impl<T: NumericElement> CsrArray<T> {
    /// Creates a new CSR matrix with the given dimensions (empty).
    ///
    /// # Arguments
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            data: Vec::new(),
            col_indices: Vec::new(),
            row_ptr: vec![0; nrows + 1],
        }
    }

    /// Creates a CSR matrix from a sorted COO matrix.
    ///
    /// The COO matrix must be sorted by row, then column.
    pub fn from_coo(coo: CooArray<T>) -> Self {
        let nrows = coo.nrows();
        let ncols = coo.ncols();
        let nnz = coo.nnz();

        let mut data = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut row_ptr = vec![0usize; nrows + 1];

        let mut current_row = 0;
        for (row, col, value) in coo.entries() {
            // Update row pointers for skipped rows
            while current_row < row {
                current_row += 1;
                row_ptr[current_row] = data.len();
            }

            data.push(*value);
            col_indices.push(col);
        }

        // Fill remaining row pointers
        while current_row < nrows {
            current_row += 1;
            row_ptr[current_row] = data.len();
        }

        Self {
            nrows,
            ncols,
            data,
            col_indices,
            row_ptr,
        }
    }

    /// Returns an iterator over the non-zero entries in a specific row.
    ///
    /// # Arguments
    /// * `row` - Row index
    pub fn row_entries(&self, row: usize) -> impl Iterator<Item = (usize, &T)> {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        self.col_indices[start..end]
            .iter()
            .zip(self.data[start..end].iter())
            .map(|(&col, value)| (col, value))
    }

    /// Returns the number of non-zero entries in a specific row.
    pub fn row_nnz(&self, row: usize) -> usize {
        self.row_ptr[row + 1] - self.row_ptr[row]
    }

    /// Converts to CSC format (transpose operation).
    pub fn to_csc(&self) -> CscArray<T> {
        let mut coo = CooArray::with_capacity(self.ncols, self.nrows, self.nnz());
        for row in 0..self.nrows {
            for (col, value) in self.row_entries(row) {
                coo.add(col, row, *value);
            }
        }
        CscArray::from_coo(coo)
    }

    /// Converts to COO format.
    pub fn to_coo(&self) -> CooArray<T> {
        let mut triplets = Vec::with_capacity(self.nnz());
        for row in 0..self.nrows {
            for (col, value) in self.row_entries(row) {
                triplets.push((row, col, *value));
            }
        }
        CooArray::from_triplets(self.nrows, self.ncols, triplets)
    }
}

impl<T: NumericElement> SparseStorage<T> for CsrArray<T> {
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
        SparseFormat::Csr
    }

    fn get(&self, row: usize, col: usize) -> Option<T> {
        if row >= self.nrows || col >= self.ncols {
            return None;
        }

        for (c, value) in self.row_entries(row) {
            if c == col {
                return Some(*value);
            }
            if c > col {
                break; // Columns are sorted
            }
        }
        None
    }
}

impl<T: NumericElement> SparseStorageMut<T> for CsrArray<T> {
    fn set(&mut self, row: usize, col: usize, value: T) {
        // For simplicity, convert to COO, modify, and convert back
        // In production, this would be optimized for in-place modification
        let mut coo = self.to_coo();
        coo.set(row, col, value);
        *self = CsrArray::from_coo(coo);
    }

    fn add(&mut self, row: usize, col: usize, delta: T) {
        let mut coo = self.to_coo();
        coo.add(row, col, delta);
        *self = CsrArray::from_coo(coo);
    }

    fn remove(&mut self, row: usize, col: usize) {
        let mut coo = self.to_coo();
        coo.remove(row, col);
        *self = CsrArray::from_coo(coo);
    }

    fn reserve(&mut self, _additional: usize) {
        // CSR doesn't support direct reservation after construction
        // This is a no-op for the current implementation
    }

    fn clear(&mut self) {
        self.data.clear();
        self.col_indices.clear();
        for ptr in &mut self.row_ptr {
            *ptr = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_from_coo() {
        let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 1, 3.0)];
        let mut coo = CooArray::from_triplets(2, 2, triplets);
        coo.sort_by_row_column();

        let csr = CsrArray::from_coo(coo);
        assert_eq!(csr.nrows(), 2);
        assert_eq!(csr.ncols(), 2);
        assert_eq!(csr.nnz(), 3);
    }

    #[test]
    fn test_csr_row_entries() {
        let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 1, 3.0)];
        let mut coo = CooArray::from_triplets(2, 2, triplets);
        coo.sort_by_row_column();

        let csr = CsrArray::from_coo(coo);

        let row0: Vec<_> = csr.row_entries(0).collect();
        assert_eq!(row0.len(), 2);

        let row1: Vec<_> = csr.row_entries(1).collect();
        assert_eq!(row1.len(), 1);
    }

    #[test]
    fn test_csr_get() {
        let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 1, 3.0)];
        let mut coo = CooArray::from_triplets(2, 2, triplets);
        coo.sort_by_row_column();

        let csr = CsrArray::from_coo(coo);
        assert_eq!(csr.get(0, 0), Some(1.0));
        assert_eq!(csr.get(0, 1), Some(2.0));
        assert_eq!(csr.get(1, 1), Some(3.0));
        assert_eq!(csr.get(1, 0), None);
    }
}
