//! Coordinate (COO) sparse matrix format.
//!
//! COO format stores non-zero entries as triplets (row, column, value).
//! This format is efficient for matrix construction and format conversion.

use crate::infrastructure::sparse::traits::{SparseFormat, SparseStorage, SparseStorageMut};
use eunomia::NumericElement;

// Forward declarations
use super::{CscArray, CsrArray};

/// Coordinate format sparse matrix.
///
/// Stores non-zero entries as (row, column, value) triplets.
/// Efficient for construction and format conversion, but less efficient
/// for arithmetic operations compared to CSR/CSC.
#[derive(Debug, Clone)]
pub struct CooArray<T: NumericElement> {
    /// Number of rows in the matrix.
    nrows: usize,
    /// Number of columns in the matrix.
    ncols: usize,
    /// Row indices of non-zero entries.
    row_indices: Vec<usize>,
    /// Column indices of non-zero entries.
    col_indices: Vec<usize>,
    /// Values of non-zero entries.
    data: Vec<T>,
}

impl<T: NumericElement> CooArray<T> {
    /// Creates a new empty COO matrix with the given dimensions.
    ///
    /// # Arguments
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Creates a new COO matrix with pre-allocated capacity.
    ///
    /// # Arguments
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `capacity` - Pre-allocated capacity for non-zero entries
    pub fn with_capacity(nrows: usize, ncols: usize, capacity: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_indices: Vec::with_capacity(capacity),
            col_indices: Vec::with_capacity(capacity),
            data: Vec::with_capacity(capacity),
        }
    }

    /// Creates a COO matrix from triplets.
    ///
    /// # Arguments
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `triplets` - Iterator of (row, col, value) tuples
    pub fn from_triplets<I>(nrows: usize, ncols: usize, triplets: I) -> Self
    where
        I: IntoIterator<Item = (usize, usize, T)>,
    {
        let triplets: Vec<_> = triplets.into_iter().collect();
        let capacity = triplets.len();

        let mut row_indices = Vec::with_capacity(capacity);
        let mut col_indices = Vec::with_capacity(capacity);
        let mut data = Vec::with_capacity(capacity);

        for (row, col, value) in triplets {
            row_indices.push(row);
            col_indices.push(col);
            data.push(value);
        }

        Self {
            nrows,
            ncols,
            row_indices,
            col_indices,
            data,
        }
    }

    /// Returns an iterator over the non-zero entries as (row, col, value) triplets.
    pub fn entries(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        self.row_indices
            .iter()
            .zip(self.col_indices.iter())
            .zip(self.data.iter())
            .map(|((&row, &col), value)| (row, col, value))
    }

    /// Sorts entries by row, then column (required for conversion to CSR).
    pub fn sort_by_row_column(&mut self) {
        let mut entries: Vec<_> = self
            .row_indices
            .iter()
            .zip(self.col_indices.iter())
            .zip(self.data.iter())
            .map(|((&row, &col), &value)| (row, col, value))
            .collect();

        entries.sort_by_key(|&(row, col, _value)| (row, col));

        self.row_indices.clear();
        self.col_indices.clear();
        self.data.clear();

        for (row, col, value) in entries {
            self.row_indices.push(row);
            self.col_indices.push(col);
            self.data.push(value);
        }
    }

    /// Converts to CSR format.
    pub fn to_csr(&self) -> CsrArray<T> {
        let mut sorted = self.clone();
        sorted.sort_by_row_column();
        CsrArray::from_coo(sorted)
    }

    /// Converts to CSC format.
    pub fn to_csc(&self) -> CscArray<T> {
        self.to_csr().to_csc()
    }
}

impl<T: NumericElement> SparseStorage<T> for CooArray<T> {
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
        SparseFormat::Coo
    }

    fn get(&self, row: usize, col: usize) -> Option<T> {
        for (i, (&r, &c)) in self
            .row_indices
            .iter()
            .zip(self.col_indices.iter())
            .enumerate()
        {
            if r == row && c == col {
                return Some(self.data[i]);
            }
        }
        None
    }
}

impl<T: NumericElement> SparseStorageMut<T> for CooArray<T> {
    fn set(&mut self, row: usize, col: usize, value: T) {
        // Check if entry exists
        for i in 0..self.row_indices.len() {
            if self.row_indices[i] == row && self.col_indices[i] == col {
                if value == T::ZERO {
                    // Remove entry if setting to zero
                    self.row_indices.remove(i);
                    self.col_indices.remove(i);
                    self.data.remove(i);
                } else {
                    self.data[i] = value;
                }
                return;
            }
        }

        // Add new entry if value is non-zero
        if value != T::ZERO {
            self.row_indices.push(row);
            self.col_indices.push(col);
            self.data.push(value);
        }
    }

    fn add(&mut self, row: usize, col: usize, delta: T) {
        for i in 0..self.row_indices.len() {
            if self.row_indices[i] == row && self.col_indices[i] == col {
                self.data[i] += delta;
                if self.data[i] == T::ZERO {
                    self.row_indices.remove(i);
                    self.col_indices.remove(i);
                    self.data.remove(i);
                }
                return;
            }
        }

        // Add new entry if delta is non-zero
        if delta != T::ZERO {
            self.row_indices.push(row);
            self.col_indices.push(col);
            self.data.push(delta);
        }
    }

    fn remove(&mut self, row: usize, col: usize) {
        for i in 0..self.row_indices.len() {
            if self.row_indices[i] == row && self.col_indices[i] == col {
                self.row_indices.remove(i);
                self.col_indices.remove(i);
                self.data.remove(i);
                return;
            }
        }
    }

    fn reserve(&mut self, additional: usize) {
        self.row_indices.reserve(additional);
        self.col_indices.reserve(additional);
        self.data.reserve(additional);
    }

    fn clear(&mut self) {
        self.row_indices.clear();
        self.col_indices.clear();
        self.data.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_construction() {
        let coo = CooArray::<f64>::new(3, 3);
        assert_eq!(coo.nrows(), 3);
        assert_eq!(coo.ncols(), 3);
        assert_eq!(coo.nnz(), 0);
    }

    #[test]
    fn test_coo_from_triplets() {
        let triplets = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
        let coo = CooArray::from_triplets(3, 3, triplets);
        assert_eq!(coo.nnz(), 3);
        assert_eq!(coo.get(0, 0), Some(1.0));
        assert_eq!(coo.get(1, 1), Some(2.0));
        assert_eq!(coo.get(2, 2), Some(3.0));
    }

    #[test]
    fn test_coo_set_get() {
        let mut coo = CooArray::<f64>::new(3, 3);
        coo.set(0, 0, 1.0);
        assert_eq!(coo.get(0, 0), Some(1.0));
        assert_eq!(coo.get(1, 1), None);

        coo.set(0, 0, 0.0); // Set to zero should remove
        assert_eq!(coo.get(0, 0), None);
    }

    #[test]
    fn test_coo_add() {
        let mut coo = CooArray::<f64>::new(3, 3);
        coo.add(0, 0, 1.0);
        coo.add(0, 0, 2.0);
        assert_eq!(coo.get(0, 0), Some(3.0));
    }

    #[test]
    fn test_coo_remove() {
        let mut coo = CooArray::from_triplets(3, 3, vec![(0, 0, 1.0)]);
        coo.remove(0, 0);
        assert_eq!(coo.nnz(), 0);
        assert_eq!(coo.get(0, 0), None);
    }
}
