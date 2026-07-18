//! Core traits for sparse array storage formats.

use eunomia::NumericElement;

/// Sparse matrix format identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    /// Coordinate format (COO) - efficient for construction
    Coo,
    /// Compressed Sparse Row (CSR) - efficient for row operations
    Csr,
    /// Compressed Sparse Column (CSC) - efficient for column operations
    Csc,
    /// Block-sparse format - efficient for structured sparsity
    Block,
}

/// Read-only sparse storage trait.
///
/// Provides the minimal interface required for read operations on sparse matrices.
/// All sparse formats implement this trait to enable format-agnostic algorithms.
pub trait SparseStorage<T: NumericElement> {
    /// Returns the number of rows in the sparse matrix.
    fn nrows(&self) -> usize;

    /// Returns the number of columns in the sparse matrix.
    fn ncols(&self) -> usize;

    /// Returns the number of non-zero elements.
    fn nnz(&self) -> usize;

    /// Returns the sparse format type.
    fn format(&self) -> SparseFormat;

    /// Returns the value at the specified position, or None if the entry is zero.
    fn get(&self, row: usize, col: usize) -> Option<T>;
}

/// Mutable sparse storage trait.
///
/// Extends read-only storage with mutation operations for building and modifying
/// sparse matrices.
pub trait SparseStorageMut<T: NumericElement>: SparseStorage<T> {
    /// Sets the value at the specified position.
    /// If the value is zero, the entry is removed from the sparse structure.
    fn set(&mut self, row: usize, col: usize, value: T);

    /// Adds a value to the existing entry at the specified position.
    /// If the entry doesn't exist, it's created.
    fn add(&mut self, row: usize, col: usize, delta: T);

    /// Removes the entry at the specified position (sets to zero).
    fn remove(&mut self, row: usize, col: usize);

    /// Reserves capacity for additional non-zero entries.
    fn reserve(&mut self, additional: usize);

    /// Clears all entries, preserving the matrix dimensions.
    fn clear(&mut self);
}
