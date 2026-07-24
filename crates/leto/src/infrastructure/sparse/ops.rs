//! Sparse-dense arithmetic operations.
//!
//! This module provides arithmetic operations between sparse and dense arrays,
//! following the principle that sparse APIs mirror dense APIs.

use crate::infrastructure::sparse::csc::CscArray;
use crate::infrastructure::sparse::csr::CsrArray;
use crate::infrastructure::sparse::traits::{SparseStorage, SparseStorageMut};
use eunomia::NumericElement;

/// Sparse matrix-dense vector multiplication (CSR format).
///
/// Computes y = A * x where A is sparse (CSR) and x is a dense slice.
pub fn csr_dense_matvec<T: NumericElement>(csr: &CsrArray<T>, x: &[T]) -> Vec<T> {
    assert_eq!(
        csr.ncols(),
        x.len(),
        "Matrix columns must match vector length"
    );

    let mut y = vec![T::ZERO; csr.nrows()];

    for (row, output) in y.iter_mut().enumerate() {
        let mut sum = T::ZERO;
        for (col, value) in csr.row_entries(row) {
            sum += *value * x[col];
        }
        *output = sum;
    }

    y
}

/// Sparse matrix-dense vector multiplication (CSC format).
///
/// Computes y = A * x where A is sparse (CSC) and x is a dense slice.
pub fn csc_dense_matvec<T: NumericElement>(csc: &CscArray<T>, x: &[T]) -> Vec<T> {
    assert_eq!(
        csc.ncols(),
        x.len(),
        "Matrix columns must match vector length"
    );

    let mut y = vec![T::ZERO; csc.nrows()];

    for (col, &x_col) in x.iter().enumerate() {
        for (row, value) in csc.col_entries(col) {
            y[row] += *value * x_col;
        }
    }

    y
}

/// Dense vector-sparse matrix multiplication (CSR format).
///
/// Computes y = x^T * A where x is a dense slice and A is sparse (CSR).
pub fn dense_csr_matvec<T: NumericElement>(x: &[T], csr: &CsrArray<T>) -> Vec<T> {
    assert_eq!(x.len(), csr.nrows(), "Vector length must match matrix rows");

    let mut y = vec![T::ZERO; csr.ncols()];

    for (row, &x_row) in x.iter().enumerate() {
        for (col, value) in csr.row_entries(row) {
            y[col] += x_row * *value;
        }
    }

    y
}

/// Dense vector-sparse matrix multiplication (CSC format).
///
/// Computes y = x^T * A where x is a dense slice and A is sparse (CSC).
pub fn dense_csc_matvec<T: NumericElement>(x: &[T], csc: &CscArray<T>) -> Vec<T> {
    assert_eq!(x.len(), csc.nrows(), "Vector length must match matrix rows");

    let mut y = vec![T::ZERO; csc.ncols()];

    for (col, output) in y.iter_mut().enumerate() {
        let mut sum = T::ZERO;
        for (row, value) in csc.col_entries(col) {
            sum += x[row] * *value;
        }
        *output = sum;
    }

    y
}

/// Sparse matrix-sparse matrix addition (CSR format).
///
/// Computes C = A + B where both matrices are sparse (CSR).
pub fn csr_add_csr<T: NumericElement>(a: &CsrArray<T>, b: &CsrArray<T>) -> CsrArray<T> {
    assert_eq!(a.nrows(), b.nrows(), "Matrix dimensions must match");
    assert_eq!(a.ncols(), b.ncols(), "Matrix dimensions must match");

    // Convert to COO for easier addition
    let mut coo_a = a.to_coo();
    let coo_b = b.to_coo();

    // Add entries from B to A
    for (row, col, value) in coo_b.entries() {
        coo_a.add(row, col, *value);
    }

    coo_a.sort_by_row_column();
    CsrArray::from_coo(coo_a)
}

/// Sparse matrix-sparse matrix multiplication (CSR format).
///
/// Computes C = A * B where both matrices are sparse (CSR).
/// Uses the standard sparse matrix multiplication algorithm.
pub fn csr_mul_csr<T: NumericElement>(a: &CsrArray<T>, b: &CsrArray<T>) -> CsrArray<T> {
    assert_eq!(a.ncols(), b.nrows(), "A columns must match B rows");

    let nrows = a.nrows();
    let ncols = b.ncols();
    let mut coo = crate::infrastructure::sparse::coo::CooArray::with_capacity(
        nrows,
        ncols,
        a.nnz() + b.nnz(),
    );

    // Standard sparse matrix multiplication
    for i in 0..nrows {
        for (k, a_ik) in a.row_entries(i) {
            for (j, b_kj) in b.row_entries(k) {
                coo.add(i, j, *a_ik * *b_kj);
            }
        }
    }

    coo.sort_by_row_column();
    CsrArray::from_coo(coo)
}

/// Sparse matrix-scalar multiplication.
///
/// Computes C = A * scalar where A is sparse (CSR).
pub fn csr_mul_scalar<T: NumericElement>(csr: &CsrArray<T>, scalar: T) -> CsrArray<T> {
    let coo = csr.to_coo();
    let mut scaled = crate::infrastructure::sparse::coo::CooArray::with_capacity(
        csr.nrows(),
        csr.ncols(),
        csr.nnz(),
    );

    for (row, col, value) in coo.entries() {
        scaled.add(row, col, *value * scalar);
    }

    CsrArray::from_coo(scaled)
}

/// Sparse matrix addition with scalar (broadcast).
///
/// Computes C = A + scalar where A is sparse (CSR).
/// This adds the scalar to all non-zero entries only.
pub fn csr_add_scalar<T: NumericElement>(csr: &CsrArray<T>, scalar: T) -> CsrArray<T> {
    let coo = csr.to_coo();
    let mut result = crate::infrastructure::sparse::coo::CooArray::with_capacity(
        csr.nrows(),
        csr.ncols(),
        csr.nnz(),
    );

    for (row, col, value) in coo.entries() {
        result.add(row, col, *value + scalar);
    }

    CsrArray::from_coo(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::sparse::coo::CooArray;

    #[test]
    fn test_csr_dense_matvec() {
        let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 1, 3.0)];
        let mut coo = CooArray::from_triplets(2, 2, triplets);
        coo.sort_by_row_column();

        let csr = CsrArray::from_coo(coo);
        let x = vec![1.0, 1.0];

        let y = csr_dense_matvec(&csr, &x);
        // y[0] = 1*1 + 2*1 = 3
        // y[1] = 0*1 + 3*1 = 3
        assert_eq!(y[0], 3.0);
        assert_eq!(y[1], 3.0);
    }

    #[test]
    fn test_csr_mul_scalar() {
        let triplets = vec![(0, 0, 1.0), (1, 1, 2.0)];
        let mut coo = CooArray::from_triplets(2, 2, triplets);
        coo.sort_by_row_column();

        let csr = CsrArray::from_coo(coo);
        let scaled = csr_mul_scalar(&csr, 2.0);

        assert_eq!(scaled.get(0, 0), Some(2.0));
        assert_eq!(scaled.get(1, 1), Some(4.0));
    }

    #[test]
    fn test_csr_add_csr() {
        let triplets_a = vec![(0, 0, 1.0), (1, 1, 2.0)];
        let mut coo_a = CooArray::from_triplets(2, 2, triplets_a);
        coo_a.sort_by_row_column();
        let csr_a = CsrArray::from_coo(coo_a);

        let triplets_b = vec![(0, 0, 3.0), (0, 1, 1.0)];
        let mut coo_b = CooArray::from_triplets(2, 2, triplets_b);
        coo_b.sort_by_row_column();
        let csr_b = CsrArray::from_coo(coo_b);

        let result = csr_add_csr(&csr_a, &csr_b);

        assert_eq!(result.get(0, 0), Some(4.0)); // 1 + 3
        assert_eq!(result.get(0, 1), Some(1.0)); // 0 + 1
        assert_eq!(result.get(1, 1), Some(2.0)); // 2 + 0
    }
}
