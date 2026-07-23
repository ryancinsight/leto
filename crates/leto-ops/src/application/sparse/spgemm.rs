//! Sparse matrix-matrix product `C = A · B` for CSR inputs.

use super::CsrMatrix;
use crate::domain::scalar::Scalar;
use leto::{LetoError, Result};
use std::collections::BTreeMap;

/// Compute `C = A · B` for two CSR matrices, returning CSR output.
///
/// Each output row accumulates the nonzeros reachable through row `i` of `A`
/// into a sorted map keyed by output column, then writes the nonzero entries in
/// ascending column order. Exact-zero accumulated entries are omitted, so
/// cancellation preserves the CSR invariant that only stored nonzeros remain.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] if `A.ncols() != B.nrows()`, or
/// [`LetoError::StorageError`] if the constructed CSR structure is invalid.
#[must_use = "spgemm computes a sparse matrix product"]
pub fn spgemm<T: Scalar>(a: &CsrMatrix<T>, b: &CsrMatrix<T>) -> Result<CsrMatrix<T>> {
    let (a_rows, a_cols) = a.shape();
    let (b_rows, b_cols) = b.shape();
    if a_cols != b_rows {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![a_rows, a_cols],
            rhs: vec![b_rows, b_cols],
        });
    }

    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptr = Vec::with_capacity(a_rows + 1);
    row_ptr.push(0);

    let (a_values, a_col_indices, a_row_ptr) = a.as_parts();
    let (b_values, b_col_indices, b_row_ptr) = b.as_parts();
    let mut row_accumulator = BTreeMap::<usize, T>::new();

    for row in 0..a_rows {
        row_accumulator.clear();
        for a_index in a_row_ptr[row]..a_row_ptr[row + 1] {
            let inner = a_col_indices[a_index];
            let a_value = a_values[a_index];
            for b_index in b_row_ptr[inner]..b_row_ptr[inner + 1] {
                let product = a_value * b_values[b_index];
                if product == T::ZERO {
                    continue;
                }
                row_accumulator
                    .entry(b_col_indices[b_index])
                    .and_modify(|entry| *entry += product)
                    .or_insert(product);
            }
        }

        for (&col, &value) in &row_accumulator {
            if value != T::ZERO {
                col_indices.push(col);
                values.push(value);
            }
        }
        row_ptr.push(values.len());
    }

    CsrMatrix::from_parts(values, col_indices, row_ptr, a_rows, b_cols)
}
