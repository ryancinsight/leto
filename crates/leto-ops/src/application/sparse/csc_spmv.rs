//! Sparse matrix–vector product `y = A · x` over CSC, in `O(nnz)` time.
//!
//! CSC SpMV accesses each nonzero exactly once: for each column `j`, every
//! stored entry `(i, value)` in that column contributes `value * x[j]` to
//! `y[i]`.  This is a gather-free scatter-add pattern (no indirection on `x`),
//! unlike CSR which needs `x[col_indices[p]]`.

use super::CscMatrix;
use crate::domain::scalar::Scalar;
use leto::{Array1, ArrayView1, LetoError, Result};

/// Compute `y = A · x` into the caller-owned slice `y` (length `nrows`),
/// overwriting it. One pass over the stored nonzeros (`O(nnz)`).
///
/// # Errors
/// [`LetoError::ShapeMismatch`] if `x` is not length `ncols` or `y` is not
/// length `nrows`.
pub fn csc_spmv_into<T: Scalar>(
    a: &CscMatrix<T>,
    x: &ArrayView1<'_, T>,
    y: &mut [T],
) -> Result<()> {
    let (nrows, ncols) = a.shape();
    if x.shape() != [ncols] {
        return Err(LetoError::ShapeMismatch {
            lhs: x.shape().to_vec(),
            rhs: vec![ncols],
        });
    }
    if y.len() != nrows {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![y.len()],
            rhs: vec![nrows],
        });
    }

    // Zero the output vector before accumulation.
    for slot in y.iter_mut() {
        *slot = T::ZERO;
    }

    let (values, row_indices, col_ptr) = a.as_parts();

    if let Some(xs) = x.as_slice() {
        for j in 0..ncols {
            let xj = xs[j];
            if xj == T::ZERO {
                continue;
            }
            for p in col_ptr[j]..col_ptr[j + 1] {
                y[row_indices[p]] = y[row_indices[p]].add(values[p].mul(xj));
            }
        }
    } else {
        for j in 0..ncols {
            let xj = *x.get([j])?;
            if xj == T::ZERO {
                continue;
            }
            for p in col_ptr[j]..col_ptr[j + 1] {
                y[row_indices[p]] = y[row_indices[p]].add(values[p].mul(xj));
            }
        }
    }

    Ok(())
}

/// Compute `y = A · x`, allocating the length-`nrows` result. Thin wrapper over
/// [`csc_spmv_into`] (SSOT).
///
/// # Errors
/// [`LetoError::ShapeMismatch`] if `x` is not length `ncols`.
pub fn csc_spmv<T: Scalar>(a: &CscMatrix<T>, x: &ArrayView1<'_, T>) -> Result<Array1<T>> {
    let (nrows, _) = a.shape();
    let mut y = vec![T::ZERO; nrows];
    csc_spmv_into(a, x, &mut y)?;
    Array1::from_shape_vec([nrows], y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csc_spmv_basic() {
        let a = CscMatrix::from_dense(
            &leto::Array2::from_shape_vec([3, 2], vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0])
                .unwrap()
                .view(),
        );
        let x = leto::Array1::from_shape_vec([2], vec![2.0, 5.0]).unwrap();
        let y = csc_spmv(&a, &x.view()).unwrap();
        assert_eq!(
            y,
            leto::Array1::from_shape_vec([3], vec![2.0, 10.0, 6.0]).unwrap()
        );
    }

    #[test]
    fn csc_spmv_identity() {
        let a = CscMatrix::from_dense(
            &leto::Array2::from_shape_vec([2, 2], vec![1.0, 0.0, 0.0, 1.0])
                .unwrap()
                .view(),
        );
        let x = leto::Array1::from_shape_vec([2], vec![3.0, 7.0]).unwrap();
        let y = csc_spmv(&a, &x.view()).unwrap();
        assert_eq!(
            y,
            leto::Array1::from_shape_vec([2], vec![3.0, 7.0]).unwrap()
        );
    }

    #[test]
    fn csc_spmv_shape_mismatch() {
        let a = CscMatrix::<f64>::zeros(3, 2);
        let x = leto::Array1::from_shape_vec([3], vec![1.0, 2.0, 3.0]).unwrap();
        assert!(csc_spmv(&a, &x.view()).is_err());
    }
}
