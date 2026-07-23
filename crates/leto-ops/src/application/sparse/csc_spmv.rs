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

    y.fill(T::ZERO);

    let (values, row_indices, col_ptr) = a.as_parts();

    // Zip each column's `x` value with its `col_ptr` window, then slice the
    // column's row-index/value runs so the per-nonzero `values[p]`/
    // `row_indices[p]` bounds checks collapse into one `O(ncols)` slice check
    // per column. The residual `y[i]` scatter check is data-dependent and stays.
    if let Some(xs) = x.as_slice() {
        for (&xj, window) in xs.iter().zip(col_ptr.windows(2)) {
            if xj == T::ZERO {
                continue;
            }
            let rows = &row_indices[window[0]..window[1]];
            let vals = &values[window[0]..window[1]];
            for (&i, &value) in rows.iter().zip(vals) {
                y[i] = y[i].add(value.mul(xj));
            }
        }
    } else {
        for (j, window) in col_ptr.windows(2).enumerate() {
            let xj = *x.get([j])?;
            if xj == T::ZERO {
                continue;
            }
            let rows = &row_indices[window[0]..window[1]];
            let vals = &values[window[0]..window[1]];
            for (&i, &value) in rows.iter().zip(vals) {
                y[i] = y[i].add(value.mul(xj));
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
