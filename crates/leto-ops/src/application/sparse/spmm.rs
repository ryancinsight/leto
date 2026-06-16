//! Sparse–dense matrix product `C = A · B` (CSR `A`, dense `B`) in `O(nnz·k)`.

use super::CsrMatrix;
use crate::domain::scalar::Scalar;
use leto::{Array2, ArrayView2, LetoError, Result, Storage};

/// Compute `C = A · B` into the caller-owned row-major slice `c`
/// (`nrows × bcols`), overwriting it. `A` is `nrows × ncols` (CSR), `B` is
/// `ncols × bcols` (dense).
///
/// # Theorem (correctness and complexity)
/// `C[i, :] = Σ_j A[i,j] · B[j, :] = Σ_{p ∈ row i} values[p] · B[col_indices[p], :]`
/// by the CSR identity (module theorem). The loop accumulates exactly that:
/// for each stored nonzero it scales one row of `B` into one row of `C` (a fused
/// `axpy`), so each nonzero contributes `O(bcols)` work — `Θ(nnz·bcols + m·bcols)`
/// total, versus dense `Θ(m·n·bcols)`. The per-`C[i,t]` accumulation order is the
/// nonzero order within row `i`. ∎
///
/// The row-scale-accumulate is dispatched through [`Scalar::axpy_slice`], so it
/// inherits the SIMD path on contiguous rows (SSOT with the dense matmul kernel).
///
/// # Errors
/// [`LetoError::ShapeMismatch`] if `B`'s row count `≠ ncols`, or `c.len() ≠
/// nrows·bcols`.
pub fn spmm_into<T: Scalar>(a: &CsrMatrix<T>, b: &ArrayView2<'_, T>, c: &mut [T]) -> Result<()> {
    let (nrows, ncols) = a.shape();
    let [b_rows, bcols] = b.shape();
    if b_rows != ncols {
        return Err(LetoError::ShapeMismatch {
            lhs: b.shape().to_vec(),
            rhs: vec![ncols, bcols],
        });
    }
    let output_len = nrows
        .checked_mul(bcols)
        .ok_or_else(|| LetoError::StorageError {
            reason: "SpMM output length overflows usize".to_string(),
        })?;
    if c.len() != output_len {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![c.len()],
            rhs: vec![output_len],
        });
    }

    if let Some(bs) = b.as_slice() {
        return spmm_slice_into(a, bcols, bs, c);
    }

    // Materialize non-contiguous B once: each B-row `B[col, :]` is then a plain
    // contiguous slice the SIMD axpy can stream.
    let b_contiguous = b.to_contiguous();
    spmm_slice_into(a, bcols, b_contiguous.storage().as_slice(), c)
}

fn spmm_slice_into<T: Scalar>(a: &CsrMatrix<T>, bcols: usize, bs: &[T], c: &mut [T]) -> Result<()> {
    let (nrows, _) = a.shape();
    let (values, col_indices, row_ptr) = a.as_parts();

    c.fill(T::ZERO);
    for i in 0..nrows {
        let c_row = &mut c[i * bcols..i * bcols + bcols];
        for p in row_ptr[i]..row_ptr[i + 1] {
            let col = col_indices[p];
            let b_row = &bs[col * bcols..col * bcols + bcols];
            // c_row += values[p] · b_row  (fused, SIMD-dispatched).
            T::axpy_slice(values[p], b_row, c_row);
        }
    }
    Ok(())
}

/// Compute `C = A · B`, allocating the `nrows × bcols` result. Thin wrapper over
/// [`spmm_into`] (SSOT).
///
/// # Errors
/// [`LetoError::ShapeMismatch`] if `B`'s row count `≠ ncols`.
pub fn spmm<T: Scalar>(a: &CsrMatrix<T>, b: &ArrayView2<'_, T>) -> Result<Array2<T>> {
    let (nrows, _) = a.shape();
    let [_, bcols] = b.shape();
    let output_len = nrows
        .checked_mul(bcols)
        .ok_or_else(|| LetoError::StorageError {
            reason: "SpMM output length overflows usize".to_string(),
        })?;
    let mut c = vec![T::ZERO; output_len];
    spmm_into(a, b, &mut c)?;
    Array2::from_shape_vec([nrows, bcols], c)
}
