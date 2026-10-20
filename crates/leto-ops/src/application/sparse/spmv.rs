//! Sparse matrix–vector product `y = A · x` over CSR, in `O(nnz)` time.

use super::{CsrMatrix, CsrView};
use crate::domain::scalar::Scalar;
use leto::{Array1, ArrayView1, LetoError, Result, Storage};

/// Compute `y = A · x` into the caller-owned slice `y` (length `nrows`),
/// overwriting it. One pass over the stored nonzeros (`O(nnz)`); see the module
/// theorem for correctness and complexity. Borrowed-CSR SSOT: [`spmv_into`]
/// delegates here through [`CsrMatrix::as_view`], so owned and borrowed
/// matrices share one kernel with zero copies either way.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] if `x` is not length `ncols` or `y` is not
/// length `nrows`.
pub fn spmv_view_into<T: Scalar>(a: &CsrView<T>, x: &ArrayView1<'_, T>, y: &mut [T]) -> Result<()> {
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

    if let Some(xs) = x.as_slice() {
        return spmv_slice_into(a, xs, y);
    }

    // Materialize non-contiguous x once: the gather `x[col_indices[p]]` then
    // reads a plain slice instead of performing a strided lookup per nonzero.
    let x_contiguous = x.to_contiguous();
    spmv_slice_into(a, x_contiguous.storage().as_slice(), y)
}

/// Compute `y = A · x` over an owned matrix. Thin wrapper over
/// [`spmv_view_into`] (SSOT) through [`CsrMatrix::as_view`] (zero-copy).
///
/// # Errors
/// [`LetoError::ShapeMismatch`] if `x` is not length `ncols` or `y` is not
/// length `nrows`.
pub fn spmv_into<T: Scalar>(a: &CsrMatrix<T>, x: &ArrayView1<'_, T>, y: &mut [T]) -> Result<()> {
    spmv_view_into(&a.as_view(), x, y)
}

fn spmv_slice_into<T: Scalar>(a: &CsrView<T>, xs: &[T], y: &mut [T]) -> Result<()> {
    let (values, col_indices, row_ptr) = a.as_parts();

    // Iterate rows through `row_ptr.windows(2)` zipped with `y` — eliding the
    // `y[i]`/`row_ptr[i]`/`row_ptr[i+1]` bounds checks — and slice each row's
    // value/column runs so the per-nonzero `values[p]`/`col_indices[p]` checks
    // fall away too (one `O(nrows)` slice check replaces `O(nnz)` element
    // checks). The residual `xs[col]` check is data-dependent; the CSR
    // invariant (`col_indices[p] < ncols == xs.len()`, from `from_parts`) proves
    // it in-bounds but the compiler cannot, so it stays for safety.
    for (slot, window) in y.iter_mut().zip(row_ptr.windows(2)) {
        let (start, end) = (window[0], window[1]);
        let cols = &col_indices[start..end];
        let vals = &values[start..end];
        *slot = gather_dot(vals, cols, xs);
    }
    Ok(())
}

/// Gather-dot `sum(vals[p] * xs[cols[p]])` over one CSR row.
///
/// Four independent accumulators over stride-4 lanes keep several
/// multiply-add chains and gather loads in flight: the single-accumulator
/// form serializes each nonzero behind the previous add's latency and issues
/// one outstanding gather at a time. The reduction order
/// `(acc0 + acc1) + (acc2 + acc3)` with the sub-4 tail folded into `acc0` is
/// fixed for all row lengths. A finite result keeps this native rounding
/// order unchanged. If it is nonfinite, retry the original stored-order
/// fold: parallel lanes can overflow before opposite signs cancel, while
/// that fold stays finite. This does not guarantee a finite result whenever
/// the exact real sum is representable; the stored fold may overflow too.
///
/// Different finite orders can lose different small terms under cancellation.
/// Integer-valued inputs alone do not imply bitwise equality: products and
/// every partial sum must also be exactly representable. No compensated sum,
/// widened accumulator or universal correctly-rounded result is promised.
fn gather_dot<T: Scalar>(vals: &[T], cols: &[usize], xs: &[T]) -> T {
    let mut acc0 = T::ZERO;
    let mut acc1 = T::ZERO;
    let mut acc2 = T::ZERO;
    let mut acc3 = T::ZERO;
    // CsrView validation gives equal value/column lengths. Fixed array
    // groups express the four-element bounds directly without changing any
    // per-lane update, zero addition, tail update or final reduction order.
    debug_assert_eq!(vals.len(), cols.len());
    let (value_groups, value_tail) = vals.as_chunks::<4>();
    let (column_groups, column_tail) = cols.as_chunks::<4>();
    for (values, columns) in value_groups.iter().zip(column_groups) {
        acc0 = acc0.add(values[0].mul(xs[columns[0]]));
        acc1 = acc1.add(values[1].mul(xs[columns[1]]));
        acc2 = acc2.add(values[2].mul(xs[columns[2]]));
        acc3 = acc3.add(values[3].mul(xs[columns[3]]));
    }
    for (&value, &column) in value_tail.iter().zip(column_tail) {
        acc0 = acc0.add(value.mul(xs[column]));
    }
    let result = acc0.add(acc1).add(acc2.add(acc3));
    if result.is_finite() {
        result
    } else {
        // Reassociation can overflow lanes even when the stored-order fold
        // stays finite. Retry that original native order without allocation;
        // genuinely nonfinite inputs/results retain its nonfinite outcome.
        vals.iter()
            .zip(cols)
            .fold(T::ZERO, |acc, (&value, &col)| acc.add(value.mul(xs[col])))
    }
}

/// Compute `y = A · x`, allocating the length-`nrows` result. Thin wrapper over
/// [`spmv_into`] (SSOT).
///
/// # Errors
/// [`LetoError::ShapeMismatch`] if `x` is not length `ncols`.
pub fn spmv<T: Scalar>(a: &CsrMatrix<T>, x: &ArrayView1<'_, T>) -> Result<Array1<T>> {
    let (nrows, _) = a.shape();
    let mut y = vec![T::ZERO; nrows];
    spmv_into(a, x, &mut y)?;
    Array1::from_shape_vec([nrows], y)
}
