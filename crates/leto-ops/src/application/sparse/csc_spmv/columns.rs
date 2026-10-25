//! The CSC SpMV product entries, the column walk and the scatter-add kernels.
//!
//! CSC SpMV is gather-free: a column's `x` value is one scalar reused across
//! the column, and each stored entry scatters into `y`. One walk serves the
//! serial and the strided entry; [`scatter_column_window`] is the row-range form
//! the parallel driver hands each task, so both paths scatter with the same
//! `y[i] = y[i] + value * xj` update and the same rounding.

#[cfg(feature = "parallel")]
use super::parallel;
use crate::application::sparse::{CscMatrix, CscView};
use crate::domain::scalar::Scalar;
use leto::{Array1, ArrayView1, LetoError, Result};

/// Compute `y = A · x` into the caller-owned slice `y` (length `nrows`),
/// overwriting it. One pass over the stored nonzeros (`O(nnz)`).
/// Borrowed-CSC SSOT: [`csc_spmv_into`] delegates here through
/// [`CscMatrix::as_view`], so owned and borrowed matrices share one kernel
/// with zero copies either way.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] if `x` is not length `ncols` or `y` is not
/// length `nrows`.
pub fn csc_spmv_view_into<T: Scalar>(
    a: &CscView<T>,
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
    match x.as_slice() {
        Some(xs) => csc_spmv_slice_into(values, row_indices, col_ptr, xs, y),
        None => csc_column_strided_into(values, row_indices, col_ptr, x, y)?,
    }

    Ok(())
}

/// Dispatch one contiguous-`x` column pass: the parallel row-range driver past
/// its gate, the serial column walk otherwise. Both write every `y` slot
/// exactly once and accumulate each slot's columns in ascending order, so the
/// choice changes only which thread runs the walk.
fn csc_spmv_slice_into<T: Scalar>(
    values: &[T],
    row_indices: &[usize],
    col_ptr: &[usize],
    xs: &[T],
    y: &mut [T],
) {
    #[cfg(feature = "parallel")]
    if parallel::should_parallelize(values.len(), col_ptr.len().saturating_sub(1)) {
        parallel::csc_spmv_rows_parallel(values, row_indices, col_ptr, xs, y);
        return;
    }
    csc_column_run_into(values, row_indices, col_ptr, xs, y);
}

/// Compute `y = A · x` over an owned matrix. Thin wrapper over
/// [`csc_spmv_view_into`] (SSOT) through [`CscMatrix::as_view`] (zero-copy).
///
/// # Errors
/// [`LetoError::ShapeMismatch`] if `x` is not length `ncols` or `y` is not
/// length `nrows`.
pub fn csc_spmv_into<T: Scalar>(
    a: &CscMatrix<T>,
    x: &ArrayView1<'_, T>,
    y: &mut [T],
) -> Result<()> {
    csc_spmv_view_into(&a.as_view(), x, y)
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

/// Scatter-add every column's contributions, reading the contiguous `xs`.
pub(super) fn csc_column_run_into<T: Scalar>(
    values: &[T],
    row_indices: &[usize],
    col_ptr: &[usize],
    xs: &[T],
    y: &mut [T],
) {
    // Zip each column's `x` value with its `col_ptr` window, then slice the
    // column's row-index/value runs so the per-nonzero `values[p]`/
    // `row_indices[p]` bounds checks collapse into one `O(ncols)` slice check
    // per column. The residual `y[i]` scatter check is data-dependent and stays.
    for (&xj, window) in xs.iter().zip(col_ptr.windows(2)) {
        scatter_column(values, row_indices, window[0], window[1], xj, y);
    }
}

/// Scatter-add every column's contributions, reading `x` through its view.
pub(super) fn csc_column_strided_into<T: Scalar>(
    values: &[T],
    row_indices: &[usize],
    col_ptr: &[usize],
    x: &ArrayView1<'_, T>,
    y: &mut [T],
) -> Result<()> {
    for (j, window) in col_ptr.windows(2).enumerate() {
        scatter_column(values, row_indices, window[0], window[1], *x.get([j])?, y);
    }
    Ok(())
}

/// `y[i] += value * xj` over one column's `[start, end)` nonzero run.
#[inline]
pub(super) fn scatter_column<T: Scalar>(
    values: &[T],
    row_indices: &[usize],
    start: usize,
    end: usize,
    xj: T,
    y: &mut [T],
) {
    if xj == T::ZERO {
        return;
    }
    let rows = &row_indices[start..end];
    let vals = &values[start..end];
    for (&i, &value) in rows.iter().zip(vals) {
        y[i] = y[i].add(value.mul(xj));
    }
}

/// `y[i - base] += value * xj` over the entries of one column whose row falls in
/// `[base, base + y.len())`.
///
/// This is [`scatter_column`] restricted to one task's rows: rows ascend within
/// a column (a [`CscView`](super::CscView) invariant), so the entries outside
/// the window are one contiguous prefix and one contiguous suffix, found with a
/// pair of binary searches. It stays separate rather than a `base = 0` call of
/// the serial kernel because the serial walk would then pay those searches per
/// column for nothing. Both fold the window's entries in ascending row order, so
/// a column contributes to `y` in the same order either way.
#[inline]
pub(super) fn scatter_column_window<T: Scalar>(
    values: &[T],
    row_indices: &[usize],
    start: usize,
    end: usize,
    xj: T,
    base: usize,
    y: &mut [T],
) {
    if xj == T::ZERO {
        return;
    }
    let rows = &row_indices[start..end];
    if rows.is_empty() {
        return;
    }
    // The window is `[base, base + y.len())`. Most columns miss most windows,
    // and the first and last stored rows answer that in two comparisons: in
    // column-major order those entries of neighbouring columns sit close
    // together, so the rejection stays in cache.
    let end_of_window = base + y.len();
    if rows[rows.len() - 1] < base || rows[0] >= end_of_window {
        return;
    }
    let vals = &values[start..end];
    let first = rows.partition_point(|&i| i < base);
    let past = first + rows[first..].partition_point(|&i| i < end_of_window);
    for (&i, &value) in rows[first..past].iter().zip(&vals[first..past]) {
        let slot = i - base;
        y[slot] = y[slot].add(value.mul(xj));
    }
}
