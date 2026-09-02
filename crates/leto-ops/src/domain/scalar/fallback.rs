//! Scalar fallback kernels behind the SIMD hooks: the reference path every
//! strategy must match, and the path a type without a SIMD lane set takes.

use eunomia::NumericElement;

use super::Scalar;

#[inline]
pub(super) fn scalar_axpy_rows_fallback<T: Scalar>(
    alphas: &[T],
    x: &[T],
    out: &mut [T],
    row_stride: usize,
    rows: usize,
    cols: usize,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    let Some(last_row_offset) = rows
        .checked_sub(1)
        .and_then(|row| row.checked_mul(row_stride))
    else {
        return;
    };
    let Some(required_out_len) = last_row_offset.checked_add(cols) else {
        return;
    };
    if alphas.len() < rows || x.len() < cols || row_stride < cols || out.len() < required_out_len {
        return;
    }

    for (row, alpha) in alphas.iter().copied().take(rows).enumerate() {
        let start = row * row_stride;
        T::axpy_slice(alpha, x, &mut out[start..start + cols]);
    }
}

#[inline]
pub(super) fn scalar_axpy_rows_batch_fallback<T: Scalar>(
    alphas: &[T],
    x_panel: &[T],
    out: &mut [T],
    row_stride: usize,
    rows: usize,
    depth: usize,
    cols: usize,
) {
    if rows == 0 || depth == 0 || cols == 0 {
        return;
    }
    let Some(alpha_len) = rows.checked_mul(depth) else {
        return;
    };
    let Some(panel_len) = depth.checked_mul(cols) else {
        return;
    };
    if alphas.len() < alpha_len || x_panel.len() < panel_len {
        return;
    }

    for shared in 0..depth {
        let alpha_start = shared * rows;
        let x_start = shared * cols;
        scalar_axpy_rows_fallback(
            &alphas[alpha_start..alpha_start + rows],
            &x_panel[x_start..x_start + cols],
            out,
            row_stride,
            rows,
            cols,
        );
    }
}

/// Scalar fallback for [`Scalar::gemv_strided`]: `y[r] += sum_c a[r*lda + c]*x[c]`
/// over the `nrows x ncols` row-major sub-matrix (`lda >= ncols`).
#[inline]
pub(super) fn scalar_gemv_strided_fallback<T: Scalar>(
    a: &[T],
    x: &[T],
    y: &mut [T],
    nrows: usize,
    ncols: usize,
    lda: usize,
) {
    if lda < ncols || x.len() < ncols || y.len() < nrows {
        return;
    }
    let a_needed = if nrows == 0 {
        0
    } else {
        (nrows - 1) * lda + ncols
    };
    if a.len() < a_needed {
        return;
    }
    for (r, yr) in y.iter_mut().enumerate().take(nrows) {
        let row = &a[r * lda..r * lda + ncols];
        let mut acc = <T as NumericElement>::ZERO;
        for (&av, &xv) in row.iter().zip(x.iter()) {
            acc = acc.add(av.mul(xv));
        }
        *yr = yr.add(acc);
    }
}

/// Scalar fallback for [`Scalar::gemv_transpose_strided`]:
/// `y[c] += sum_r a[r*lda + c]*x[r]` over the `nrows x ncols` row-major
/// sub-matrix (`lda >= ncols`).
#[inline]
pub(super) fn scalar_gemv_transpose_strided_fallback<T: Scalar>(
    a: &[T],
    x: &[T],
    y: &mut [T],
    nrows: usize,
    ncols: usize,
    lda: usize,
) {
    if lda < ncols || x.len() < nrows || y.len() < ncols {
        return;
    }
    let a_needed = if nrows == 0 {
        0
    } else {
        (nrows - 1) * lda + ncols
    };
    if a.len() < a_needed {
        return;
    }
    for (r, &xr) in x.iter().enumerate().take(nrows) {
        let row = &a[r * lda..r * lda + ncols];
        for (yc, &av) in y.iter_mut().zip(row.iter()) {
            *yc = yc.add(av.mul(xr));
        }
    }
}

#[inline]
pub(super) fn scalar_tiled_gemm_fallback<T: Scalar>(
    a: &[T],
    b: &[T],
    c: &mut [T],
    m: usize,
    n: usize,
    k: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    if a.len() < m * k || b.len() < k * n || c.len() < m * n {
        return;
    }
    for r in 0..m {
        for kk in 0..k {
            let a_val = a[r * k + kk];
            if a_val == <T as NumericElement>::ZERO {
                continue;
            }
            for col in 0..n {
                c[r * n + col] = c[r * n + col].add(a_val.mul(b[kk * n + col]));
            }
        }
    }
}
