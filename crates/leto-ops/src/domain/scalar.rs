use crate::domain::strategy::{SimdOperations, SimdStrategy};
use eunomia::NumericElement;
use half::{bf16, f16};

/// Leto operation scalar contract.
///
/// Eunomia owns the foundational numeric element contract: constants, primitive
/// arithmetic traits, bit operations, finite/NaN predicates, and representation
/// metadata. `Scalar` is the Leto operation extension over that SSOT. It keeps
/// the slice-level CPU/SIMD hooks used by array kernels and the construction
/// helper for dimension-derived scalar values.
pub trait Scalar: NumericElement {
    /// Construct a scalar from a non-negative element count.
    fn from_usize(value: usize) -> Self;

    /// Element-wise slice addition: `out = a + b`.
    #[inline]
    fn add_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
        assert_eq!(a.len(), b.len(), "add_slice: a.len() != b.len()");
        assert_eq!(a.len(), out.len(), "add_slice: output length mismatch");
        for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = x + y;
        }
    }

    /// Element-wise slice subtraction: `out = a - b`.
    #[inline]
    fn sub_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
        assert_eq!(a.len(), b.len(), "sub_slice: a.len() != b.len()");
        assert_eq!(a.len(), out.len(), "sub_slice: output length mismatch");
        for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = x - y;
        }
    }

    /// Element-wise slice multiplication: `out = a * b`.
    #[inline]
    fn mul_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
        assert_eq!(a.len(), b.len(), "mul_slice: a.len() != b.len()");
        assert_eq!(a.len(), out.len(), "mul_slice: output length mismatch");
        for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = x * y;
        }
    }

    /// Element-wise slice division: `out = a / b`.
    #[inline]
    fn div_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
        assert_eq!(a.len(), b.len(), "div_slice: a.len() != b.len()");
        assert_eq!(a.len(), out.len(), "div_slice: output length mismatch");
        for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *o = x / y;
        }
    }

    /// Sum reduction over a slice.
    #[inline]
    fn sum_slice(s: &[Self]) -> Self {
        s.iter()
            .copied()
            .fold(<Self as NumericElement>::ZERO, |acc, x| acc + x)
    }

    /// Dot product reduction over two equal-length slices.
    #[inline]
    fn dot_slice(a: &[Self], b: &[Self]) -> Self {
        assert_eq!(a.len(), b.len(), "dot_slice: a.len() != b.len()");
        a.iter()
            .copied()
            .zip(b.iter().copied())
            .fold(<Self as NumericElement>::ZERO, |acc, (x, y)| acc + x * y)
    }

    /// Fused row update over equal-length slices: `out[i] += alpha * x[i]`.
    #[inline]
    fn axpy_slice(alpha: Self, x: &[Self], out: &mut [Self]) {
        assert_eq!(x.len(), out.len(), "axpy_slice: x.len() != out.len()");
        for (o, &xv) in out.iter_mut().zip(x.iter()) {
            *o += alpha * xv;
        }
    }

    /// Fused multi-row update: `out[row, i] += alphas[row] * x[i]`.
    #[inline]
    fn axpy_rows(
        alphas: &[Self],
        x: &[Self],
        out: &mut [Self],
        row_stride: usize,
        rows: usize,
        cols: usize,
    ) {
        scalar_axpy_rows_fallback(alphas, x, out, row_stride, rows, cols);
    }

    /// Fused batched multi-row update:
    /// `out[row, i] += sum_k alphas[k, row] * x_panel[k, i]`.
    #[inline]
    fn axpy_rows_batch(
        alphas: &[Self],
        x_panel: &[Self],
        out: &mut [Self],
        row_stride: usize,
        rows: usize,
        depth: usize,
        cols: usize,
    ) {
        scalar_axpy_rows_batch_fallback(alphas, x_panel, out, row_stride, rows, depth, cols);
    }

    /// Register-blocked sub-matrix GEMV `y += A*x`.
    ///
    /// `A` is row-major `nrows x ncols` with row stride `lda >= ncols`.
    /// Accumulates into `y`; zero it first for `y = A*x`.
    #[inline]
    fn gemv_strided(
        a: &[Self],
        x: &[Self],
        y: &mut [Self],
        nrows: usize,
        ncols: usize,
        lda: usize,
    ) {
        scalar_gemv_strided_fallback(a, x, y, nrows, ncols, lda);
    }

    /// Register-blocked transposed sub-matrix GEMV `y += A^T*x`.
    ///
    /// `A` is row-major `nrows x ncols` with row stride `lda >= ncols`.
    #[inline]
    fn gemv_transpose_strided(
        a: &[Self],
        x: &[Self],
        y: &mut [Self],
        nrows: usize,
        ncols: usize,
        lda: usize,
    ) {
        scalar_gemv_transpose_strided_fallback(a, x, y, nrows, ncols, lda);
    }

    /// Register-blocked tiled GEMM: `c += A * B`.
    #[inline]
    fn tiled_gemm(a: &[Self], b: &[Self], c: &mut [Self], m: usize, n: usize, k: usize) {
        scalar_tiled_gemm_fallback(a, b, c, m, n, k);
    }

    /// Min reduction over a slice.
    #[inline]
    fn min_slice(s: &[Self]) -> Self {
        s.iter()
            .copied()
            .fold(<Self as NumericElement>::MAX_VALUE, |acc, x| {
                if x < acc {
                    x
                } else {
                    acc
                }
            })
    }

    /// Max reduction over a slice.
    #[inline]
    fn max_slice(s: &[Self]) -> Self {
        s.iter()
            .copied()
            .fold(<Self as NumericElement>::MIN_VALUE, |acc, x| {
                if x > acc {
                    x
                } else {
                    acc
                }
            })
    }

    /// Jaccard distance between two slices.
    #[inline]
    fn jaccard_distance(_a: &[Self], _b: &[Self]) -> Option<f64> {
        None
    }

    /// Hamming distance between two slices.
    #[inline]
    fn hamming_distance(_a: &[Self], _b: &[Self]) -> Option<u64> {
        None
    }
}

macro_rules! impl_scalar_simd {
    ($t:ty) => {
        impl Scalar for $t {
            #[inline(always)]
            fn from_usize(value: usize) -> Self {
                value as $t
            }

            #[inline]
            fn add_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                assert_eq!(a.len(), b.len(), "add_slice: a.len() != b.len()");
                assert_eq!(a.len(), out.len(), "add_slice: output length mismatch");
                if <SimdStrategy as SimdOperations<Self>>::add_slice(a, b, out).is_ok() {
                    return;
                }
                for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                    *o = x + y;
                }
            }

            #[inline]
            fn sub_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                assert_eq!(a.len(), b.len(), "sub_slice: a.len() != b.len()");
                assert_eq!(a.len(), out.len(), "sub_slice: output length mismatch");
                if <SimdStrategy as SimdOperations<Self>>::sub_slice(a, b, out).is_ok() {
                    return;
                }
                for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                    *o = x - y;
                }
            }

            #[inline]
            fn mul_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                assert_eq!(a.len(), b.len(), "mul_slice: a.len() != b.len()");
                assert_eq!(a.len(), out.len(), "mul_slice: output length mismatch");
                if <SimdStrategy as SimdOperations<Self>>::mul_slice(a, b, out).is_ok() {
                    return;
                }
                for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                    *o = x * y;
                }
            }

            #[inline]
            fn div_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                assert_eq!(a.len(), b.len(), "div_slice: a.len() != b.len()");
                assert_eq!(a.len(), out.len(), "div_slice: output length mismatch");
                if <SimdStrategy as SimdOperations<Self>>::div_slice(a, b, out).is_ok() {
                    return;
                }
                for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                    *o = x / y;
                }
            }

            #[inline]
            fn sum_slice(s: &[Self]) -> Self {
                if let Some(res) = <SimdStrategy as SimdOperations<Self>>::sum_slice(s) {
                    res
                } else {
                    s.iter()
                        .copied()
                        .fold(<Self as NumericElement>::ZERO, |acc, x| acc + x)
                }
            }

            #[inline]
            fn dot_slice(a: &[Self], b: &[Self]) -> Self {
                assert_eq!(a.len(), b.len(), "dot_slice: a.len() != b.len()");
                if let Some(res) = <SimdStrategy as SimdOperations<Self>>::dot_slice(a, b) {
                    res
                } else {
                    a.iter()
                        .copied()
                        .zip(b.iter().copied())
                        .fold(<Self as NumericElement>::ZERO, |acc, (x, y)| acc + x * y)
                }
            }

            #[inline]
            fn axpy_slice(alpha: Self, x: &[Self], out: &mut [Self]) {
                assert_eq!(x.len(), out.len(), "axpy_slice: x.len() != out.len()");
                if <SimdStrategy as SimdOperations<Self>>::axpy_slice(alpha, x, out).is_ok() {
                    return;
                }
                for (o, &xv) in out.iter_mut().zip(x.iter()) {
                    *o += alpha * xv;
                }
            }

            #[inline]
            fn axpy_rows(
                alphas: &[Self],
                x: &[Self],
                out: &mut [Self],
                row_stride: usize,
                rows: usize,
                cols: usize,
            ) {
                if <SimdStrategy as SimdOperations<Self>>::axpy_rows(
                    alphas, x, out, row_stride, rows, cols,
                )
                .is_ok()
                {
                    return;
                }
                scalar_axpy_rows_fallback(alphas, x, out, row_stride, rows, cols);
            }

            #[inline]
            fn axpy_rows_batch(
                alphas: &[Self],
                x_panel: &[Self],
                out: &mut [Self],
                row_stride: usize,
                rows: usize,
                depth: usize,
                cols: usize,
            ) {
                if <SimdStrategy as SimdOperations<Self>>::axpy_rows_batch(
                    alphas, x_panel, out, row_stride, rows, depth, cols,
                )
                .is_ok()
                {
                    return;
                }
                scalar_axpy_rows_batch_fallback(
                    alphas, x_panel, out, row_stride, rows, depth, cols,
                );
            }

            #[inline]
            fn tiled_gemm(a: &[Self], b: &[Self], c: &mut [Self], m: usize, n: usize, k: usize) {
                if <SimdStrategy as SimdOperations<Self>>::tiled_gemm(a, b, c, m, n, k).is_ok() {
                    return;
                }
                scalar_tiled_gemm_fallback(a, b, c, m, n, k);
            }

            #[inline]
            fn gemv_strided(
                a: &[Self],
                x: &[Self],
                y: &mut [Self],
                nrows: usize,
                ncols: usize,
                lda: usize,
            ) {
                if <SimdStrategy as SimdOperations<Self>>::gemv_strided(a, x, y, nrows, ncols, lda)
                    .is_ok()
                {
                    return;
                }
                scalar_gemv_strided_fallback(a, x, y, nrows, ncols, lda);
            }

            #[inline]
            fn gemv_transpose_strided(
                a: &[Self],
                x: &[Self],
                y: &mut [Self],
                nrows: usize,
                ncols: usize,
                lda: usize,
            ) {
                if <SimdStrategy as SimdOperations<Self>>::gemv_transpose_strided(
                    a, x, y, nrows, ncols, lda,
                )
                .is_ok()
                {
                    return;
                }
                scalar_gemv_transpose_strided_fallback(a, x, y, nrows, ncols, lda);
            }

            #[inline]
            fn min_slice(s: &[Self]) -> Self {
                if let Some(res) = <SimdStrategy as SimdOperations<Self>>::min_slice(s) {
                    res
                } else {
                    s.iter()
                        .copied()
                        .fold(<Self as NumericElement>::MAX_VALUE, |acc, x| {
                            if x < acc {
                                x
                            } else {
                                acc
                            }
                        })
                }
            }

            #[inline]
            fn max_slice(s: &[Self]) -> Self {
                if let Some(res) = <SimdStrategy as SimdOperations<Self>>::max_slice(s) {
                    res
                } else {
                    s.iter()
                        .copied()
                        .fold(<Self as NumericElement>::MIN_VALUE, |acc, x| {
                            if x > acc {
                                x
                            } else {
                                acc
                            }
                        })
                }
            }

            #[inline]
            fn jaccard_distance(a: &[Self], b: &[Self]) -> Option<f64> {
                <SimdStrategy as SimdOperations<Self>>::jaccard_distance(a, b)
            }

            #[inline]
            fn hamming_distance(a: &[Self], b: &[Self]) -> Option<u64> {
                <SimdStrategy as SimdOperations<Self>>::hamming_distance(a, b)
            }
        }
    };
}

macro_rules! impl_scalar_plain {
    ($t:ty) => {
        impl Scalar for $t {
            #[inline(always)]
            fn from_usize(value: usize) -> Self {
                value as $t
            }
        }
    };
}

impl_scalar_simd!(f32);
impl_scalar_simd!(f64);

impl Scalar for f16 {
    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        Self::from_f32(value as f32)
    }
}

impl Scalar for bf16 {
    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        Self::from_f32(value as f32)
    }
}

impl_scalar_plain!(i8);
impl_scalar_plain!(u8);
impl_scalar_plain!(i16);
impl_scalar_plain!(u16);
impl_scalar_plain!(i32);
impl_scalar_plain!(u32);
impl_scalar_plain!(i64);
impl_scalar_plain!(u64);
impl_scalar_plain!(isize);
impl_scalar_plain!(usize);

#[inline]
fn scalar_axpy_rows_fallback<T: Scalar>(
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
fn scalar_axpy_rows_batch_fallback<T: Scalar>(
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
fn scalar_gemv_strided_fallback<T: Scalar>(
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
fn scalar_gemv_transpose_strided_fallback<T: Scalar>(
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
fn scalar_tiled_gemm_fallback<T: Scalar>(
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
