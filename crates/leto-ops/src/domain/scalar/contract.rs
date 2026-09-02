//! The `Scalar` operation contract over eunomia's numeric element.

use eunomia::NumericElement;

use super::fallback::{
    scalar_axpy_rows_batch_fallback, scalar_axpy_rows_fallback, scalar_gemv_strided_fallback,
    scalar_gemv_transpose_strided_fallback, scalar_tiled_gemm_fallback,
};

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
