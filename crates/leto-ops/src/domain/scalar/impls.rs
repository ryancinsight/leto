//! `Scalar` implementations: SIMD-backed for the floating-point types, plain
//! for the integers, and the complex lift.

use crate::domain::strategy::{SimdOperations, SimdStrategy};
use eunomia::{Bf16, CastFrom, Complex, NumericElement, F16};

use super::fallback::{
    scalar_axpy_rows_batch_fallback, scalar_axpy_rows_fallback, scalar_gemv_strided_fallback,
    scalar_gemv_transpose_strided_fallback, scalar_tiled_gemm_fallback,
};
use super::Scalar;

/// Routes every slice operation through [`SimdStrategy`] first, falling back
/// to the scalar loop when the strategy declines. `$from_usize` is the
/// type's index conversion: a primitive cast for the machine floats, the
/// widening constructor for the reduced-precision types.
macro_rules! impl_scalar_simd {
    ($t:ty, $from_usize:expr) => {
        impl Scalar for $t {
            #[inline(always)]
            fn from_usize(value: usize) -> Self {
                ($from_usize)(value)
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

impl_scalar_simd!(f32, |value: usize| value as f32);
impl_scalar_simd!(f64, |value: usize| value as f64);
// hermes serves every routed operation at F16 (elementwise, reductions,
// axpy, gemv, tiled GEMM), so the half-precision type takes the same path as
// the machine floats; Bf16 waits on its provider kernels
// (hermes HS-REDUCED-PRECISION-ELEMENTWISE-2026-09-01).
impl_scalar_simd!(F16, |value: usize| F16::from_f32(value as f32));

impl Scalar for Bf16 {
    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        Self::from_f32(value as f32)
    }
}

/// Complex scalars participate in the operation contract through the same
/// element-wise defaults as the plain real and integer types.
///
/// The SIMD specializations are deliberately not taken: Hermes lanes are
/// real-valued, so a complex slice operation lowers to the interleaved
/// re/im arithmetic the default methods already express. Admitting complex
/// here is what lets the canonical containers — `CsrMatrix<T>`,
/// `CooMatrix<T>`, and the dense arrays — serve frequency-domain consumers
/// (boundary-element Helmholtz operators, spectral kernels) without a
/// parallel complex-only container.
///
/// Ordering-dependent surfaces stay real by construction: they are bound on
/// [`RealScalar`](crate::domain::real::RealScalar), which complex does not
/// implement because the complex field admits no total order.
impl<T> Scalar for Complex<T>
where
    T: Scalar + CastFrom<i32> + core::ops::Neg<Output = T>,
{
    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        Self::new(
            <T as Scalar>::from_usize(value),
            <T as NumericElement>::ZERO,
        )
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
