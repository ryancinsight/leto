use crate::domain::scalar::Scalar;
use eunomia::{Bf16, F16};

mod sealed {
    pub trait Sealed {}
}

/// Marker trait for operation execution strategies.
pub trait ExecutionStrategy: Send + Sync + 'static {}

/// Zero-sized type marker routing operations through scalar execution loops.
pub struct ScalarStrategy;
impl ExecutionStrategy for ScalarStrategy {}

/// Zero-sized type marker routing operations through SIMD (hermes-simd) execution paths.
pub struct SimdStrategy;
impl ExecutionStrategy for SimdStrategy {}

/// Zero-sized type marker routing operations through multi-threaded parallel execution schedules via moirai.
#[cfg(feature = "parallel")]
pub struct ParallelStrategy;
#[cfg(feature = "parallel")]
impl ExecutionStrategy for ParallelStrategy {}

impl sealed::Sealed for SimdStrategy {}

/// Sealed compute trait abstracting SIMD operations from hermes-simd.
pub trait SimdOperations<T: Scalar>: sealed::Sealed {
    /// Vectorized slice addition.
    fn add_slice(a: &[T], b: &[T], out: &mut [T]) -> Result<(), &'static str>;
    /// Vectorized slice subtraction.
    fn sub_slice(a: &[T], b: &[T], out: &mut [T]) -> Result<(), &'static str>;
    /// Vectorized slice multiplication.
    fn mul_slice(a: &[T], b: &[T], out: &mut [T]) -> Result<(), &'static str>;
    /// Vectorized slice division.
    fn div_slice(a: &[T], b: &[T], out: &mut [T]) -> Result<(), &'static str>;
    /// Vectorized sum reduction.
    fn sum_slice(s: &[T]) -> Option<T>;
    /// Vectorized dot product reduction.
    fn dot_slice(a: &[T], b: &[T]) -> Option<T>;
    /// Vectorized fused row update: `out[i] += alpha * x[i]`.
    fn axpy_slice(alpha: T, x: &[T], out: &mut [T]) -> Result<(), &'static str>;
    /// Vectorized fused multi-row update: `out[row, i] += alphas[row] * x[i]`.
    fn axpy_rows(
        alphas: &[T],
        x: &[T],
        out: &mut [T],
        row_stride: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(), &'static str>;
    /// Vectorized batched multi-row update:
    /// `out[row, i] += sum_k alphas[k, row] * x_panel[k, i]`.
    fn axpy_rows_batch(
        alphas: &[T],
        x_panel: &[T],
        out: &mut [T],
        row_stride: usize,
        rows: usize,
        depth: usize,
        cols: usize,
    ) -> Result<(), &'static str>;
    /// Register-blocked tiled GEMM: `c += A * B`.
    fn tiled_gemm(
        a: &[T],
        b: &[T],
        c: &mut [T],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), &'static str>;
    /// Register-blocked sub-matrix GEMV `y += A·x` (row-major `nrows×ncols`,
    /// row stride `lda ≥ ncols`).
    fn gemv_strided(
        a: &[T],
        x: &[T],
        y: &mut [T],
        nrows: usize,
        ncols: usize,
        lda: usize,
    ) -> Result<(), &'static str>;
    /// Register-blocked transposed sub-matrix GEMV `y += Aᵀ·x` (row-major
    /// `nrows×ncols`, row stride `lda ≥ ncols`).
    fn gemv_transpose_strided(
        a: &[T],
        x: &[T],
        y: &mut [T],
        nrows: usize,
        ncols: usize,
        lda: usize,
    ) -> Result<(), &'static str>;
    /// Vectorized absolute-sum reduction: `Σ |x|`.
    fn abs_sum_slice(s: &[T]) -> Option<T>;
    /// Vectorized absolute-max reduction: `max |x|`.
    fn abs_max_slice(s: &[T]) -> Option<T>;
    /// Vectorized min reduction.
    fn min_slice(s: &[T]) -> Option<T>;
    /// Vectorized max reduction.
    fn max_slice(s: &[T]) -> Option<T>;
    /// Jaccard distance between two binary vectors.
    fn jaccard_distance(a: &[T], b: &[T]) -> Option<f64>;
    /// Hamming distance between two binary vectors.
    fn hamming_distance(a: &[T], b: &[T]) -> Option<u64>;
}

macro_rules! impl_simd_ops_native {
    ($t:ty) => {
        impl SimdOperations<$t> for SimdStrategy {
            #[inline(always)]
            fn add_slice(a: &[$t], b: &[$t], out: &mut [$t]) -> Result<(), &'static str> {
                hermes_simd::elementwise_add::<$t>(a, b, out).map_err(|_| "simd add failed")
            }
            #[inline(always)]
            fn sub_slice(a: &[$t], b: &[$t], out: &mut [$t]) -> Result<(), &'static str> {
                hermes_simd::elementwise_sub::<$t>(a, b, out).map_err(|_| "simd sub failed")
            }
            #[inline(always)]
            fn mul_slice(a: &[$t], b: &[$t], out: &mut [$t]) -> Result<(), &'static str> {
                hermes_simd::elementwise_mul::<$t>(a, b, out).map_err(|_| "simd mul failed")
            }
            #[inline(always)]
            fn div_slice(a: &[$t], b: &[$t], out: &mut [$t]) -> Result<(), &'static str> {
                hermes_simd::elementwise_div::<$t>(a, b, out).map_err(|_| "simd div failed")
            }
            #[inline(always)]
            fn sum_slice(s: &[$t]) -> Option<$t> {
                Some(hermes_simd::sum::<$t>(s))
            }
            #[inline(always)]
            fn dot_slice(a: &[$t], b: &[$t]) -> Option<$t> {
                hermes_simd::dot::<$t>(a, b).ok()
            }
            #[inline(always)]
            fn axpy_slice(alpha: $t, x: &[$t], out: &mut [$t]) -> Result<(), &'static str> {
                hermes_simd::axpy::<$t>(alpha, x, out).map_err(|_| "simd axpy failed")
            }
            #[inline(always)]
            fn axpy_rows(
                alphas: &[$t],
                x: &[$t],
                out: &mut [$t],
                row_stride: usize,
                rows: usize,
                cols: usize,
            ) -> Result<(), &'static str> {
                hermes_simd::axpy_rows::<$t>(alphas, x, out, row_stride, rows, cols)
                    .map_err(|_| "simd axpy rows failed")
            }
            #[inline(always)]
            fn axpy_rows_batch(
                alphas: &[$t],
                x_panel: &[$t],
                out: &mut [$t],
                row_stride: usize,
                rows: usize,
                depth: usize,
                cols: usize,
            ) -> Result<(), &'static str> {
                hermes_simd::axpy_rows_batch::<$t>(
                    alphas, x_panel, out, row_stride, rows, depth, cols,
                )
                .map_err(|_| "simd axpy rows batch failed")
            }
            #[inline(always)]
            fn tiled_gemm(
                a: &[$t],
                b: &[$t],
                c: &mut [$t],
                m: usize,
                n: usize,
                k: usize,
            ) -> Result<(), &'static str> {
                hermes_simd::tiled_gemm::<$t>(a, b, c, m, n, k)
                    .map_err(|_| "simd tiled gemm failed")
            }
            #[inline(always)]
            fn gemv_strided(
                a: &[$t],
                x: &[$t],
                y: &mut [$t],
                nrows: usize,
                ncols: usize,
                lda: usize,
            ) -> Result<(), &'static str> {
                hermes_simd::gemv_strided::<$t>(a, x, y, nrows, ncols, lda)
                    .map_err(|_| "simd gemv_strided failed")
            }
            #[inline(always)]
            fn gemv_transpose_strided(
                a: &[$t],
                x: &[$t],
                y: &mut [$t],
                nrows: usize,
                ncols: usize,
                lda: usize,
            ) -> Result<(), &'static str> {
                hermes_simd::gemv_transpose_strided::<$t>(a, x, y, nrows, ncols, lda)
                    .map_err(|_| "simd gemv_transpose_strided failed")
            }
            #[inline(always)]
            fn abs_sum_slice(s: &[$t]) -> Option<$t> {
                Some(hermes_simd::abs_sum::<$t>(s))
            }
            #[inline(always)]
            fn abs_max_slice(s: &[$t]) -> Option<$t> {
                Some(hermes_simd::abs_max::<$t>(s))
            }
            #[inline(always)]
            fn min_slice(s: &[$t]) -> Option<$t> {
                Some(hermes_simd::min::<$t>(s))
            }
            #[inline(always)]
            fn max_slice(s: &[$t]) -> Option<$t> {
                Some(hermes_simd::max::<$t>(s))
            }
            #[inline(always)]
            fn jaccard_distance(a: &[$t], b: &[$t]) -> Option<f64> {
                let intersection = hermes_simd::reduce_popcount_and(a, b).ok()?;
                let union = hermes_simd::reduce_popcount_or(a, b).ok()?;
                if union == 0 {
                    Some(0.0)
                } else {
                    Some(1.0 - (intersection as f64) / (union as f64))
                }
            }
            #[inline(always)]
            fn hamming_distance(a: &[$t], b: &[$t]) -> Option<u64> {
                let dist = hermes_simd::reduce_popcount_xor(a, b).ok()?;
                Some(dist as u64)
            }
        }
    };
}

impl_simd_ops_native!(f32);
impl_simd_ops_native!(f64);

// Reduced-precision types use the scalar fallback until the complete
// `SimdOperations` surface has native Hermes kernels.
macro_rules! impl_simd_ops_unsupported {
    ($t:ty) => {
        impl SimdOperations<$t> for SimdStrategy {
            #[inline(always)]
            fn add_slice(_a: &[$t], _b: &[$t], _out: &mut [$t]) -> Result<(), &'static str> {
                Err("simd unsupported for type")
            }
            #[inline(always)]
            fn sub_slice(_a: &[$t], _b: &[$t], _out: &mut [$t]) -> Result<(), &'static str> {
                Err("simd unsupported for type")
            }
            #[inline(always)]
            fn mul_slice(_a: &[$t], _b: &[$t], _out: &mut [$t]) -> Result<(), &'static str> {
                Err("simd unsupported for type")
            }
            #[inline(always)]
            fn div_slice(_a: &[$t], _b: &[$t], _out: &mut [$t]) -> Result<(), &'static str> {
                Err("simd unsupported for type")
            }
            #[inline(always)]
            fn sum_slice(_s: &[$t]) -> Option<$t> {
                None
            }
            #[inline(always)]
            fn dot_slice(_a: &[$t], _b: &[$t]) -> Option<$t> {
                None
            }
            #[inline(always)]
            fn axpy_slice(_alpha: $t, _x: &[$t], _out: &mut [$t]) -> Result<(), &'static str> {
                Err("simd unsupported for type")
            }
            #[inline(always)]
            fn axpy_rows(
                _alphas: &[$t],
                _x: &[$t],
                _out: &mut [$t],
                _row_stride: usize,
                _rows: usize,
                _cols: usize,
            ) -> Result<(), &'static str> {
                Err("simd unsupported for type")
            }
            #[inline(always)]
            fn axpy_rows_batch(
                _alphas: &[$t],
                _x_panel: &[$t],
                _out: &mut [$t],
                _row_stride: usize,
                _rows: usize,
                _depth: usize,
                _cols: usize,
            ) -> Result<(), &'static str> {
                Err("simd unsupported for type")
            }
            #[inline(always)]
            fn tiled_gemm(
                _a: &[$t],
                _b: &[$t],
                _c: &mut [$t],
                _m: usize,
                _n: usize,
                _k: usize,
            ) -> Result<(), &'static str> {
                Err("simd unsupported for type")
            }
            #[inline(always)]
            fn gemv_strided(
                _a: &[$t],
                _x: &[$t],
                _y: &mut [$t],
                _nrows: usize,
                _ncols: usize,
                _lda: usize,
            ) -> Result<(), &'static str> {
                Err("simd unsupported for type")
            }
            #[inline(always)]
            fn gemv_transpose_strided(
                _a: &[$t],
                _x: &[$t],
                _y: &mut [$t],
                _nrows: usize,
                _ncols: usize,
                _lda: usize,
            ) -> Result<(), &'static str> {
                Err("simd unsupported for type")
            }
            #[inline(always)]
            fn abs_sum_slice(_s: &[$t]) -> Option<$t> {
                None
            }
            #[inline(always)]
            fn abs_max_slice(_s: &[$t]) -> Option<$t> {
                None
            }
            #[inline(always)]
            fn min_slice(_s: &[$t]) -> Option<$t> {
                None
            }
            #[inline(always)]
            fn max_slice(_s: &[$t]) -> Option<$t> {
                None
            }
            #[inline(always)]
            fn jaccard_distance(_a: &[$t], _b: &[$t]) -> Option<f64> {
                None
            }
            #[inline(always)]
            fn hamming_distance(_a: &[$t], _b: &[$t]) -> Option<u64> {
                None
            }
        }
    };
}

impl_simd_ops_unsupported!(F16);
impl_simd_ops_unsupported!(Bf16);
