use crate::domain::scalar::Scalar;
use half::{bf16, f16};

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
    /// Vectorized min reduction.
    fn min_slice(s: &[T]) -> Option<T>;
    /// Vectorized max reduction.
    fn max_slice(s: &[T]) -> Option<T>;
}

#[cfg(feature = "simd")]
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
            fn min_slice(s: &[$t]) -> Option<$t> {
                Some(hermes_simd::min::<$t>(s))
            }
            #[inline(always)]
            fn max_slice(s: &[$t]) -> Option<$t> {
                Some(hermes_simd::max::<$t>(s))
            }
        }
    };
}

#[cfg(feature = "simd")]
impl_simd_ops_native!(f32);
#[cfg(feature = "simd")]
impl_simd_ops_native!(f64);

#[cfg(not(feature = "simd"))]
macro_rules! impl_simd_ops_fallback {
    ($t:ty) => {
        impl SimdOperations<$t> for SimdStrategy {
            #[inline(always)]
            fn add_slice(_a: &[$t], _b: &[$t], _out: &mut [$t]) -> Result<(), &'static str> {
                Err("simd disabled")
            }
            #[inline(always)]
            fn sub_slice(_a: &[$t], _b: &[$t], _out: &mut [$t]) -> Result<(), &'static str> {
                Err("simd disabled")
            }
            #[inline(always)]
            fn mul_slice(_a: &[$t], _b: &[$t], _out: &mut [$t]) -> Result<(), &'static str> {
                Err("simd disabled")
            }
            #[inline(always)]
            fn div_slice(_a: &[$t], _b: &[$t], _out: &mut [$t]) -> Result<(), &'static str> {
                Err("simd disabled")
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
            fn min_slice(_s: &[$t]) -> Option<$t> {
                None
            }
            #[inline(always)]
            fn max_slice(_s: &[$t]) -> Option<$t> {
                None
            }
        }
    };
}

#[cfg(not(feature = "simd"))]
impl_simd_ops_fallback!(f32);
#[cfg(not(feature = "simd"))]
impl_simd_ops_fallback!(f64);

// f16 and bf16 always use fallback
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
            fn min_slice(_s: &[$t]) -> Option<$t> {
                None
            }
            #[inline(always)]
            fn max_slice(_s: &[$t]) -> Option<$t> {
                None
            }
        }
    };
}

impl_simd_ops_unsupported!(f16);
impl_simd_ops_unsupported!(bf16);
