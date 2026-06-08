use half::{f16, bf16};

/// A trait representing scalar numeric types with native precision execution.
pub trait Scalar: Copy + Send + Sync + PartialEq + 'static {
    /// The zero value.
    const ZERO: Self;
    /// The one value.
    const ONE: Self;

    /// Scalar addition.
    fn add(self, other: Self) -> Self;
    /// Scalar subtraction.
    fn sub(self, other: Self) -> Self;
    /// Scalar multiplication.
    fn mul(self, other: Self) -> Self;
    /// Scalar division.
    fn div(self, other: Self) -> Self;

    /// Element-wise slice addition: `out = a + b`.
    fn add_slice(a: &[Self], b: &[Self], out: &mut [Self]);
    /// Element-wise slice subtraction: `out = a - b`.
    fn sub_slice(a: &[Self], b: &[Self], out: &mut [Self]);
    /// Element-wise slice multiplication: `out = a * b`.
    fn mul_slice(a: &[Self], b: &[Self], out: &mut [Self]);
    /// Element-wise slice division: `out = a / b`.
    fn div_slice(a: &[Self], b: &[Self], out: &mut [Self]);

    /// Sum reduction over a slice.
    fn sum_slice(s: &[Self]) -> Self;
    /// Min reduction over a slice.
    fn min_slice(s: &[Self]) -> Self;
    /// Max reduction over a slice.
    fn max_slice(s: &[Self]) -> Self;
}

// Helper macros for implementing Scalar for native and half types
macro_rules! impl_scalar_native {
    ($t:ty) => {
        impl Scalar for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;

            #[inline(always)]
            fn add(self, other: Self) -> Self { self + other }
            #[inline(always)]
            fn sub(self, other: Self) -> Self { self - other }
            #[inline(always)]
            fn mul(self, other: Self) -> Self { self * other }
            #[inline(always)]
            fn div(self, other: Self) -> Self { self / other }

            #[inline]
            fn add_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                #[cfg(feature = "simd")]
                {
                    if hermes_simd::elementwise_add::<$t>(a, b, out).is_ok() {
                        return;
                    }
                }
                for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                    *o = x + y;
                }
            }

            #[inline]
            fn sub_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                #[cfg(feature = "simd")]
                {
                    if hermes_simd::elementwise_sub::<$t>(a, b, out).is_ok() {
                        return;
                    }
                }
                for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                    *o = x - y;
                }
            }

            #[inline]
            fn mul_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                #[cfg(feature = "simd")]
                {
                    if hermes_simd::elementwise_mul::<$t>(a, b, out).is_ok() {
                        return;
                    }
                }
                for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                    *o = x * y;
                }
            }

            #[inline]
            fn div_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                #[cfg(feature = "simd")]
                {
                    if hermes_simd::elementwise_div::<$t>(a, b, out).is_ok() {
                        return;
                    }
                }
                for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                    *o = x / y;
                }
            }

            #[inline]
            fn sum_slice(s: &[Self]) -> Self {
                #[cfg(feature = "simd")]
                {
                    hermes_simd::sum::<$t>(s)
                }
                #[cfg(not(feature = "simd"))]
                {
                    s.iter().copied().fold(Self::ZERO, |acc, x| acc + x)
                }
            }

            #[inline]
            fn min_slice(s: &[Self]) -> Self {
                #[cfg(feature = "simd")]
                {
                    hermes_simd::min::<$t>(s)
                }
                #[cfg(not(feature = "simd"))]
                {
                    s.iter().copied().fold(Self::INFINITY, |acc, x| if x < acc { x } else { acc })
                }
            }

            #[inline]
            fn max_slice(s: &[Self]) -> Self {
                #[cfg(feature = "simd")]
                {
                    hermes_simd::max::<$t>(s)
                }
                #[cfg(not(feature = "simd"))]
                {
                    s.iter().copied().fold(Self::NEG_INFINITY, |acc, x| if x > acc { x } else { acc })
                }
            }
        }
    };
}

macro_rules! impl_scalar_half {
    ($t:ty) => {
        impl Scalar for $t {
            const ZERO: Self = Self::ZERO;
            const ONE: Self = Self::ONE;

            #[inline(always)]
            fn add(self, other: Self) -> Self { self + other }
            #[inline(always)]
            fn sub(self, other: Self) -> Self { self - other }
            #[inline(always)]
            fn mul(self, other: Self) -> Self { self * other }
            #[inline(always)]
            fn div(self, other: Self) -> Self { self / other }

            #[inline]
            fn add_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                    *o = x + y;
                }
            }

            #[inline]
            fn sub_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                    *o = x - y;
                }
            }

            #[inline]
            fn mul_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                    *o = x * y;
                }
            }

            #[inline]
            fn div_slice(a: &[Self], b: &[Self], out: &mut [Self]) {
                for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
                    *o = x / y;
                }
            }

            #[inline]
            fn sum_slice(s: &[Self]) -> Self {
                s.iter().copied().fold(Self::ZERO, |acc, x| acc + x)
            }

            #[inline]
            fn min_slice(s: &[Self]) -> Self {
                s.iter().copied().fold(Self::INFINITY, |acc, x| if x < acc { x } else { acc })
            }

            #[inline]
            fn max_slice(s: &[Self]) -> Self {
                s.iter().copied().fold(Self::NEG_INFINITY, |acc, x| if x > acc { x } else { acc })
            }
        }
    };
}

impl_scalar_native!(f32);
impl_scalar_native!(f64);
impl_scalar_half!(f16);
impl_scalar_half!(bf16);
