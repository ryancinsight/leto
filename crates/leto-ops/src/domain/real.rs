use crate::domain::scalar::Scalar;
use eunomia::{FloatElement, NumericElement};

/// Floating-point scalars that provide the real math surface required by Leto
/// operations.
///
/// Eunomia owns the real scalar SSOT through [`FloatElement`]. `RealScalar`
/// only adds Leto's operation-local dense norm-reduction kernels. Arithmetic
/// and reductions still execute in the selected scalar precision; no hidden
/// wider compute path is introduced here.
pub trait RealScalar: Scalar + FloatElement + core::ops::Neg<Output = Self> {
    /// `sum |x|` over a dense slice (L1-norm accumulator).
    ///
    /// Default: scalar fold in the precision of `Self`. Native impls override
    /// through the same SIMD strategy used by [`Scalar`] reductions.
    #[inline]
    fn abs_sum_slice(s: &[Self]) -> Self {
        s.iter().fold(<Self as NumericElement>::ZERO, |acc, &x| {
            acc.add(<Self as NumericElement>::abs(x))
        })
    }

    /// `max |x|` over a dense slice (infinity-norm accumulator); `ZERO` for empty.
    ///
    /// Default: scalar fold. Native impls override through the same SIMD
    /// strategy used by [`Scalar`] reductions.
    #[inline]
    fn abs_max_slice(s: &[Self]) -> Self {
        s.iter().fold(<Self as NumericElement>::ZERO, |acc, &x| {
            let magnitude = <Self as NumericElement>::abs(x);
            if magnitude > acc {
                magnitude
            } else {
                acc
            }
        })
    }
}

macro_rules! impl_real_simd {
    ($t:ty) => {
        impl RealScalar for $t {
            #[inline]
            fn abs_sum_slice(s: &[Self]) -> Self {
                use crate::domain::strategy::{SimdOperations, SimdStrategy};
                if let Some(res) = <SimdStrategy as SimdOperations<Self>>::abs_sum_slice(s) {
                    res
                } else {
                    s.iter().fold(<Self as NumericElement>::ZERO, |acc, &x| {
                        acc + <Self as NumericElement>::abs(x)
                    })
                }
            }

            #[inline]
            fn abs_max_slice(s: &[Self]) -> Self {
                use crate::domain::strategy::{SimdOperations, SimdStrategy};
                if let Some(res) = <SimdStrategy as SimdOperations<Self>>::abs_max_slice(s) {
                    res
                } else {
                    s.iter().fold(<Self as NumericElement>::ZERO, |acc, &x| {
                        let magnitude = <Self as NumericElement>::abs(x);
                        if magnitude > acc {
                            magnitude
                        } else {
                            acc
                        }
                    })
                }
            }
        }
    };
}

impl_real_simd!(f32);
impl_real_simd!(f64);

impl RealScalar for half::f16 {}
impl RealScalar for half::bf16 {}
