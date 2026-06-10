use crate::domain::scalar::Scalar;
use half::{bf16, f16};

/// Floating-point scalars that provide the real transcendental and shape
/// functions required by elementwise math operations.
///
/// This is a deliberately segregated extension of [`Scalar`]: arithmetic and
/// reductions live on `Scalar`, while the transcendental surface lives here so
/// that integer scalars can implement `Scalar` without being forced to provide
/// meaningless `exp`/`ln` operations.
///
/// Native types (`f32`, `f64`) map each method to the hardware/std operation.
/// Reduced-precision types (`f16`, `bf16`) have no hardware transcendental
/// path on common ISAs; their implementations document the single sanctioned
/// fallback (compute in `f32`, round back) rather than a hidden widen-narrow.
pub trait RealScalar: Scalar {
    /// `e^self`.
    fn exp(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// Sine (radians).
    fn sin(self) -> Self;
    /// Cosine (radians).
    fn cos(self) -> Self;
    /// Square root.
    fn sqrt(self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// Additive inverse.
    fn neg(self) -> Self;
    /// Reciprocal `1 / self`.
    fn recip(self) -> Self;
    /// `self` raised to the power `exponent`.
    fn powf(self, exponent: Self) -> Self;
    /// Four-quadrant arctangent of `self / other` (radians).
    fn atan2(self, other: Self) -> Self;
    /// Returns true when the value is neither infinite nor NaN.
    fn is_finite(self) -> bool;
}

macro_rules! impl_real_native {
    ($t:ty) => {
        impl RealScalar for $t {
            #[inline(always)]
            fn exp(self) -> Self {
                self.exp()
            }
            #[inline(always)]
            fn ln(self) -> Self {
                self.ln()
            }
            #[inline(always)]
            fn sin(self) -> Self {
                self.sin()
            }
            #[inline(always)]
            fn cos(self) -> Self {
                self.cos()
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            #[inline(always)]
            fn abs(self) -> Self {
                self.abs()
            }
            #[inline(always)]
            fn neg(self) -> Self {
                -self
            }
            #[inline(always)]
            fn recip(self) -> Self {
                self.recip()
            }
            #[inline(always)]
            fn powf(self, exponent: Self) -> Self {
                self.powf(exponent)
            }
            #[inline(always)]
            fn atan2(self, other: Self) -> Self {
                self.atan2(other)
            }
            #[inline(always)]
            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }
        }
    };
}

/// Reduced-precision implementation. No common ISA exposes scalar `f16`/`bf16`
/// transcendental instructions, so the sanctioned fallback computes in `f32`
/// and rounds back. This is the documented precision contract, not a hidden
/// widen-narrow: the caller selected reduced precision and accepts a single
/// rounding at the boundary of each elementwise call.
macro_rules! impl_real_half {
    ($t:ty) => {
        impl RealScalar for $t {
            #[inline]
            fn exp(self) -> Self {
                <$t>::from_f32(self.to_f32().exp())
            }
            #[inline]
            fn ln(self) -> Self {
                <$t>::from_f32(self.to_f32().ln())
            }
            #[inline]
            fn sin(self) -> Self {
                <$t>::from_f32(self.to_f32().sin())
            }
            #[inline]
            fn cos(self) -> Self {
                <$t>::from_f32(self.to_f32().cos())
            }
            #[inline]
            fn sqrt(self) -> Self {
                <$t>::from_f32(self.to_f32().sqrt())
            }
            #[inline]
            fn abs(self) -> Self {
                <$t>::from_f32(self.to_f32().abs())
            }
            #[inline]
            fn neg(self) -> Self {
                -self
            }
            #[inline]
            fn recip(self) -> Self {
                <$t>::from_f32(self.to_f32().recip())
            }
            #[inline]
            fn powf(self, exponent: Self) -> Self {
                <$t>::from_f32(self.to_f32().powf(exponent.to_f32()))
            }
            #[inline]
            fn atan2(self, other: Self) -> Self {
                <$t>::from_f32(self.to_f32().atan2(other.to_f32()))
            }
            #[inline]
            fn is_finite(self) -> bool {
                self.to_f32().is_finite()
            }
        }
    };
}

impl_real_native!(f32);
impl_real_native!(f64);
impl_real_half!(f16);
impl_real_half!(bf16);
