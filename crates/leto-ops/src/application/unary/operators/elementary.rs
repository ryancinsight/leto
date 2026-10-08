//! The elementary scalar operations: one shared macro, one impl each.

use super::contract::UnaryOp;
use crate::domain::RealScalar;
use eunomia::NumericElement;

macro_rules! define_unary_op {
    ($(#[$meta:meta])* $name:ident => $method:ident) => {
        define_unary_op!($(#[$meta])* $name => $method, true);
    };
    ($(#[$meta:meta])* $name:ident => $method:ident, $compute_bound:expr) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl<T: RealScalar> UnaryOp<T> for $name {
            const COMPUTE_BOUND: bool = $compute_bound;
            #[inline(always)]
            fn apply(&self, x: T) -> T {
                x.$method()
            }
        }
    };
}
define_unary_op!(/// `e^x` operation marker.
    ExpOp => exp);
define_unary_op!(/// Natural logarithm operation marker.
    LnOp => ln);
define_unary_op!(/// Gauss error function operation marker.
    ErfOp => erf);
define_unary_op!(/// Complementary error function operation marker.
    ErfcOp => erfc);
define_unary_op!(/// Natural logarithm of the absolute gamma function operation marker.
    LgammaOp => lgamma);
define_unary_op!(/// Sine operation marker.
    SinOp => sin);
define_unary_op!(/// Cosine operation marker.
    CosOp => cos);
define_unary_op!(/// Square-root operation marker.
    SqrtOp => sqrt);
define_unary_op!(/// Absolute-value operation marker.
    AbsOp => abs, false);
define_unary_op!(/// Additive-inverse operation marker.
    NegOp => neg, false);
define_unary_op!(/// Reciprocal operation marker.
    RecipOp => recip);
define_unary_op!(/// Tangent operation marker.
    TanOp => tan);
define_unary_op!(/// Arcsine operation marker.
    AsinOp => asin);
define_unary_op!(/// Arccosine operation marker.
    AcosOp => acos);
define_unary_op!(/// Arctangent operation marker.
    AtanOp => atan);
define_unary_op!(/// Hyperbolic sine operation marker.
    SinhOp => sinh);
define_unary_op!(/// Hyperbolic cosine operation marker.
    CoshOp => cosh);
define_unary_op!(/// Hyperbolic tangent operation marker.
    TanhOp => tanh);
define_unary_op!(/// Base-2 logarithm operation marker.
    Log2Op => log2);
define_unary_op!(/// Base-10 logarithm operation marker.
    Log10Op => log10);
define_unary_op!(/// Base-2 exponential operation marker.
    Exp2Op => exp2);
define_unary_op!(/// Inverse hyperbolic tangent operation marker.
    AtanhOp => atanh);
define_unary_op!(/// Inverse hyperbolic sine operation marker.
    AsinhOp => asinh);
define_unary_op!(/// Inverse hyperbolic cosine operation marker.
    AcoshOp => acosh);
define_unary_op!(/// `e^x - 1` operation marker, accurate near zero.
    Expm1Op => exp_m1);
define_unary_op!(/// `ln(1 + x)` operation marker, accurate near zero.
    Log1pOp => ln_1p);
define_unary_op!(/// Floor operation marker.
    FloorOp => floor, false);
define_unary_op!(/// Ceiling operation marker.
    CeilOp => ceil, false);
define_unary_op!(/// Round-to-nearest operation marker.
    RoundOp => round, false);
define_unary_op!(/// Truncation operation marker.
    TruncOp => trunc, false);
/// Negated exponential `e^-x` operation marker, matching hephaestus `ExpNegOp`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpNegOp;
impl<T: RealScalar> UnaryOp<T> for ExpNegOp {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        (-x).exp()
    }
}
/// Sign operation marker: `0` for `±0` and `NaN`, `1`/`-1` otherwise.
///
/// Matches hephaestus `SignOp` (ADR 0061 Decision 7), NOT eunomia's `signum`:
/// the GPU renderings cannot propagate NaN, so NaN reads back as zero.
#[derive(Clone, Copy, Debug, Default)]
pub struct SignOp;
impl<T: RealScalar> UnaryOp<T> for SignOp {
    const COMPUTE_BOUND: bool = false;
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if x.is_nan() {
            <T as NumericElement>::ZERO
        } else if x > <T as NumericElement>::ZERO {
            <T as NumericElement>::ONE
        } else if x < <T as NumericElement>::ZERO {
            -<T as NumericElement>::ONE
        } else {
            <T as NumericElement>::ZERO
        }
    }
}
/// Power operation carrying its exponent. Zero-cost: monomorphizes to a direct
/// `powf` call with the captured exponent.
#[derive(Clone, Copy, Debug)]
pub struct PowfOp<T: RealScalar> {
    /// The exponent applied to every element.
    pub exponent: T,
}
impl<T: RealScalar> UnaryOp<T> for PowfOp<T> {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        x.powf(self.exponent)
    }
}
/// Unnormalized `sinc` operation marker: `sin(x)/x` with `sinc(0) = 1`.
///
/// Zero-sized: monomorphizes to a direct `sin`/divide pair per element. The
/// removable singularity branches on exact zero (including `-0.0`, which
/// compares equal), so no per-precision epsilon literal is needed; NaN
/// propagates through the division. This is the elementwise counterpart of
/// [`sinc`](crate::application::special::sinc), which keeps its epsilon band
/// for direct scalar calls.
#[derive(Clone, Copy, Debug, Default)]
pub struct SincOp;
impl<T: RealScalar> UnaryOp<T> for SincOp {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        if x == <T as NumericElement>::ZERO {
            <T as NumericElement>::ONE
        } else {
            x.sin() / x
        }
    }
}
