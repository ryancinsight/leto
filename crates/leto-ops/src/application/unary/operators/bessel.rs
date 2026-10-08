//! The Bessel-function operations and their series kernels.

use super::contract::UnaryOp;
use crate::domain::RealScalar;
use eunomia::NumericElement;

/// Bessel J0 over any real scalar: the Numerical-Recipes rational
/// approximation for `|x| < 8` and Hankel's asymptotic expansion otherwise,
/// transcribed verbatim from [`j0`](crate::application::special::j0) with the
/// same nesting so every precision evaluates the same program. Coefficients
/// enter through [`FloatElement::from_f64`](eunomia::FloatElement::from_f64),
/// precision-correct per element type; the literal arguments constant-fold.
fn bessel_j0_value<T: RealScalar>(x: T) -> T {
    use core::f64::consts::{FRAC_PI_4, PI};
    if x == <T as NumericElement>::ZERO {
        return <T as NumericElement>::ONE;
    }
    let ax = <T as NumericElement>::abs(x);
    if ax < T::from_f64(8.0) {
        let y = x * x;
        let num = T::from_f64(57568490574.0)
            + y * (T::from_f64(-13362590354.0)
                + y * (T::from_f64(651619640.7)
                    + y * (T::from_f64(-11214424.18)
                        + y * (T::from_f64(77392.33017) + y * T::from_f64(-184.9052456)))));
        let den = T::from_f64(57568490411.0)
            + y * (T::from_f64(1029532985.0)
                + y * (T::from_f64(9494680.718)
                    + y * (T::from_f64(59272.64853) + y * (T::from_f64(267.8532712) + y))));
        num / den
    } else {
        let z = T::from_f64(8.0) / ax;
        let y = z * z;
        let xx = ax - T::from_f64(FRAC_PI_4);
        let p = <T as NumericElement>::ONE
            + y * (T::from_f64(-0.001098628627)
                + y * (T::from_f64(0.000002734510407)
                    + y * (T::from_f64(-2.073370639e-6) + y * T::from_f64(2.093887211e-7))));
        let q = T::from_f64(-0.01562499995)
            + y * (T::from_f64(0.0001430488765)
                + y * (T::from_f64(-6.911147651e-5)
                    + y * (T::from_f64(7.621095161e-5) - y * T::from_f64(9.34935152e-7))));
        (T::from_f64(2.0) / (T::from_f64(PI) * ax)).sqrt() * (p * xx.cos() - z * q * xx.sin())
    }
}
/// Bessel J1 over any real scalar: the same Numerical-Recipes/Hankel
/// program as [`j1`](crate::application::special::j1), transcribed verbatim
/// with the same nesting. See [`bessel_j0_value`] for the coefficient
/// convention.
fn bessel_j1_value<T: RealScalar>(x: T) -> T {
    use core::f64::consts::PI;
    if x == <T as NumericElement>::ZERO {
        return <T as NumericElement>::ZERO;
    }
    let ax = <T as NumericElement>::abs(x);
    if ax < T::from_f64(8.0) {
        let y = x * x;
        let num = x
            * (T::from_f64(72362614232.0)
                + y * (T::from_f64(-7895059235.0)
                    + y * (T::from_f64(242396853.1)
                        + y * (T::from_f64(-2972611.439)
                            + y * (T::from_f64(15704.48260) + y * T::from_f64(-30.16036606))))));
        let den = T::from_f64(144725228442.0)
            + y * (T::from_f64(2300535178.0)
                + y * (T::from_f64(18583304.74)
                    + y * (T::from_f64(99447.43394) + y * (T::from_f64(376.9991397) + y))));
        num / den
    } else {
        let z = T::from_f64(8.0) / ax;
        let y = z * z;
        let xx = ax - T::from_f64(3.0 * PI / 4.0);
        let p = <T as NumericElement>::ONE
            + y * (T::from_f64(0.183105e-2)
                + y * (T::from_f64(-3.516396496e-5)
                    + y * (T::from_f64(2.457520174e-5) - y * T::from_f64(2.400505341e-7))));
        let q = T::from_f64(0.04687499995)
            + y * (T::from_f64(-0.2002690873e-3)
                + y * (T::from_f64(8.449199096e-5)
                    + y * (T::from_f64(-8.8228987e-5) + y * T::from_f64(1.050343160e-6))));
        let r =
            (T::from_f64(2.0) / (T::from_f64(PI) * ax)).sqrt() * (p * xx.cos() - z * q * xx.sin());
        if x < <T as NumericElement>::ZERO {
            -r
        } else {
            r
        }
    }
}
/// Bessel function of the first kind J0 operation marker.
///
/// Zero-sized: monomorphizes to the Numerical-Recipes rational/Hankel
/// program per element. This is the elementwise counterpart of
/// [`j0`](crate::application::special::j0), evaluating the same nested
/// formulation in the lane precision.
#[derive(Clone, Copy, Debug, Default)]
pub struct J0Op;
impl<T: RealScalar> UnaryOp<T> for J0Op {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        bessel_j0_value(x)
    }
}
/// Bessel function of the first kind J1 operation marker.
///
/// Zero-sized: monomorphizes to the Numerical-Recipes rational/Hankel
/// program per element. This is the elementwise counterpart of
/// [`j1`](crate::application::special::j1), evaluating the same nested
/// formulation in the lane precision.
#[derive(Clone, Copy, Debug, Default)]
pub struct J1Op;
impl<T: RealScalar> UnaryOp<T> for J1Op {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        bessel_j1_value(x)
    }
}
/// Modified Bessel function of the second kind K0 over any real scalar:
/// Abramowitz & Stegun 9.8.5 for `0 < x <= 2` and 9.8.6 for `x > 2`,
/// transcribed verbatim from [`bessel_k0`](crate::application::special::bessel_k0)
/// with the same nesting so every precision evaluates the same program.
/// Non-positive and non-finite arguments yield `NaN`, as the scalar does.
/// One deliberate rendering difference: the scalar fuses its final
/// `ln(x/2) * -I0 + correction` with `mul_add`, while this form keeps the
/// multiply and add separate so the transcription stays expressible in
/// every lane type; the two roundings differ by at most 1 ULP.
fn bessel_k0_value<T: RealScalar>(x: T) -> T {
    if !(x.is_finite() && x > <T as NumericElement>::ZERO) {
        return <T as NumericElement>::NAN;
    }
    if x <= T::from_f64(2.0) {
        let t1 = (x / T::from_f64(3.75)) * (x / T::from_f64(3.75));
        let i0 = <T as NumericElement>::ONE
            + t1 * (T::from_f64(3.515_622_9)
                + t1 * (T::from_f64(3.089_942_4)
                    + t1 * (T::from_f64(1.206_749_2)
                        + t1 * (T::from_f64(0.265_973_2)
                            + t1 * (T::from_f64(0.036_076_8) + t1 * T::from_f64(0.004_581_3))))));
        let t2 = (x * T::from_f64(0.5)) * (x * T::from_f64(0.5));
        let correction = T::from_f64(-0.577_215_66)
            + t2 * (T::from_f64(0.422_784_20)
                + t2 * (T::from_f64(0.230_697_56)
                    + t2 * (T::from_f64(0.034_885_90)
                        + t2 * (T::from_f64(0.002_626_98)
                            + t2 * (T::from_f64(0.000_107_50) + t2 * T::from_f64(7.4e-6))))));
        (x * T::from_f64(0.5)).ln() * -i0 + correction
    } else {
        let t = T::from_f64(2.0) / x;
        let series = T::from_f64(1.253_314_14)
            + t * (T::from_f64(-0.078_323_58)
                + t * (T::from_f64(0.021_895_68)
                    + t * (T::from_f64(-0.010_624_46)
                        + t * (T::from_f64(0.005_878_72)
                            + t * (T::from_f64(-0.002_515_40) + t * T::from_f64(0.000_532_08))))));
        (-x).exp() / x.sqrt() * series
    }
}
/// Modified Bessel function of the second kind K0 operation marker.
///
/// Zero-sized: monomorphizes to the Abramowitz & Stegun 9.8.5/9.8.6 program
/// per element. This is the elementwise counterpart of
/// [`bessel_k0`](crate::application::special::bessel_k0), evaluating the
/// same nested formulation in the lane precision.
#[derive(Clone, Copy, Debug, Default)]
pub struct K0Op;
impl<T: RealScalar> UnaryOp<T> for K0Op {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        bessel_k0_value(x)
    }
}
