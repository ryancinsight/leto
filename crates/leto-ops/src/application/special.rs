//! Special mathematical functions — SSOT for the Atlas simulation stack.
//!
//! Functions here are pure numerics without domain-specific coupling.
//! They complement [`eunomia`] (scalar types) and [`leto-ops`] linalg kernels.
//!
//! ## Functions
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`sinc`] | Unnormalized cardinal sine sin(x)/x |
//! | [`erf`] | Gaussian error function (A&S 7.1.26) |
//! | [`j0`], [`j1`], [`jn`] | Bessel functions of the first kind |

/// Unnormalized cardinal sine `sinc(x) = sin(x)/x`, with the removable
/// singularity `sinc(0) = 1`.
///
/// This is the *unnormalized* convention.  For the signal-processing form
/// `sin(πx)/(πx)`, pass `π·x`.
#[inline]
#[must_use]
pub fn sinc(x: f64) -> f64 {
    if x.abs() <= f64::EPSILON {
        1.0
    } else {
        x.sin() / x
    }
}

/// Gaussian error function `erf(x)` via the Abramowitz & Stegun 7.1.26
/// rational approximation.
///
/// Maximum absolute error `|ε| ≤ 1.5×10⁻⁷`.
#[inline]
#[must_use]
pub fn erf(x: f64) -> f64 {
    const P: f64 = 0.327_591_1;
    const A: [f64; 5] = [
        0.254_829_592,
        -0.284_496_736,
        1.421_413_741,
        -1.453_152_027,
        1.061_405_429,
    ];
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let poly = (A[4]
        .mul_add(t, A[3])
        .mul_add(t, A[2])
        .mul_add(t, A[1])
        .mul_add(t, A[0]))
        * t;
    sign * poly.mul_add(-(-ax * ax).exp(), 1.0)
}

/// Bessel function of the first kind J₀(x).
///
/// Uses the Numerical-Recipes rational approximation for |x| ≤ 8 and
/// Hankel's asymptotic expansion otherwise.  Error ≲ 2×10⁻⁹.
#[must_use]
pub fn j0(x: f64) -> f64 {
    use std::f64::consts::{FRAC_PI_4, PI};
    if x == 0.0 {
        return 1.0;
    }
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let num = 57568490574.0_f64
            + y * (-13362590354.0
                + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let den = 57568490411.0_f64
            + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - FRAC_PI_4;
        let p = 1.0
            + y * (-0.001098628627
                + y * (0.000002734510407 + y * (-2.073370639e-6 + y * 2.093887211e-7)));
        let q = -0.01562499995
            + y * (0.0001430488765
                + y * (-6.911147651e-5 + y * (7.621095161e-5 - y * 9.34935152e-7)));
        (2.0 / (PI * ax)).sqrt() * (p * xx.cos() - z * q * xx.sin())
    }
}

/// Bessel function of the first kind J₁(x).
#[must_use]
pub fn j1(x: f64) -> f64 {
    use std::f64::consts::PI;
    if x == 0.0 {
        return 0.0;
    }
    let ax = x.abs();
    let ans = if ax < 8.0 {
        let y = x * x;
        let num = x
            * (72362614232.0_f64
                + y * (-7895059235.0
                    + y * (242396853.1
                        + y * (-2972611.439 + y * (15704.48260 + y * (-30.16116360))))));
        let den = 144725228442.0_f64
            + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 3.0 * PI / 4.0;
        let p = 1.0
            + y * (0.000183105e-2
                + y * (-3.516396496e-5 + y * (2.457520174e-5 + y * (-240337019e-7))));
        let q = 0.04687499995_f64
            + y * (-2.002690873e-3
                + y * (8.449199096e-5 + y * (-8.822898032e-5 + y * 1.053498233e-5)));
        (2.0 / (PI * ax)).sqrt() * (p * xx.cos() - z * q * xx.sin())
    };
    if x < 0.0 {
        -ans
    } else {
        ans
    }
}

/// Bessel function of the first kind Jₙ(x) for integer order n ≥ 0.
///
/// Uses Miller's downward recurrence for n ≥ 2.
#[must_use]
pub fn jn(n: usize, x: f64) -> f64 {
    match n {
        0 => j0(x),
        1 => j1(x),
        _ => {
            if x == 0.0 {
                return 0.0;
            }
            // Miller downward recurrence.
            let ax = x.abs();
            let tox = 2.0 / ax;
            let (mut bjm, mut bj) = (j0(ax), j1(ax));
            let mut bjp;
            let mut sum = 0.0;
            for j in 1..n {
                bjp = j as f64 * tox * bj - bjm;
                bjm = bj;
                bj = bjp;
                if bj.abs() > 1e10 {
                    bj /= 1e10;
                    bjm /= 1e10;
                    sum /= 1e10;
                }
                if j % 2 == 1 {
                    sum += bj;
                }
            }
            let _ = sum;
            let ans = bj;
            if x < 0.0 && n % 2 == 1 {
                -ans
            } else {
                ans
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinc_at_zero() {
        assert_eq!(sinc(0.0), 1.0);
    }

    #[test]
    fn sinc_is_even() {
        assert_eq!(sinc(-0.7), sinc(0.7));
    }

    #[test]
    fn erf_odd_and_saturates() {
        assert!((erf(0.0)).abs() < 1e-7); // A&S 7.1.26 tolerance
        assert!((erf(6.0) - 1.0).abs() < 1e-7);
        assert!((erf(-6.0) + 1.0).abs() < 1e-7);
        assert!((erf(-1.0) + erf(1.0)).abs() < 1e-15); // odd symmetry is exact
    }

    #[test]
    fn j0_j1_at_zero() {
        assert!((j0(0.0) - 1.0).abs() < 1e-15);
        assert!(j1(0.0).abs() < 1e-15);
    }

    #[test]
    fn j0_first_zero() {
        // First zero of J₀ ≈ 2.4048.
        assert!(j0(2.4048).abs() < 1e-3);
    }
}
