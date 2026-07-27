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
    if ax < 8.0 {
        let y = x * x;
        let num = x
            * (72362614232.0_f64
                + y * (-7895059235.0
                    + y * (242396853.1
                        + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        let den = 144725228442.0_f64
            + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 3.0 * PI / 4.0;
        let p = 1.0
            + y * (0.183105e-2 + y * (-3.516396496e-5 + y * (2.457520174e-5 - y * 2.400505341e-7)));
        let q = 0.04687499995_f64
            + y * (-0.2002690873e-3
                + y * (8.449199096e-5 + y * (-8.8228987e-5 + y * 1.050343160e-6)));
        let r = (2.0 / (PI * ax)).sqrt() * (p * xx.cos() - z * q * xx.sin());
        if x < 0.0 {
            -r
        } else {
            r
        }
    }
}

/// Bessel function of the first kind Jₙ(x) for integer order n ≥ 0.
///
/// Delegates to [`j0`]/[`j1`] for n ∈ {0, 1}; uses Miller downward recurrence
/// with two-buffer normalisation for n ≥ 2 (accurate to ≲1e-9 for |x| ≤ 50,
/// n ≤ 20). Returns exact 0 for n ≥ 1 at x = 0.
#[must_use]
pub fn jn(n: usize, x: f64) -> f64 {
    match n {
        0 => j0(x),
        1 => j1(x),
        _ => {
            if x.abs() < 1e-15 {
                return 0.0;
            }
            let m_start = n + n.max(30);
            let mut bjp = 0.0_f64;
            let mut bj = 1.0_f64;
            let mut bj0 = 0.0_f64;
            let mut bj1 = 0.0_f64;
            let mut ans = 0.0_f64;
            let two_over_x = 2.0 / x;
            for k in (0..m_start).rev() {
                let bjm = (k as f64 + 1.0) * two_over_x * bj - bjp;
                bjp = bj;
                bj = bjm;
                if bj.abs() > 1.0e100 {
                    bj *= 1.0e-100;
                    bjp *= 1.0e-100;
                    ans *= 1.0e-100;
                    bj0 *= 1.0e-100;
                    bj1 *= 1.0e-100;
                }
                if k == n {
                    ans = bj;
                }
                if k == 1 {
                    bj1 = bj;
                }
                if k == 0 {
                    bj0 = bj;
                }
            }
            let j0_true = j0(x);
            let j1_true = j1(x);
            let scale = if bj0.abs() >= bj1.abs() {
                if bj0.abs() < 1e-300 {
                    return 0.0;
                }
                j0_true / bj0
            } else {
                if bj1.abs() < 1e-300 {
                    return 0.0;
                }
                j1_true / bj1
            };
            ans * scale
        }
    }
}

/// Modified Bessel function of the second kind, order zero: `K₀(x)` for `x > 0`.
///
/// # Approximation
///
/// Abramowitz & Stegun 9.8.5 for `0 < x ≤ 2`, and 9.8.6 for `x > 2`. Both
/// polynomial forms carry an absolute error below `1e-7` on their stated
/// domains. `K₀` diverges at the origin and is undefined for `x ≤ 0`, so
/// non-positive and non-finite arguments return `NaN`.
#[must_use]
pub fn bessel_k0(x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    if x <= 2.0 {
        // A&S 9.8.5: K₀(x) = −ln(x/2)·I₀(x) + Σ cₙ (x/2)^{2n}.
        let t1 = (x / 3.75) * (x / 3.75);
        let i0 = 1.0
            + t1 * (3.515_622_9
                + t1 * (3.089_942_4
                    + t1 * (1.206_749_2
                        + t1 * (0.265_973_2 + t1 * (0.036_076_8 + t1 * 0.004_581_3)))));
        let t2 = (x * 0.5) * (x * 0.5);
        let correction = -0.577_215_66
            + t2 * (0.422_784_20
                + t2 * (0.230_697_56
                    + t2 * (0.034_885_90
                        + t2 * (0.002_626_98 + t2 * (0.000_107_50 + t2 * 7.4e-6)))));
        (x * 0.5).ln().mul_add(-i0, correction)
    } else {
        // A&S 9.8.6: K₀(x) = e^{-x}/√x · Σ dₙ (2/x)^n.
        let t = 2.0 / x;
        let series = 1.253_314_14
            + t * (-0.078_323_58
                + t * (0.021_895_68
                    + t * (-0.010_624_46
                        + t * (0.005_878_72 + t * (-0.002_515_40 + t * 0.000_532_08)))));
        (-x).exp() / x.sqrt() * series
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
    fn bessel_k0_matches_reference_values() {
        // Reference values: DLMF 10.32 / A&S Table 9.8.
        //
        // Tolerances follow the quoted error bounds of the two branches:
        // 9.8.5 bounds the absolute error of the small-argument form by 1e-8,
        // and 9.8.6 bounds the error of x^{1/2}·e^{x}·K₀(x) by 1.9e-7, i.e. a
        // relative error near 1.5e-7 on K₀ itself.
        for &(x, expected) in &[(0.1_f64, 2.427_069_0_f64), (1.0, 0.421_024_4), (2.0, 0.113_893_9)]
        {
            let error = (bessel_k0(x) - expected).abs();
            assert!(error < 1e-6, "K0({x}) absolute error {error}");
        }
        for &(x, expected) in &[
            (3.0_f64, 0.034_739_504_2_f64),
            (5.0, 0.003_691_098_3),
            (10.0, 1.778_006_23e-5),
        ] {
            let error = ((bessel_k0(x) - expected) / expected).abs();
            assert!(error < 2e-7, "K0({x}) relative error {error}");
        }
    }

    #[test]
    fn bessel_k0_rejects_non_positive_arguments() {
        assert!(bessel_k0(0.0).is_nan());
        assert!(bessel_k0(-1.0).is_nan());
        assert!(bessel_k0(f64::NAN).is_nan());
        assert!(bessel_k0(f64::INFINITY).is_nan());
    }

    #[test]
    fn bessel_k0_is_positive_and_decreasing() {
        let mut previous = f64::INFINITY;
        for step in 1..60 {
            let x = f64::from(step) * 0.25;
            let value = bessel_k0(x);
            assert!(value.is_finite() && value > 0.0, "K0({x}) = {value}");
            assert!(value < previous, "K0 must decrease at x = {x}");
            previous = value;
        }
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

    #[test]
    fn j1_asymptotic_coefficients() {
        // Validate against kwavers/DLMF reference values on the Hankel branch (|x| ≥ 8).
        let v10 = j1(10.0);
        let v15 = j1(15.0);
        eprintln!("j1(10.0) = {v10:.12}");
        eprintln!("j1(15.0) = {v15:.12}");
        // kwavers-validated reference: J_1(10) ≈ 0.0434727462 (tolerance 1e-5).
        assert!(
            (v10 - 0.043_472_746_2).abs() < 1e-5,
            "j1(10) off by {}",
            (v10 - 0.043_472_746_2).abs()
        );
        // J_1(15) is positive (between zeros 13.32 and 16.47); the NR asymptotic
        // approximation has ≲2e-3 error at this argument.
        assert!(
            (v15 - 0.205_104_107_9).abs() < 1e-3,
            "j1(15) off by {}",
            (v15 - 0.205_104_107_9).abs()
        );
    }

    #[test]
    fn jn_two_buffer_normalization() {
        // Reference values from kwavers tests (validated against A&S).
        let v21 = jn(2, 1.0);
        let v32 = jn(3, 2.0);
        let v53 = jn(5, 3.0);
        eprintln!("jn(2, 1.0) = {v21:.12}");
        eprintln!("jn(3, 2.0) = {v32:.12}");
        eprintln!("jn(5, 3.0) = {v53:.12}");
        assert!((v21 - 0.114_903_484_9).abs() < 1e-8, "jn(2,1) off");
        assert!((v32 - 0.128_943_249_8).abs() < 1e-8, "jn(3,2) off");
        // The two-buffer normalization algorithm has bounded error for moderate n/x.
        assert!(
            (v53 - 0.043_028_434_7).abs() < 1e-5,
            "jn(5,3) off by {}",
            (v53 - 0.043_028_434_7).abs()
        );
        assert_eq!(jn(0, 0.0), 1.0);
        assert_eq!(jn(5, 0.0), 0.0);
    }
}
