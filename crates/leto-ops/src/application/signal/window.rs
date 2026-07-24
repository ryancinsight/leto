//! Canonical window-coefficient functions.
//!
//! Each function takes a **normalized symmetric position** `x = i/(N−1) ∈ [0, 1]`
//! and returns the window weight at that position.
//!
//! Hosting these here (the `leto-ops` foundation layer) lets all Atlas consumers
//! (kwavers, helios, CFDrs) evaluate identical formulas without duplication.
//!
//! # References
//! - Harris, F.J. (1978). "On the Use of Windows for Harmonic Analysis with the
//!   Discrete Fourier Transform". *Proc. IEEE* 66(1):51–83.

use std::f64::consts::PI;

const TWO_PI: f64 = 2.0 * PI;
const FOUR_PI: f64 = 4.0 * PI;

/// Hann (von Hann) window: `w(x) = 0.5·(1 − cos(2πx))`.
///
/// Zero at both endpoints; unity at the centre (`x = 0.5`).
/// Good general-purpose choice: first sidelobe ≈ −31 dB.
#[inline]
#[must_use]
pub fn hann(x: f64) -> f64 {
    0.5 * (1.0 - (TWO_PI * x).cos())
}

/// Hamming window: `w(x) = 0.54 − 0.46·cos(2πx)`.
///
/// Non-zero pedestal (≈ 0.08) at the endpoints; first sidelobe ≈ −43 dB.
#[inline]
#[must_use]
pub fn hamming(x: f64) -> f64 {
    0.46f64.mul_add(-(TWO_PI * x).cos(), 0.54)
}

/// Blackman window: `w(x) = 0.42 − 0.5·cos(2πx) + 0.08·cos(4πx)`.
///
/// Near-zero at the endpoints; sidelobes ≈ −58 dB.
#[inline]
#[must_use]
pub fn blackman(x: f64) -> f64 {
    0.08f64.mul_add(
        (FOUR_PI * x).cos(),
        0.5f64.mul_add(-(TWO_PI * x).cos(), 0.42),
    )
}

/// Tukey (tapered-cosine) window with cosine fraction `r ∈ [0, 1]`.
///
/// Cosine tapers over the outer `r/2` fraction of each end; flat unity inside.
///
/// - `r = 0` → rectangular window (`w ≡ 1`).
/// - `r = 1` → Hann window.
///
/// For `x = i/(N−1) ∈ [0, 1]`:
/// ```text
/// w(x) = 0.5·(1 + cos((2π/r)·(x − r/2)))         for x < r/2
///      = 1                                        for r/2 ≤ x ≤ 1 − r/2
///      = 0.5·(1 + cos((2π/r)·(x − 1 + r/2)))      for x > 1 − r/2
/// ```
#[inline]
#[must_use]
pub fn tukey(x: f64, r: f64) -> f64 {
    let r = r.clamp(0.0, 1.0);
    if r == 0.0 {
        return 1.0;
    }
    let half = 0.5 * r;
    if x < half {
        0.5 * (1.0 + (TWO_PI / r * (x - half)).cos())
    } else if x <= 1.0 - half {
        1.0
    } else {
        0.5 * (1.0 + (TWO_PI / r * (x - 1.0 + half)).cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_endpoints_and_centre() {
        assert!((hann(0.0)).abs() < 1e-12);
        assert!((hann(1.0)).abs() < 1e-12);
        assert!((hann(0.5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn hamming_pedestal_and_centre() {
        assert!((hamming(0.0) - 0.08).abs() < 1e-12);
        assert!((hamming(0.5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn blackman_endpoints_and_centre() {
        assert!((blackman(0.0)).abs() < 1e-12);
        assert!((blackman(0.5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn tukey_degenerates_to_rectangular_and_hann() {
        for &x in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!((tukey(x, 0.0) - 1.0).abs() < 1e-12, "r=0 at x={x}");
        }
        for k in 0..=10 {
            let x = k as f64 / 10.0;
            assert!(
                (tukey(x, 1.0) - hann(x)).abs() < 1e-12,
                "r=1 vs hann at x={x}"
            );
        }
    }

    #[test]
    fn tukey_taper_and_flat_top() {
        let r = 0.5;
        assert!((tukey(0.0, r)).abs() < 1e-12);
        assert!((tukey(1.0, r)).abs() < 1e-12);
        assert!((tukey(0.25, r) - 1.0).abs() < 1e-12);
        assert!((tukey(0.5, r) - 1.0).abs() < 1e-12);
        assert!((tukey(0.1, r) - tukey(0.9, r)).abs() < 1e-12);
    }
}
