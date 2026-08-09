//! Canonical phase-angle utilities.
//!
//! Phase differences must be compared on the circle, where `θ` and `θ ± 2πk`
//! are identical. This module is the SSOT for wrapping to the principal interval,
//! used by FWI misfit functions, PINN phase losses, and beamforming.

use std::f64::consts::PI;

const TWO_PI: f64 = 2.0 * PI;

/// Wrap a phase angle to the principal interval `(−π, π]`.
///
/// For any finite `θ`, `wrap_to_pi(θ) ≡ θ (mod 2π)` and the result lies in
/// `(−π, π]`. The implementation uses `rem_euclid` (branch-free, exact for
/// finite inputs) rather than iterative `±2π` subtraction.
#[must_use]
#[inline]
pub fn wrap_to_pi(theta: f64) -> f64 {
    let r = theta.rem_euclid(TWO_PI);
    if r > PI { r - TWO_PI } else { r }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_into_principal_interval() {
        for k in -5..=5_i32 {
            for &base in &[-3.0, -1.0, 0.0, 0.5, 2.9, PI - 0.01] {
                let theta = base + TWO_PI * f64::from(k);
                let w = wrap_to_pi(theta);
                assert!(
                    w > -PI - 1e-9 && w <= PI + 1e-9,
                    "wrap_to_pi({theta}) = {w} outside (-π, π]"
                );
                let diff = (w - theta) / TWO_PI;
                assert!(
                    (diff - diff.round()).abs() < 1e-9,
                    "wrap changed angle modulo 2π: {theta} -> {w}"
                );
            }
        }
    }

    #[test]
    fn fixed_points_and_boundaries() {
        assert!((wrap_to_pi(0.0)).abs() < 1e-12);
        assert!((wrap_to_pi(0.5) - 0.5).abs() < 1e-12);
        assert!((wrap_to_pi(-0.5) + 0.5).abs() < 1e-12);
        assert!((wrap_to_pi(PI) - PI).abs() < 1e-9);
        assert!((wrap_to_pi(-PI) - PI).abs() < 1e-9);
        assert!((wrap_to_pi(TWO_PI - 0.1) + 0.1).abs() < 1e-9);
    }
}
