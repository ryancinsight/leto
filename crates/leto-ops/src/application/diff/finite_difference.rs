//! Generic 1-D finite-difference operator.

use super::schemes::FiniteDifferenceScheme;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, LetoError, Result};

#[inline]
fn f<T: FloatElement>(v: f64) -> T {
    T::from_f64(v)
}

/// Generic finite-difference operator for 1-D slice data.
///
/// Computes the first derivative `f'(x)` from a uniformly-sampled slice
/// `values` with grid spacing `spacing`.  All stencils keep the output length
/// equal to the input length by falling back to one-sided differences near the
/// boundary.
#[derive(Debug, Clone, Copy)]
pub struct FiniteDifference<T: RealField + Copy> {
    scheme: FiniteDifferenceScheme,
    spacing: T,
}

impl<T: RealField + FloatElement + Copy> FiniteDifference<T> {
    /// Create a finite-difference operator with the given scheme and spacing.
    #[must_use]
    pub fn new(scheme: FiniteDifferenceScheme, spacing: T) -> Self {
        Self { scheme, spacing }
    }

    /// Central-difference operator (2nd-order, default).
    #[must_use]
    pub fn central(spacing: T) -> Self {
        Self::new(FiniteDifferenceScheme::Central, spacing)
    }

    /// Forward-difference operator (1st-order).
    #[must_use]
    pub fn forward(spacing: T) -> Self {
        Self::new(FiniteDifferenceScheme::Forward, spacing)
    }

    /// Backward-difference operator (1st-order).
    #[must_use]
    pub fn backward(spacing: T) -> Self {
        Self::new(FiniteDifferenceScheme::Backward, spacing)
    }

    /// Return the active finite-difference scheme.
    #[must_use]
    pub fn scheme(&self) -> FiniteDifferenceScheme {
        self.scheme
    }

    /// Return the grid spacing.
    #[must_use]
    pub fn spacing(&self) -> T {
        self.spacing
    }

    /// Compute the first derivative `f'` of `values`.
    ///
    /// # Errors
    /// - [`LetoError::InvalidInput`] when fewer than 2 points are supplied.
    /// - [`LetoError::InvalidInput`] when the 2nd-order schemes need fewer than 3 points.
    pub fn first_derivative(&self, values: &[T]) -> Result<Array1<T>> {
        if values.len() < 2 {
            return Err(LetoError::InvalidInput(
                "Need at least 2 points for differentiation".into(),
            ));
        }
        let n = values.len();
        let inv_h = <T as NumericElement>::ONE / self.spacing;
        let two_h = f::<T>(2.0) * self.spacing;
        let mut out = Array1::zeros([n]);

        match self.scheme {
            FiniteDifferenceScheme::Forward => {
                for i in 0..n - 1 {
                    out[i] = (values[i + 1] - values[i]) * inv_h;
                }
                // Last point falls back to backward.
                out[n - 1] = (values[n - 1] - values[n - 2]) * inv_h;
            }
            FiniteDifferenceScheme::Backward => {
                // First point falls back to forward.
                out[0] = (values[1] - values[0]) * inv_h;
                for i in 1..n {
                    out[i] = (values[i] - values[i - 1]) * inv_h;
                }
            }
            FiniteDifferenceScheme::Central => {
                // First and last fall back to first-order one-sided.
                out[0] = (values[1] - values[0]) * inv_h;
                for i in 1..n - 1 {
                    out[i] = (values[i + 1] - values[i - 1]) / two_h;
                }
                if n > 1 {
                    out[n - 1] = (values[n - 1] - values[n - 2]) * inv_h;
                }
            }
            FiniteDifferenceScheme::ForwardSecondOrder => {
                if n < 3 {
                    return Err(LetoError::InvalidInput(
                        "Need at least 3 points for 2nd-order forward difference".into(),
                    ));
                }
                let (_two, three, four) = (f::<T>(2.0), f::<T>(3.0), f::<T>(4.0));
                for i in 0..n - 2 {
                    out[i] = (-three * values[i] + four * values[i + 1] - values[i + 2]) / two_h;
                }
                // Last two fall back to central.
                out[n - 2] = (values[n - 1] - values[n - 3]) / two_h;
                out[n - 1] = (values[n - 1] - values[n - 2]) * inv_h;
            }
            FiniteDifferenceScheme::BackwardSecondOrder => {
                if n < 3 {
                    return Err(LetoError::InvalidInput(
                        "Need at least 3 points for 2nd-order backward difference".into(),
                    ));
                }
                let (_two, three, four) = (f::<T>(2.0), f::<T>(3.0), f::<T>(4.0));
                // First two fall back to central.
                out[0] = (values[1] - values[0]) * inv_h;
                out[1] = (values[2] - values[0]) / two_h;
                for i in 2..n {
                    out[i] = (values[i - 2] - four * values[i - 1] + three * values[i]) / two_h;
                }
            }
        }
        Ok(out)
    }

    /// Compute the second derivative `f''` using a central difference stencil.
    ///
    /// Interior: `f''(x) ≈ (f(x+h) − 2f(x) + f(x−h)) / h²`
    ///
    /// # Errors
    /// [`LetoError::InvalidInput`] when fewer than 3 points are supplied.
    pub fn second_derivative(&self, values: &[T]) -> Result<Array1<T>> {
        if values.len() < 3 {
            return Err(LetoError::InvalidInput(
                "Need at least 3 points for second derivative".into(),
            ));
        }
        let n = values.len();
        let h_sq = self.spacing * self.spacing;
        let two = f::<T>(2.0);
        let mut out = Array1::zeros([n]);

        // Interior: central difference.
        for i in 1..n - 1 {
            out[i] = (values[i + 1] - two * values[i] + values[i - 1]) / h_sq;
        }
        // Boundary: one-sided second differences.
        out[0] = (values[2] - two * values[1] + values[0]) / h_sq;
        out[n - 1] = (values[n - 1] - two * values[n - 2] + values[n - 3]) / h_sq;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
            .collect()
    }

    #[test]
    fn central_diff_of_sin_is_cos() {
        let xs = linspace(0.0, std::f64::consts::PI, 1000);
        let vals: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
        let h = xs[1] - xs[0];
        let op = FiniteDifference::central(h);
        let dv = op.first_derivative(&vals).unwrap();
        // Interior should match cos(x) to O(h²).
        for i in 10..990 {
            let expected = xs[i].cos();
            assert!(
                (dv[i] - expected).abs() < 1e-4,
                "i={i} dv={} exp={expected}",
                dv[i]
            );
        }
    }

    #[test]
    fn second_derivative_of_quadratic_is_constant() {
        // f(x) = x²  →  f''(x) = 2
        let vals: Vec<f64> = (0..20).map(|i| (i as f64).powi(2)).collect();
        let op = FiniteDifference::central(1.0);
        let d2v = op.second_derivative(&vals).unwrap();
        for i in 1..19 {
            assert!((d2v[i] - 2.0).abs() < 1e-10, "i={i}");
        }
    }

    #[test]
    fn too_few_points_rejected() {
        let op = FiniteDifference::central(0.1);
        assert!(op.first_derivative(&[1.0]).is_err());
        assert!(op.second_derivative(&[1.0, 2.0]).is_err());
    }
}
