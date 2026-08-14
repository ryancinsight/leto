//! Barycentric Lagrange interpolation.
//!
//! Computes the unique polynomial of degree ≤ n−1 through n nodes using the
//! second barycentric form (numerically stable, O(n) per evaluation after O(n)
//! setup).
//!
//! **Warning** (Runge phenomenon): for equidistant nodes and n > ~20, the
//! interpolant can oscillate wildly near the endpoints.  Prefer Chebyshev
//! nodes for high-degree accuracy.
//!
//! ## Reference
//! Berrut & Trefethen (2004). *Barycentric Lagrange Interpolation*, SIAM Rev.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::{check_bounds, validate_nodes, Interpolation1D};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{LetoError, Result};

/// Barycentric Lagrange interpolation.
#[derive(Debug, Clone)]
pub struct LagrangeInterpolation<T: RealField + Copy> {
    x_data: Vec<T>,
    y_data: Vec<T>,
    /// Barycentric weights: w[i] = 1 / ∏_{j≠i}(x[i] − x[j])
    weights: Vec<T>,
}

impl<T: RealField + FloatElement + Copy> LagrangeInterpolation<T> {
    /// Construct from node data.  Computes barycentric weights in O(n²).
    ///
    /// # Errors
    /// Same as [`LinearInterpolation::new`].
    pub fn new(x_data: Vec<T>, y_data: Vec<T>) -> Result<Self> {
        if x_data.len() != y_data.len() {
            return Err(LetoError::InvalidInput(
                "x_data and y_data must have equal length".into(),
            ));
        }
        validate_nodes(&x_data, 2)?;
        let weights = barycentric_weights(&x_data);
        Ok(Self {
            x_data,
            y_data,
            weights,
        })
    }
}

fn barycentric_weights<T: RealField + FloatElement + Copy>(x: &[T]) -> Vec<T> {
    let n = x.len();
    let mut w = vec![T::from_f64(1.0); n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                w[i] *= x[i] - x[j];
            }
        }
        w[i] = T::from_f64(1.0) / w[i];
    }
    w
}

impl<T: RealField + FloatElement + Copy> Interpolation1D<T> for LagrangeInterpolation<T> {
    fn interpolate(&self, x: T) -> Result<T> {
        let n = self.x_data.len();
        check_bounds(x, self.x_data[0], self.x_data[n - 1])?;

        // Check for exact node match (avoids division by zero).
        for (i, &xi) in self.x_data.iter().enumerate() {
            if (x - xi).abs() < T::EPSILON {
                return Ok(self.y_data[i]);
            }
        }

        // Second barycentric form: p(x) = (Σ w[i]·y[i]/(x−x[i])) / (Σ w[i]/(x−x[i]))
        let mut num = <T as NumericElement>::ZERO;
        let mut den = <T as NumericElement>::ZERO;
        for i in 0..n {
            let diff = x - self.x_data[i];
            let term = self.weights[i] / diff;
            num += term * self.y_data[i];
            den += term;
        }
        if den == <T as NumericElement>::ZERO {
            return Err(LetoError::NumericalBreakdown(
                "Lagrange denominator is zero".into(),
            ));
        }
        Ok(num / den)
    }

    fn bounds(&self) -> (T, T) {
        let n = self.x_data.len();
        (self.x_data[0], self.x_data[n - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_exact() {
        // f(x) = x² + x + 1 through nodes 0, 1, 2.
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![1.0, 3.0, 7.0]; // 0²+0+1, 1+1+1, 4+2+1
        let interp = LagrangeInterpolation::new(xs, ys).unwrap();
        assert!((interp.interpolate(0.5).unwrap() - 1.75).abs() < 1e-10);
    }

    #[test]
    fn nodes_exact() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![1.0, 3.0, 7.0];
        let interp = LagrangeInterpolation::new(xs.clone(), ys.clone()).unwrap();
        for (&xi, &yi) in xs.iter().zip(ys.iter()) {
            assert!((interp.interpolate(xi).unwrap() - yi).abs() < 1e-10);
        }
    }
}
