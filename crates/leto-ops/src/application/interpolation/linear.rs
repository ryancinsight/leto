//! Piecewise-linear interpolation.
//!
//! For each query `x` in `[x_data[i], x_data[i+1]]`:
//!
//! ```text
//! y = y_data[i] + (y_data[i+1] - y_data[i]) * (x - x_data[i]) / (x_data[i+1] - x_data[i])
//! ```
//!
//! ## Reference
//! Burden & Faires, *Numerical Analysis*, §3.1.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::{check_bounds, search::find_interval, validate_nodes, Interpolation1D};
use eunomia::RealField;
use leto::{LetoError, Result};

/// Piecewise-linear interpolation.
#[derive(Debug, Clone)]
pub struct LinearInterpolation<T: RealField + Copy> {
    x_data: Vec<T>,
    y_data: Vec<T>,
}

impl<T: RealField + Copy> LinearInterpolation<T> {
    /// Construct from sorted node vectors.
    ///
    /// # Errors
    /// - [`LetoError::InvalidInput`] if `x_data.len() != y_data.len()`.
    /// - [`LetoError::InvalidInput`] if fewer than 2 points are supplied.
    /// - [`LetoError::InvalidInput`] if `x_data` is not strictly increasing.
    pub fn new(x_data: Vec<T>, y_data: Vec<T>) -> Result<Self> {
        if x_data.len() != y_data.len() {
            return Err(LetoError::InvalidInput(
                "x_data and y_data must have equal length".into(),
            ));
        }
        validate_nodes(&x_data, 2)?;
        Ok(Self { x_data, y_data })
    }
}

impl<T: RealField + Copy> Interpolation1D<T> for LinearInterpolation<T> {
    fn interpolate(&self, x: T) -> Result<T> {
        let n = self.x_data.len();
        check_bounds(x, self.x_data[0], self.x_data[n - 1])?;
        let i = find_interval(&self.x_data, &x)
            .ok_or_else(|| LetoError::InvalidInput("query outside data range".into()))?;
        let dx = self.x_data[i + 1] - self.x_data[i];
        let t = (x - self.x_data[i]) / dx;
        Ok(self.y_data[i] + t * (self.y_data[i + 1] - self.y_data[i]))
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
    fn midpoint_linear() {
        let interp =
            LinearInterpolation::new(vec![0.0_f64, 1.0, 2.0], vec![0.0, 1.0, 4.0]).unwrap();
        assert!((interp.interpolate(0.5_f64).unwrap() - 0.5).abs() < 1e-12);
        assert!((interp.interpolate(1.5_f64).unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn at_nodes() {
        let interp =
            LinearInterpolation::new(vec![0.0_f64, 1.0, 2.0], vec![0.0, 1.0, 4.0]).unwrap();
        assert!((interp.interpolate(0.0_f64).unwrap() - 0.0).abs() < 1e-12);
        assert!((interp.interpolate(2.0_f64).unwrap() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn extrapolation_rejected() {
        let interp = LinearInterpolation::new(vec![0.0_f64, 1.0], vec![0.0, 1.0]).unwrap();
        assert!(interp.interpolate(-0.1_f64).is_err());
        assert!(interp.interpolate(1.1_f64).is_err());
    }

    #[test]
    fn duplicate_nodes_rejected() {
        assert!(LinearInterpolation::new(vec![0.0_f64, 1.0, 1.0], vec![0.0, 1.0, 2.0]).is_err());
    }
}
