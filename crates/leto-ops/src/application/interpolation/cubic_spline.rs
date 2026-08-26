//! Natural cubic spline interpolation.
//!
//! Minimises bending energy ∫[S''(x)]² dx subject to S(xᵢ) = yᵢ.
//!
//! **Error bound** (Burden & Faires): for f ∈ C⁴(\[a,b\]) and mesh step h,
//! ‖f − S‖_∞ ≤ (5/384) h⁴ ‖f⁽⁴⁾‖_∞.
//!
//! ## Reference
//! Burden & Faires, *Numerical Analysis*, §3.5.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::{check_bounds, search::find_interval, validate_nodes, Interpolation1D};
use eunomia::{FloatElement, RealField};
use leto::{LetoError, Result};

#[inline]
fn f64_as<T: FloatElement>(v: f64) -> T {
    T::from_f64(v)
}

/// Natural cubic spline interpolation.
#[derive(Debug, Clone)]
pub struct CubicSplineInterpolation<T: RealField + Copy> {
    x_data: Vec<T>,
    a: Vec<T>, // y values at nodes
    b: Vec<T>, // first-derivative coefficients
    c: Vec<T>, // second-derivative / 2
    d: Vec<T>, // third-derivative / 6
}

impl<T: RealField + FloatElement + Copy> CubicSplineInterpolation<T> {
    /// Construct a natural cubic spline from node data.
    ///
    /// # Errors
    /// - [`LetoError::InvalidInput`] if arrays have unequal length.
    /// - [`LetoError::InvalidInput`] if fewer than 3 points.
    /// - [`LetoError::InvalidInput`] if `x_data` is not strictly increasing.
    pub fn new(x_data: Vec<T>, y_data: Vec<T>) -> Result<Self> {
        if x_data.len() != y_data.len() {
            return Err(LetoError::InvalidInput(
                "x_data and y_data must have equal length".into(),
            ));
        }
        validate_nodes(&x_data, 3)?;
        let (a, b, c, d) = compute_coefficients(&x_data, &y_data)?;
        Ok(Self { x_data, a, b, c, d })
    }
}

#[allow(clippy::type_complexity)]
fn compute_coefficients<T: RealField + FloatElement + Copy>(
    x: &[T],
    y: &[T],
) -> Result<(Vec<T>, Vec<T>, Vec<T>, Vec<T>)> {
    let n = x.len();
    let nm1 = n - 1;
    let zero = f64_as::<T>(0.0);
    let three = f64_as::<T>(3.0);
    let two = f64_as::<T>(2.0);

    let a = y.to_vec();
    let h: Vec<T> = (0..nm1).map(|i| x[i + 1] - x[i]).collect();
    let mut alpha = vec![zero; nm1];
    for i in 1..nm1 {
        alpha[i] = three * (a[i + 1] - a[i]) / h[i] - three * (a[i] - a[i - 1]) / h[i - 1];
    }

    // Thomas algorithm for tridiagonal system.
    let mut l = vec![zero; n];
    let mut mu = vec![zero; n];
    let mut z = vec![zero; n];
    l[0] = f64_as(1.0);
    for i in 1..nm1 {
        l[i] = two * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        if l[i] == zero {
            return Err(LetoError::NumericalBreakdown(
                "cubic spline: singular tridiagonal system".into(),
            ));
        }
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }
    l[nm1] = f64_as(1.0);

    let mut c = vec![zero; n];
    let mut b = vec![zero; nm1];
    let mut d = vec![zero; nm1];
    for j in (0..nm1).rev() {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + two * c[j]) / three;
        d[j] = (c[j + 1] - c[j]) / (three * h[j]);
    }
    // Drop the last c (boundary condition, not a coefficient of the piecewise polynomial).
    c.truncate(nm1);
    Ok((a, b, c, d))
}

impl<T: RealField + FloatElement + Copy> Interpolation1D<T> for CubicSplineInterpolation<T> {
    fn interpolate(&self, x: T) -> Result<T> {
        let n = self.x_data.len();
        check_bounds(x, self.x_data[0], self.x_data[n - 1])?;
        let i = find_interval(&self.x_data, &x)
            .ok_or_else(|| LetoError::InvalidInput("query outside data range".into()))?;
        let t = x - self.x_data[i];
        Ok(self.a[i] + t * (self.b[i] + t * (self.c[i] + t * self.d[i])))
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
    fn reproduces_linear_function() {
        let xs: Vec<f64> = (0..=5).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.clone();
        let s = CubicSplineInterpolation::new(xs, ys).unwrap();
        for xi in [0.3_f64, 1.7, 4.5] {
            assert!((s.interpolate(xi).unwrap() - xi).abs() < 1e-10, "x={xi}");
        }
    }

    #[test]
    fn nodes_reproduced_exactly() {
        let xs = vec![0.0_f64, 1.0, 2.0, 3.0];
        let ys = vec![0.0_f64, 1.0, 4.0, 9.0];
        let s = CubicSplineInterpolation::new(xs.clone(), ys.clone()).unwrap();
        for (&xi, &yi) in xs.iter().zip(ys.iter()) {
            assert!((s.interpolate(xi).unwrap() - yi).abs() < 1e-10, "xi={xi}");
        }
    }
}
