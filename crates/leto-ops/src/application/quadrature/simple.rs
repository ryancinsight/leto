//! Closed-form quadrature rules with compile-time coefficients.

use eunomia::RealField;

use super::rule::{f, Quadrature};

/// Trapezoidal rule: exact for polynomials of degree ≤ 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrapezoidalRule;

impl<T: RealField> Quadrature<T> for TrapezoidalRule {
    fn integrate<F: Fn(T) -> T>(&self, g: F, a: T, b: T) -> T {
        (b - a) * (g(a) + g(b)) / f(2.0)
    }
    fn order(&self) -> usize {
        1
    }
    fn num_points(&self) -> usize {
        2
    }
}

/// Simpson's rule: exact for polynomials of degree ≤ 3.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpsonsRule;

impl<T: RealField> Quadrature<T> for SimpsonsRule {
    fn integrate<F: Fn(T) -> T>(&self, g: F, a: T, b: T) -> T {
        let mid = (a + b) / f(2.0);
        (b - a) * (g(a) + f::<T>(4.0) * g(mid) + g(b)) / f(6.0)
    }
    fn order(&self) -> usize {
        3
    }
    fn num_points(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::{Quadrature, SimpsonsRule, TrapezoidalRule};

    #[test]
    fn trapezoidal_integrates_line() {
        let t = TrapezoidalRule;
        let v: f64 = t.integrate(|x| x, 0.0, 1.0);
        assert!((v - 0.5).abs() < 1e-12);
    }

    #[test]
    fn simpson_integrates_cubic() {
        let s = SimpsonsRule;
        let v: f64 = s.integrate(|x: f64| x.powi(3), 0.0, 2.0);
        assert!((v - 4.0).abs() < 1e-10);
    }

    /// `order()` is the polynomial degree of exactness (Burden & Faires):
    /// trapezoid exact through degree 1, Simpson through degree 3.
    #[test]
    fn order_matches_exactness_degree() {
        assert_eq!(Quadrature::<f64>::order(&TrapezoidalRule), 1);
        assert_eq!(Quadrature::<f64>::order(&SimpsonsRule), 3);
    }
}
