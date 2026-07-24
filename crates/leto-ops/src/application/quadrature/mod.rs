//! Numerical quadrature (integration) rules.
//!
//! SSOT for generic quadrature in the Atlas simulation stack.
//! All rules are generic over `T: RealField + FloatElement + Copy`.
//!
//! ## Rules
//!
//! | Type | Order | Points |
//! |------|-------|--------|
//! | [`TrapezoidalRule`] | 2 | 2 |
//! | [`SimpsonsRule`] | 4 | 3 |
//! | [`GaussLegendre2`] | 4 (exact for deg ≤ 3) | 2 |
//! | [`GaussLegendre3`] | 6 (exact for deg ≤ 5) | 3 |
//! | [`GaussLegendre5`] | 10 (exact for deg ≤ 9) | 5 |
//! | [`CompositeQuadrature`] | variable | n panels |

use eunomia::{FloatElement, NumericElement, RealField};

#[inline]
fn f<T: FloatElement>(v: f64) -> T {
    T::from_f64(v)
}

// ── Quadrature trait ──────────────────────────────────────────────────────────

/// Quadrature rule for integrating `∫_a^b f(x) dx`.
pub trait Quadrature<T: RealField + FloatElement + Copy>: Send + Sync {
    /// Integrate `f` over `[a, b]` using this rule.
    fn integrate<F: Fn(T) -> T>(&self, f: F, a: T, b: T) -> T;
    /// Polynomial degree of exactness.
    fn order(&self) -> usize;
    /// Number of function evaluations per panel.
    fn num_points(&self) -> usize;
}

// ── Simple rules ──────────────────────────────────────────────────────────────

/// Trapezoidal rule: exact for polynomials of degree ≤ 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrapezoidalRule;

impl<T: RealField + FloatElement + Copy> Quadrature<T> for TrapezoidalRule {
    fn integrate<F: Fn(T) -> T>(&self, g: F, a: T, b: T) -> T {
        (b - a) * (g(a) + g(b)) / f(2.0)
    }
    fn order(&self) -> usize {
        2
    }
    fn num_points(&self) -> usize {
        2
    }
}

/// Simpson's rule: exact for polynomials of degree ≤ 3.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpsonsRule;

impl<T: RealField + FloatElement + Copy> Quadrature<T> for SimpsonsRule {
    fn integrate<F: Fn(T) -> T>(&self, g: F, a: T, b: T) -> T {
        let mid = (a + b) / f(2.0);
        (b - a) * (g(a) + f::<T>(4.0) * g(mid) + g(b)) / f(6.0)
    }
    fn order(&self) -> usize {
        4
    }
    fn num_points(&self) -> usize {
        3
    }
}

// ── Gauss-Legendre rules ──────────────────────────────────────────────────────

/// Gauss-Legendre 2-point rule (exact for deg ≤ 3).
///
/// Nodes on [-1,1]: ±1/√3.  Weights: 1, 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct GaussLegendre2;

impl<T: RealField + FloatElement + Copy> Quadrature<T> for GaussLegendre2 {
    fn integrate<F: Fn(T) -> T>(&self, g: F, a: T, b: T) -> T {
        const XI: f64 = 0.577_350_269_189_626; // 1/√3
        let half = f::<T>(0.5);
        let mid = (a + b) * half;
        let hw = (b - a) * half;
        let t: T = f(XI);
        hw * (g(mid - hw * t) + g(mid + hw * t))
    }
    fn order(&self) -> usize {
        4
    }
    fn num_points(&self) -> usize {
        2
    }
}

/// Gauss-Legendre 3-point rule (exact for deg ≤ 5).
///
/// Nodes: 0, ±√(3/5).  Weights: 8/9, 5/9, 5/9.
#[derive(Debug, Clone, Copy, Default)]
pub struct GaussLegendre3;

impl<T: RealField + FloatElement + Copy> Quadrature<T> for GaussLegendre3 {
    fn integrate<F: Fn(T) -> T>(&self, g: F, a: T, b: T) -> T {
        const XI: f64 = 0.774_596_669_241_483; // √(3/5)
        let half = f::<T>(0.5);
        let mid = (a + b) * half;
        let hw = (b - a) * half;
        let t: T = f(XI);
        let w1: T = f(8.0 / 9.0);
        let w2: T = f(5.0 / 9.0);
        hw * (w1 * g(mid) + w2 * g(mid - hw * t) + w2 * g(mid + hw * t))
    }
    fn order(&self) -> usize {
        6
    }
    fn num_points(&self) -> usize {
        3
    }
}

/// Gauss-Legendre 5-point rule (exact for deg ≤ 9).
#[derive(Debug, Clone, Copy, Default)]
pub struct GaussLegendre5;

impl<T: RealField + FloatElement + Copy> Quadrature<T> for GaussLegendre5 {
    fn integrate<F: Fn(T) -> T>(&self, g: F, a: T, b: T) -> T {
        // Nodes and weights from Abramowitz & Stegun Table 25.4.
        const XI: [f64; 5] = [
            0.906_179_845_938_664,
            0.538_469_310_105_683,
            0.000_000_000_000_000,
            -0.538_469_310_105_683,
            -0.906_179_845_938_664,
        ];
        const W: [f64; 5] = [
            0.236_926_885_056_189,
            0.478_628_670_499_366,
            0.568_888_888_888_889,
            0.478_628_670_499_366,
            0.236_926_885_056_189,
        ];
        let half = f::<T>(0.5);
        let mid = (a + b) * half;
        let hw = (b - a) * half;
        let mut s = <T as NumericElement>::ZERO;
        for k in 0..5 {
            s += f::<T>(W[k]) * g(mid + hw * f::<T>(XI[k]));
        }
        hw * s
    }
    fn order(&self) -> usize {
        10
    }
    fn num_points(&self) -> usize {
        5
    }
}

// ── Composite rule ────────────────────────────────────────────────────────────

/// Composite quadrature: applies a base rule to each of `n_panels` uniform panels.
///
/// **Theorem** (composite p-point Gauss, Burden & Faires §4.4): for
/// f ∈ C^{2p}([a,b]) the error is O(h^{2p}) where h = (b−a)/n.
#[derive(Debug, Clone)]
pub struct CompositeQuadrature<Q> {
    base: Q,
    n_panels: usize,
}

impl<Q> CompositeQuadrature<Q> {
    /// Wrap `base` rule with `n_panels` subdivisions.
    #[must_use]
    pub fn new(base: Q, n_panels: usize) -> Self {
        assert!(n_panels >= 1, "n_panels must be ≥ 1");
        Self { base, n_panels }
    }
}

impl<T: RealField + FloatElement + Copy, Q: Quadrature<T>> Quadrature<T>
    for CompositeQuadrature<Q>
{
    fn integrate<F: Fn(T) -> T>(&self, g: F, a: T, b: T) -> T {
        let n: T = f(self.n_panels as f64);
        let h = (b - a) / n;
        let mut s = <T as NumericElement>::ZERO;
        for k in 0..self.n_panels {
            let x0 = a + h * f::<T>(k as f64);
            let x1 = x0 + h;
            s += self.base.integrate(&g, x0, x1);
        }
        s
    }
    fn order(&self) -> usize {
        self.base.order()
    }
    fn num_points(&self) -> usize {
        self.base.num_points() * self.n_panels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn gauss2_exact_for_degree3() {
        let g = GaussLegendre2;
        // ∫_0^1 x³ dx = 0.25
        let v: f64 = g.integrate(|x: f64| x.powi(3), 0.0, 1.0);
        assert!((v - 0.25).abs() < 1e-12);
    }

    #[test]
    fn composite_gauss3_converges_for_sin() {
        let rule = CompositeQuadrature::new(GaussLegendre3, 100);
        let v: f64 = rule.integrate(|x: f64| x.sin(), 0.0, std::f64::consts::PI);
        assert!((v - 2.0).abs() < 1e-12);
    }
}
