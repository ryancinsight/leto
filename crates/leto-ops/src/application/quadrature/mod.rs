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
//! | [`GaussLegendreN`] | 2n−1 (exact for deg ≤ 2n−1) | n (arbitrary) |
//! | [`CompositeQuadrature`] | variable | n panels |
//!
//! ## Arbitrary-order Gauss-Legendre
//!
//! [`gauss_legendre_nodes_weights`] computes n-point GL nodes and weights on
//! [-1, 1] via Newton iteration backed by `legendre_poly_and_deriv` (SSOT).
//! Weights satisfy `∑ wᵢ = 2` and are exact for polynomials up to degree 2n−1.
//! [`GaussLegendreN`] wraps these as a `Quadrature<f64>` rule.

use eunomia::{FloatElement, NumericElement, RealField};

use crate::application::special_legendre::legendre_poly_and_deriv;

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

// ── Pre-computed GL3 constant tables (nodes on [-1,1] and [0,1]) ──────────────

/// 3-point GL nodes on [−1, 1]: {−√(3/5), 0, √(3/5)}.
pub const GL3_NODES: [f64; 3] = [
    -0.774_596_669_241_483, // −√(3/5)
    0.000_000_000_000_000,  // 0
    0.774_596_669_241_483,  // +√(3/5)
];

/// 3-point GL weights on [−1, 1]: {5/9, 8/9, 5/9} (sum = 2).
pub const GL3_WEIGHTS: [f64; 3] = [
    0.555_555_555_555_556, // 5/9
    0.888_888_888_888_889, // 8/9
    0.555_555_555_555_556, // 5/9
];

/// 3-point GL nodes mapped to [0, 1]: {(1−√(3/5))/2, 1/2, (1+√(3/5))/2}.
///
/// Use this for integrals over unit-interval domains (e.g. Duffy-transformed BEM).
pub const GL3_NODES_UNIT: [f64; 3] = [
    0.112_701_665_379_258, // (1 − √(3/5))/2
    0.500_000_000_000_000, // 1/2
    0.887_298_334_620_742, // (1 + √(3/5))/2
];

/// 3-point GL weights on [0, 1]: {5/18, 4/9, 5/18} (sum = 1).
///
/// Corresponding weights for [`GL3_NODES_UNIT`].
pub const GL3_WEIGHTS_UNIT: [f64; 3] = [
    0.277_777_777_777_778, // 5/18
    0.444_444_444_444_444, // 4/9
    0.277_777_777_777_778, // 5/18
];

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

// ── Arbitrary-order Gauss-Legendre ────────────────────────────────────────────

/// Compute n-point Gauss-Legendre nodes (on [−1, 1]) and weights via Newton iteration.
///
/// Uses `legendre_poly_and_deriv` as the Legendre polynomial SSOT.
/// Nodes are symmetric about 0; weights sum to 2.  The rule is exact for
/// polynomials up to degree 2n − 1.
///
/// # Errors
///
/// Returns an error string if Newton iteration fails to converge for any root.
///
/// # Reference
///
/// Press et al., *Numerical Recipes* §4.6 (Gauss–Legendre quadrature).
pub fn gauss_legendre_nodes_weights(n: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    if n == 0 {
        return Err("Gauss-Legendre requires at least 1 point".to_owned());
    }
    let mut nodes = vec![0.0_f64; n];
    let mut weights = vec![0.0_f64; n];
    let half = (n + 1) / 2;
    const MAX_ITER: usize = 64;
    const TOL: f64 = 8.0 * f64::EPSILON;

    for i in 0..half {
        // Initial guess: roots of cos((π(i + 0.75)) / (n + 0.5))
        let mut x = (core::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut converged = false;
        for _ in 0..MAX_ITER {
            let (pn, dpn) = legendre_poly_and_deriv(n, x);
            if dpn.abs() < f64::MIN_POSITIVE {
                return Err(format!("Zero Legendre derivative at x={x} for n={n}"));
            }
            let dx = pn / dpn;
            x -= dx;
            if dx.abs() <= TOL * x.abs().max(1.0) {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(format!("Gauss-Legendre root {i} of {n} did not converge"));
        }
        let (_, dpn) = legendre_poly_and_deriv(n, x);
        let w = 2.0 / ((1.0 - x * x) * dpn * dpn);
        // Symmetric placement.
        nodes[i] = -x;
        nodes[n - 1 - i] = x;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    Ok((nodes, weights))
}

/// Arbitrary n-point Gauss-Legendre quadrature rule (exact for polynomials ≤ degree 2n−1).
///
/// Nodes and weights are computed once at construction via Newton iteration;
/// each `integrate` call is O(n) function evaluations.
///
/// For small fixed orders prefer the zero-cost [`GaussLegendre2`] / [`GaussLegendre3`] /
/// [`GaussLegendre5`] structs which use compile-time constants.
#[derive(Debug, Clone)]
pub struct GaussLegendreN {
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

impl GaussLegendreN {
    /// Construct an n-point rule.
    ///
    /// # Panics
    ///
    /// Panics if Newton iteration fails to converge (should not happen for n ≤ 1000).
    #[must_use]
    pub fn new(n: usize) -> Self {
        let (nodes, weights) =
            gauss_legendre_nodes_weights(n).expect("Gauss-Legendre node computation converged");
        Self { nodes, weights }
    }

    /// Try to construct an n-point rule, returning an error on convergence failure.
    pub fn try_new(n: usize) -> Result<Self, String> {
        let (nodes, weights) = gauss_legendre_nodes_weights(n)?;
        Ok(Self { nodes, weights })
    }

    /// Number of quadrature points.
    #[must_use]
    pub fn n(&self) -> usize {
        self.nodes.len()
    }

    /// Raw nodes on [−1, 1].
    #[must_use]
    pub fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    /// Raw weights (sum = 2).
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

impl Quadrature<f64> for GaussLegendreN {
    fn integrate<F: Fn(f64) -> f64>(&self, g: F, a: f64, b: f64) -> f64 {
        let half = 0.5 * (b - a);
        let mid = 0.5 * (a + b);
        self.nodes
            .iter()
            .zip(self.weights.iter())
            .map(|(&xi, &wi)| wi * g(mid + half * xi))
            .sum::<f64>()
            * half
    }

    fn order(&self) -> usize {
        2 * self.nodes.len() - 1
    }

    fn num_points(&self) -> usize {
        self.nodes.len()
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

    #[test]
    fn gauss_legendre_n_weights_sum_to_two() {
        for n in [2, 3, 5, 7, 10, 16] {
            let (_, weights) = gauss_legendre_nodes_weights(n).unwrap();
            let sum: f64 = weights.iter().sum();
            assert!((sum - 2.0).abs() < 1e-12, "n={n}: weight sum={sum}");
        }
    }

    #[test]
    fn gauss_legendre_n_nodes_symmetric() {
        for n in [2, 3, 5, 8] {
            let (nodes, _) = gauss_legendre_nodes_weights(n).unwrap();
            for i in 0..n {
                assert!((nodes[i] + nodes[n - 1 - i]).abs() < 1e-14, "n={n} i={i}");
            }
        }
    }

    #[test]
    fn gauss_legendre_n_exact_for_polynomials() {
        // n-point rule is exact for degree ≤ 2n − 1.
        for n in [3usize, 5, 7] {
            let rule = GaussLegendreN::new(n);
            let max_exact = 2 * n - 1;
            for d in 0..=max_exact {
                // ∫_{-1}^{1} x^d dx = 2/(d+1) for even d, 0 for odd d.
                let exact = if d % 2 == 0 { 2.0 / (d as f64 + 1.0) } else { 0.0 };
                let computed = rule.integrate(|x: f64| x.powi(d as i32), -1.0, 1.0);
                assert!(
                    (computed - exact).abs() < 1e-11,
                    "n={n} degree={d}: got {computed}, expected {exact}"
                );
            }
        }
    }

    #[test]
    fn gauss_legendre_n_matches_fixed_rules() {
        // 3-point GaussLegendreN must match GaussLegendre3 for sin.
        let fixed: f64 = GaussLegendre3.integrate(|x: f64| x.sin(), 0.0, 1.0);
        let dynamic: f64 = GaussLegendreN::new(3).integrate(|x: f64| x.sin(), 0.0, 1.0);
        assert!((fixed - dynamic).abs() < 1e-13, "fixed={fixed} dynamic={dynamic}");
    }
}
