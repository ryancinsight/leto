//! Composite panel quadrature over any base rule.

use eunomia::{NumericElement, RealField};

use super::rule::Quadrature;

/// Composite quadrature: applies a base rule to each of `n_panels` uniform panels.
///
/// **Theorem** (composite p-point Gauss, Burden & Faires §4.4): for
/// f ∈ C^{2p}(\[a,b\]) the error is O(h^{2p}) where h = (b−a)/n.
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

impl<T: RealField, Q: Quadrature<T>> Quadrature<T> for CompositeQuadrature<Q> {
    fn integrate<F: Fn(T) -> T>(&self, g: F, a: T, b: T) -> T {
        let n: T = T::from_count(self.n_panels);
        let h = (b - a) / n;
        let mut s = <T as NumericElement>::ZERO;
        for k in 0..self.n_panels {
            let x0 = a + h * T::from_count(k);
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
    use super::CompositeQuadrature;
    use crate::application::quadrature::GaussLegendre3;
    use crate::application::quadrature::Quadrature;

    #[test]
    fn composite_gauss3_converges_for_sin() {
        let rule = CompositeQuadrature::new(GaussLegendre3, 100);
        let v: f64 = rule.integrate(|x: f64| x.sin(), 0.0, std::f64::consts::PI);
        assert!((v - 2.0).abs() < 1e-12);
    }
}
