//! Numerical quadrature (integration) rules.
//!
//! SSOT for generic quadrature in the Atlas simulation stack.
//! All rules are generic over `T: RealField`.
//!
//! ## Rules
//!
//! | Type | Order | Points |
//! |------|-------|--------|
//! | [Trapezoidal](crate::application::quadrature::TrapezoidalRule) | 2 | 2 |
//! | [Simpson](crate::application::quadrature::SimpsonsRule) | 4 | 3 |
//! | [Gauss-Legendre 2](crate::application::quadrature::GaussLegendre2) | 4 (exact for deg ≤ 3) | 2 |
//! | [Gauss-Legendre 3](crate::application::quadrature::GaussLegendre3) | 6 (exact for deg ≤ 5) | 3 |
//! | [Gauss-Legendre 5](crate::application::quadrature::GaussLegendre5) | 10 (exact for deg ≤ 9) | 5 |
//! | [Gauss-Legendre n](crate::application::quadrature::GaussLegendreN) | 2n−1 (exact for deg ≤ 2n−1) | n (arbitrary) |
//! | [Composite](crate::application::quadrature::CompositeQuadrature) | variable | n panels |
//!
//! ## Arbitrary-order Gauss-Legendre
//!
//! [`gauss_legendre_nodes_weights`](crate::application::quadrature::gauss_legendre_nodes_weights)
//! computes n-point GL nodes and weights on
//! [-1, 1] via Newton iteration backed by `legendre_poly_and_deriv` (SSOT).
//! Weights satisfy `∑ wᵢ = 2` and are exact for polynomials up to degree 2n−1.
//! [`GaussLegendreN`](crate::application::quadrature::GaussLegendreN) wraps these as a
//! `Quadrature<f64>` rule.
//!
//! Rules integrate caller closures, which cannot cross to device kernels,
//! so quadrature is host-side by design with no GPU counterpart (principled
//! asymmetry).
//!
//! # Submodules
//!
//! | Module | Contract |
//! | --- | --- |
//! | `rule` | The [`Quadrature`](crate::application::quadrature::Quadrature) trait every rule implements. |
//! | `simple` | Closed-form two- and three-point rules. |
//! | `gauss_legendre` | Fixed-order and arbitrary-order Gauss-Legendre rules. |
//! | `composite` | Panel composition over any base rule. |

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

mod composite;
mod gauss_legendre;
mod rule;
mod simple;

pub use composite::CompositeQuadrature;
pub use gauss_legendre::{
    gauss_legendre_nodes_weights, GaussLegendre2, GaussLegendre3, GaussLegendre5, GaussLegendreN,
    GL3_NODES, GL3_NODES_UNIT, GL3_WEIGHTS, GL3_WEIGHTS_UNIT,
};
pub use rule::Quadrature;
pub use simple::{SimpsonsRule, TrapezoidalRule};
