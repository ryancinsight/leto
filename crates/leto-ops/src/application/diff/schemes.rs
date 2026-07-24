//! Finite-difference scheme enumeration.

/// Available finite-difference stencil families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FiniteDifferenceScheme {
    /// First-order forward: `f'(x) ≈ (f(x+h) − f(x)) / h`
    Forward,
    /// First-order backward: `f'(x) ≈ (f(x) − f(x−h)) / h`
    Backward,
    /// Second-order central (default): `f'(x) ≈ (f(x+h) − f(x−h)) / (2h)`
    #[default]
    Central,
    /// Second-order one-sided forward:
    /// `f'(x) ≈ (−3f(x) + 4f(x+h) − f(x+2h)) / (2h)`
    ForwardSecondOrder,
    /// Second-order one-sided backward:
    /// `f'(x) ≈ (f(x−2h) − 4f(x−h) + 3f(x)) / (2h)`
    BackwardSecondOrder,
}
