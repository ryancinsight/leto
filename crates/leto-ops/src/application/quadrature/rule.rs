//! The quadrature rule contract every rule implements.

use eunomia::RealField;

/// Quadrature rule for integrating `∫_a^b f(x) dx`.
pub trait Quadrature<T: RealField>: Send + Sync {
    /// Integrate `f` over `[a, b]` using this rule.
    fn integrate<F: Fn(T) -> T>(&self, f: F, a: T, b: T) -> T;
    /// Polynomial degree of exactness.
    fn order(&self) -> usize;
    /// Number of function evaluations per panel.
    fn num_points(&self) -> usize;
}

/// Convert a `f64` literal into the rule's scalar type.
///
/// The rule coefficients are published as `f64` tables, so every rule needs
/// the same widening step; keeping it here means one conversion site rather
/// than one per rule.
#[inline]
pub(super) fn f<T: eunomia::FloatElement>(v: f64) -> T {
    T::from_f64(v)
}
