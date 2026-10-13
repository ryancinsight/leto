//! The Bessel-function operations.
//!
//! The J0/J1/K0 markers evaluate the one generic core on
//! [`eunomia::RealField`]: `j0`/`j1` route through `libm` (native precision
//! at `f64`), `k0` evaluates the Abramowitz & Stegun 9.8.5/9.8.6 program in
//! lane precision. The GPU dialect macros keep their own coefficients —
//! shader code cannot call Rust — and the lanewise tests check them against
//! the scalar oracles.

use super::contract::UnaryOp;
use crate::domain::RealScalar;
use eunomia::RealField;

/// Bessel function of the first kind J0 operation marker.
///
/// Zero-sized: evaluates [`RealField::bessel_j0`] in lane precision
/// (libm-routed). This is no longer the same program as the scalar
/// [`j0`](crate::application::special::j0), which keeps the
/// Numerical-Recipes rational/Hankel form as the device-shader oracle; the
/// two agree within the retired program's documented error (see the
/// lanewise oracle note in `tests/ops/unary_math/special.rs`).
#[derive(Clone, Copy, Debug, Default)]
pub struct J0Op;
impl<T: RealScalar + RealField> UnaryOp<T> for J0Op {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        x.bessel_j0()
    }
}
/// Bessel function of the first kind J1 operation marker.
///
/// Zero-sized: evaluates [`RealField::bessel_j1`] in lane precision
/// (libm-routed). Same scalar-oracle relationship as [`J0Op`].
#[derive(Clone, Copy, Debug, Default)]
pub struct J1Op;
impl<T: RealScalar + RealField> UnaryOp<T> for J1Op {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        x.bessel_j1()
    }
}
/// Modified Bessel function of the second kind K0 operation marker.
///
/// Zero-sized: evaluates [`RealField::bessel_k0`] in lane precision. This is
/// the elementwise counterpart of
/// [`bessel_k0`](crate::application::special::bessel_k0), evaluating the
/// same nested formulation in the lane precision.
#[derive(Clone, Copy, Debug, Default)]
pub struct K0Op;
impl<T: RealScalar + RealField> UnaryOp<T> for K0Op {
    #[inline(always)]
    fn apply(&self, x: T) -> T {
        x.bessel_k0()
    }
}
