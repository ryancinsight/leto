//! Contract tests for the arbitrary-order staggered pair.
//!
//! The oracles are analytical throughout: published stencil coefficients, the
//! measured order of accuracy against an exactly differentiable field, the
//! adjoint identity that makes the leapfrog conservative, and the wall closure
//! the reflection is supposed to impose.

#![expect(
    clippy::unwrap_used,
    reason = "test code: unwraps on fixed fixtures are the floor's test exemption"
)]

use leto::Array3;

use super::super::coefficients::{
    central_first_derivative_coefficients, staggered_first_derivative_coefficients,
    TapCoefficients, MAX_HALF_ORDER,
};
use super::super::{FiniteDifference3D, FiniteDifference3DScheme};
use super::{Axis, StaggeredLeapfrog3D};

const AXES: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

/// Deterministic, non-separable test field: no axis is a constant multiple of
/// another, so an adjointness failure on one axis cannot hide behind another.
fn seeded(shape: [usize; 3], salt: f64) -> Array3<f64> {
    let mut field = Array3::zeros(shape);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let x = i as f64 * 0.37 + salt;
                let y = j as f64 * 0.53 - salt * 0.5;
                let z = k as f64 * 0.71 + salt * 0.25;
                field[[i, j, k]] = (x.sin() * y.cos() + z.sin() * 0.75) * (1.0 + 0.1 * salt);
            }
        }
    }
    field
}

fn dot(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
    a.as_slice()
        .unwrap()
        .iter()
        .zip(b.as_slice().unwrap())
        .fold(0.0, |sum, (&a, &b)| sum + a * b)
}

mod coefficients;
mod construction;
mod kernels;
mod sweeps;
