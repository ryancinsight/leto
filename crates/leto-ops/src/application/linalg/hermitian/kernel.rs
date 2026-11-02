//! Native-precision Hermitian reduction and eigenvector composition.

use super::reduce;
use crate::application::linalg::{
    scaling::{self, GateBound},
    symmetric_qr::ql,
};
use crate::domain::real::RealScalar;
use eunomia::Complex;
use leto::{LetoError, Result};

pub(super) struct Solution<T> {
    pub values: Vec<T>,
    pub vectors: Vec<Complex<T>>,
    pub iterations: usize,
    pub discarded_norm: T,
}

/// `parts` holds alternating real/imaginary components of a finite Hermitian
/// matrix in row-major order. All arithmetic stays in the caller's scalar.
pub(super) fn solve<T: RealScalar>(
    mut parts: Vec<T>,
    n: usize,
    tolerance: T,
    budget: usize,
) -> Result<Solution<T>> {
    // The shared QL correction has degree two. The existing power-of-two
    // gate also bounds the reduction's dimension-dependent sums.
    let exponent = scaling::gate_exponent(&parts, 2, |values, largest| {
        GateBound::factor(2 * scaling::norm_ratio_log2(values, largest))
    })?
    .unwrap_or(0);
    scaling::scale_by_power_of_two(&mut parts, -exponent);
    let h = parts
        .chunks_exact(2)
        .map(|z| Complex::new(z[0], z[1]))
        .collect();
    let (mut values, mut off, basis) = reduce::tridiagonalize(h, n);
    let mut vectors = vec![T::ZERO; n * n];
    for i in 0..n {
        vectors[i * n + i] = T::ONE;
    }
    let mut scaled_tolerance = [tolerance];
    scaling::scale_by_power_of_two(&mut scaled_tolerance, -exponent);
    let iteration = match ql::diagonalize_absolute(
        &mut values,
        &mut off,
        &mut vectors,
        n,
        budget,
        scaled_tolerance[0],
    ) {
        Ok(iteration) => iteration,
        Err(LetoError::ConvergenceError {
            max_iters,
            residual,
            ..
        }) => {
            // LetoError stores diagnostics as f64; the numerical iteration
            // remains in T. Undo input scaling for the diagnostic as well.
            let mut residual = [residual];
            scaling::restore(
                &mut residual,
                exponent,
                "Hermitian QR residual exceeds the diagnostic range",
            )?;
            return Err(LetoError::ConvergenceError {
                max_iters,
                residual: residual[0],
                tol: tolerance.to_f64(),
            });
        }
        Err(error) => return Err(error),
    };
    if values.iter().chain(vectors.iter()).any(|v| !v.is_finite()) {
        return Err(LetoError::NumericalBreakdown(
            "Hermitian tridiagonal iteration produced a non-finite result".into(),
        ));
    }
    scaling::restore(
        &mut values,
        exponent,
        "Hermitian QR eigenvalue exceeds the scalar range",
    )?;
    let mut remainder = [iteration.discarded_norm];
    scaling::restore(
        &mut remainder,
        exponent,
        "Hermitian QR residual exceeds the scalar range",
    )?;
    let zero = Complex::new(T::ZERO, T::ZERO);
    let mut result = vec![zero; n * n];
    for row in 0..n {
        for col in 0..n {
            let mut dot = zero;
            for k in 0..n {
                dot += basis[row * n + k] * vectors[col * n + k];
            }
            result[row * n + col] = dot;
        }
    }
    Ok(Solution {
        values,
        vectors: result,
        iterations: iteration.iterations,
        discarded_norm: remainder[0],
    })
}
