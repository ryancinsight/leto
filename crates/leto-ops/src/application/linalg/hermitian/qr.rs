//! Hermitian reduction and the shared implicit-shift tridiagonal iteration.

use super::{complex_eye, sort_eig, HermitianEigenConfig, HermitianEigenResult, C64};
use leto::{Array1, Array2, LetoError, Result};

pub(super) fn decompose(
    a: &Array2<C64>,
    config: HermitianEigenConfig,
) -> Result<HermitianEigenResult> {
    let n = a.shape()[0];
    if !config.tolerance.is_finite() || config.tolerance < 0.0 {
        return Err(LetoError::InvalidInput(
            "Hermitian QR tolerance must be finite and non-negative".into(),
        ));
    }
    let mut parts = Vec::with_capacity(2 * n * n);
    let mut off_norm = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let z = a[[i, j]];
            if !z.re.is_finite() || !z.im.is_finite() {
                return Err(LetoError::InvalidInput(format!(
                    "Hermitian QR input has a non-finite entry at ({i}, {j})"
                )));
            }
            if i == j && z.im != 0.0 {
                return Err(LetoError::InvalidInput(format!(
                    "Hermitian QR diagonal is not real at ({i}, {i})"
                )));
            }
            // Resolve accepted roundoff asymmetry by the lower triangle.
            let z = if j > i { a[[j, i]].conj() } else { z };
            parts.extend([z.re, z.im]);
            if i != j {
                off_norm = off_norm.hypot(z.re).hypot(z.im);
            }
        }
    }
    if off_norm <= config.tolerance {
        let values = Array1::from_shape_vec([n], (0..n).map(|i| a[[i, i]].re).collect())?;
        return finish(values, complex_eye(n), 0, off_norm, config);
    }
    let solution = super::kernel::solve(parts, n, config.tolerance, config.max_iterations)?;
    finish(
        Array1::from_shape_vec([n], solution.values)?,
        Array2::from_shape_vec([n, n], solution.vectors)?,
        solution.iterations,
        solution.discarded_norm,
        config,
    )
}

fn finish(
    values: Array1<f64>,
    vectors: Array2<C64>,
    iterations: usize,
    off_diagonal_norm: f64,
    config: HermitianEigenConfig,
) -> Result<HermitianEigenResult> {
    let (values, vectors) = if config.sort_descending {
        sort_eig(values, vectors, true)
    } else {
        (values, vectors)
    };
    let condition_number = if config.estimate_condition && !values.is_empty() {
        let smallest = values.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);
        let largest = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if smallest > 1e-14 {
            Some(largest / smallest)
        } else {
            None
        }
    } else {
        None
    };
    Ok(HermitianEigenResult {
        eigenvalues: values,
        eigenvectors: vectors,
        iterations,
        off_diagonal_norm,
        condition_number,
    })
}
