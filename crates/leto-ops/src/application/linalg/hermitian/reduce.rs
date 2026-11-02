//! Unitary Householder reduction, followed by diagonal phase normalization.
//!
//! `Qᴴ A Q = T`, with real symmetric tridiagonal `T`. Reflectors are
//! `H = I - 2 u uᴴ`, `uᴴu = 1`; the rank-two update is `A -= u wᴴ + w uᴴ`,
//! where `w = 2 A u - 2 (uᴴ A u) u`. Unlike a real-sign reflector, the phase
//! of the leading complex entry prevents cancellation when forming `u`.
//! Reference: LAPACK ZHETD2's Hermitian similarity/rank-two update contract.

use crate::application::linalg::scaling::hypot;
use crate::domain::real::RealScalar;
use eunomia::Complex;

fn phase_and_norm<T: RealScalar>(z: Complex<T>) -> (Complex<T>, T) {
    let scale = if z.re.abs() > z.im.abs() {
        z.re.abs()
    } else {
        z.im.abs()
    };
    if scale == T::ZERO {
        return (Complex::new(T::ONE, T::ZERO), T::ZERO);
    }
    // Normalize before taking the norm: rounding a subnormal norm first can
    // turn (min_subnormal, min_subnormal) into the non-unit phase (1, 1).
    let unit = z / scale;
    let norm = hypot(unit.re, unit.im);
    (unit / norm, norm.mul(scale))
}

/// Reduce a finite, safely scaled Hermitian matrix in row-major storage.
/// Return the real diagonal/off-diagonal and the unitary factor as columns.
pub(super) fn tridiagonalize<T: RealScalar>(
    mut h: Vec<Complex<T>>,
    n: usize,
) -> (Vec<T>, Vec<T>, Vec<Complex<T>>) {
    let zero = Complex::new(T::ZERO, T::ZERO);
    let two = T::from_count(2);
    let mut q = vec![zero; n * n];
    for i in 0..n {
        q[i * n + i] = Complex::new(T::ONE, T::ZERO);
    }
    let mut u = vec![zero; n];
    let mut w = vec![zero; n];
    for k in 0..n.saturating_sub(2) {
        let first = k + 1;
        let mut scale = T::ZERO;
        for i in first..n {
            u[i] = h[i * n + k];
            for component in [u[i].re.abs(), u[i].im.abs()] {
                if component > scale {
                    scale = component;
                }
            }
        }
        if scale == T::ZERO {
            continue;
        }
        let phase = phase_and_norm(u[first]).0;
        for value in &mut u[first..] {
            *value /= scale;
        }
        let norm = u[first..]
            .iter()
            .fold(T::ZERO, |acc, z| hypot(hypot(acc, z.re), z.im));
        // Form the reflector at column scale, even when the complete matrix
        // is normal but this column contains only subnormal couplings.
        let beta = -phase * norm;
        u[first] -= beta;
        let unorm = u[first..]
            .iter()
            .fold(T::ZERO, |acc, z| hypot(hypot(acc, z.re), z.im));
        for value in &mut u[first..] {
            *value /= unorm;
        }

        for i in first..n {
            w[i] = zero;
            for j in first..n {
                w[i] += h[i * n + j] * u[j];
            }
        }
        let alpha = (first..n)
            .map(|i| (u[i].conj() * w[i]).re)
            .fold(T::ZERO, |acc, value| acc.add(value));
        for i in first..n {
            w[i] = (w[i] - u[i] * alpha) * two;
        }
        for i in first..n {
            for j in first..=i {
                let value = h[i * n + j] - u[i] * w[j].conj() - w[i] * u[j].conj();
                h[i * n + j] = if i == j {
                    Complex::new(value.re, T::ZERO)
                } else {
                    value
                };
                h[j * n + i] = h[i * n + j].conj();
            }
        }
        for i in 0..n {
            let dot = (first..n)
                .map(|j| q[i * n + j] * u[j])
                .fold(zero, |acc, z| acc + z);
            for j in first..n {
                q[i * n + j] -= (dot * two) * u[j].conj();
            }
        }
        h[first * n + k] = beta * scale;
        h[k * n + first] = beta.conj() * scale;
        for i in first + 1..n {
            h[i * n + k] = zero;
            h[k * n + i] = zero;
        }
    }

    let diagonal = (0..n).map(|i| h[i * n + i].re).collect();
    let mut off_diagonal = vec![T::ZERO; n];
    let mut phase = Complex::new(T::ONE, T::ZERO);
    for i in 1..n {
        let beta = h[i * n + i - 1];
        let (edge_phase, magnitude) = phase_and_norm(beta);
        off_diagonal[i - 1] = magnitude;
        // Dᴴ T D has real non-negative subdiagonals; Q becomes Q D.
        phase = if magnitude == T::ZERO {
            Complex::new(T::ONE, T::ZERO)
        } else {
            edge_phase * phase
        };
        for row in 0..n {
            q[row * n + i] *= phase;
        }
    }
    (diagonal, off_diagonal, q)
}
