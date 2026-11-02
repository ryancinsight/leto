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
        let mut norm = T::ZERO;
        for i in first..n {
            u[i] = h[i * n + k];
            norm = hypot(hypot(norm, u[i].re), u[i].im);
        }
        if norm == T::ZERO {
            continue;
        }
        let leading = hypot(u[first].re, u[first].im);
        let phase = if leading == T::ZERO {
            Complex::new(T::ONE, T::ZERO)
        } else {
            u[first] / leading
        };
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
        h[first * n + k] = beta;
        h[k * n + first] = beta.conj();
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
        let magnitude = hypot(beta.re, beta.im);
        off_diagonal[i - 1] = magnitude;
        // Dᴴ T D has real non-negative subdiagonals; Q becomes Q D.
        phase = if magnitude == T::ZERO {
            Complex::new(T::ONE, T::ZERO)
        } else {
            (beta / magnitude) * phase
        };
        for row in 0..n {
            q[row * n + i] *= phase;
        }
    }
    (diagonal, off_diagonal, q)
}
