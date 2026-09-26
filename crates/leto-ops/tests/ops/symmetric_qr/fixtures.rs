//! Shared fixtures: seeded symmetric matrices, rounding into `T`, and the
//! a-priori and a-posteriori spectrum checks.

use super::super::backward_error::{gamma, informative, ql, ql_vectors, symmetric_certificate};
use super::super::format::epsilon;
use leto::{Array2, Storage};
use leto_ops::{RealScalar, SymmetricEigenDecomposition, SymmetricEigenWorkspace, Xorshift64};

/// Round `values` into `T`, returning the `T` matrix and the exact `f64`
/// image of what `T` holds.
pub(super) fn round_into<T: RealScalar>(values: &[f64], n: usize) -> (Array2<T>, Vec<f64>) {
    let narrow: Vec<T> = values.iter().map(|&v| T::from_f64(v)).collect();
    let image = narrow.iter().map(|v| v.to_f64()).collect();
    (Array2::from_shape_vec([n, n], narrow).unwrap(), image)
}

/// Seeded symmetric matrix with entries uniform in `[-1, 1)`.
pub(super) fn random_symmetric(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Xorshift64::new(seed);
    let mut values = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let value = 2.0 * rng.next_unit_f64() - 1.0;
            values[i * n + j] = value;
            values[j * n + i] = value;
        }
    }
    values
}

/// `Q·diag(spectrum)·Qᵀ` with `Q` the Householder reflector of a seeded
/// vector: an exact orthogonal similarity in real arithmetic, so the spectrum
/// is known.
pub(super) fn with_spectrum(spectrum: &[f64], seed: u64) -> Vec<f64> {
    let n = spectrum.len();
    let mut rng = Xorshift64::new(seed);
    let v: Vec<f64> = (0..n).map(|_| rng.next_unit_f64() - 0.5).collect();
    let scale = 2.0 / v.iter().map(|x| x * x).sum::<f64>();
    let q = |i: usize, j: usize| f64::from(u8::from(i == j)) - scale * v[i] * v[j];
    let mut values = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            values[i * n + j] = (0..n).map(|k| q(i, k) * spectrum[k] * q(j, k)).sum();
        }
    }
    // Exact symmetry: the sums above differ in rounding between (i, j) and (j, i).
    for i in 0..n {
        for j in 0..i {
            values[j * n + i] = values[i * n + j];
        }
    }
    values
}

/// Gram matrix `XᵀX` of a seeded `rows × n` matrix: rank `min(rows, n)`.
pub(super) fn random_gram(rows: usize, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Xorshift64::new(seed);
    let x: Vec<f64> = (0..rows * n).map(|_| rng.next_unit_f64() - 0.5).collect();
    let mut gram = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            gram[i * n + j] = (0..rows).map(|r| x[r * n + i] * x[r * n + j]).sum();
        }
    }
    gram
}

fn frobenius(values: &[f64]) -> f64 {
    let largest = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if largest == 0.0 {
        return 0.0;
    }
    largest
        * values
            .iter()
            .map(|v| (v / largest).powi(2))
            .sum::<f64>()
            .sqrt()
}

/// `η·‖A‖_F`, the derived a-priori eigenvalue bound (module
/// documentation), where informative.
pub(super) fn backward_bound<T: RealScalar>(values: &[f64], n: usize) -> Option<f64> {
    informative(ql(n, epsilon::<T>())).map(|eta| eta * frobenius(values))
}

/// The a-posteriori radius of a returned decomposition of the order-`n`
/// matrix whose image is `image`: every eigenvalue within it of the matching
/// (ascending) eigenvalue of `image` ([`symmetric_certificate`] on the
/// returned eigenvectors).
pub(super) fn certified<T: RealScalar>(
    image: &[f64],
    eigen: &SymmetricEigenDecomposition<T>,
) -> f64 {
    let n = eigen.eigenvalues.len();
    let values: Vec<f64> = eigen.eigenvalues.iter().map(|v| v.to_f64()).collect();
    let basis: Vec<f64> = eigen
        .eigenvectors
        .storage()
        .as_slice()
        .iter()
        .map(|v| v.to_f64())
        .collect();
    symmetric_certificate(image, &values, &basis, n)
}

/// Decompose `values` in `T`; assert every eigenpair's residual and the
/// eigenvectors' orthonormality against the derived bounds where they are
/// informative; return the eigenvalues as `f64` with their a-posteriori
/// radius: each within it of the matching (ascending) eigenvalue of the
/// image `Â` ([`symmetric_certificate`], measured on the returned pairs).
pub(super) fn assert_backward_stable<T: RealScalar>(values: &[f64], n: usize) -> (Vec<f64>, f64) {
    let (matrix, image) = round_into::<T>(values, n);
    let mut workspace = SymmetricEigenWorkspace::new();
    workspace.decompose(&matrix.view()).unwrap();
    let eps = epsilon::<T>();
    let (eta, eta_q) = (ql(n, eps), ql_vectors(n, eps));
    let vectors: Vec<Vec<f64>> = workspace
        .eigenvectors()
        .map(|v| v.iter().map(|x| x.to_f64()).collect())
        .collect();
    let lambdas: Vec<f64> = workspace.eigenvalues().iter().map(|x| x.to_f64()).collect();
    assert_eq!(vectors.len(), n);
    assert!(lambdas.windows(2).all(|pair| pair[0] <= pair[1]));
    let mut basis = vec![0.0; n * n];
    for (j, v) in vectors.iter().enumerate() {
        for (i, x) in v.iter().enumerate() {
            basis[i * n + j] = *x;
        }
    }
    let radius = symmetric_certificate(&image, &lambdas, &basis, n);
    if let Some(eta) = informative(eta + 2.0 * eta_q * (1.0 + eta)) {
        let bound = (eta + 2.0 * gamma(n as f64 + 2.0, f64::EPSILON)) * frobenius(&image);
        for (&lambda, v) in lambdas.iter().zip(&vectors) {
            let residual = (0..n)
                .map(|i| {
                    let av: f64 = (0..n).map(|j| image[i * n + j] * v[j]).sum();
                    (av - lambda * v[i]).powi(2)
                })
                .sum::<f64>()
                .sqrt();
            assert!(residual <= bound, "residual {residual:e} exceeds {bound:e}");
        }
    }
    if let Some(eta_o) = informative(2.0 * eta_q + eta_q * eta_q) {
        let orthogonality = eta_o + gamma(n as f64, f64::EPSILON);
        for (a, u) in vectors.iter().enumerate() {
            for (b, w) in vectors.iter().enumerate() {
                let dot: f64 = u.iter().zip(w).map(|(x, y)| x * y).sum();
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() <= orthogonality,
                    "v{a}·v{b} = {dot}, expected {expected}"
                );
            }
        }
    }
    (lambdas, radius)
}

/// Every computed eigenvalue matches `spectrum` (ascending) within its
/// a-posteriori radius, and within the a-priori backward bound where that is
/// informative, plus the rounding of the `f64` matrix into `T`.
pub(super) fn assert_spectrum<T: RealScalar>(values: &[f64], spectrum: &[f64]) {
    let n = spectrum.len();
    let (computed, radius) = assert_backward_stable::<T>(values, n);
    let rounding = epsilon::<T>() / 2.0 * frobenius(values);
    let a_priori = informative(ql(n, epsilon::<T>())).map(|eta| eta * frobenius(values));
    for (value, expected) in computed.iter().zip(spectrum) {
        assert!(
            (value - expected).abs() <= radius + rounding,
            "{value} vs {expected}, certified {:e}",
            radius + rounding
        );
        if let Some(bound) = a_priori {
            assert!(
                (value - expected).abs() <= bound + rounding,
                "{value} vs {expected}, bound {:e}",
                bound + rounding
            );
        }
    }
}
