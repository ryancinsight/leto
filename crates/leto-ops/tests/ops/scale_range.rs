//! Exponent sweeps for the dense factorizations: `s·A` for every power of two
//! `s` from the smallest subnormal to three binades below the largest finite
//! value, in every supported format, must factor correctly — never a wrong
//! `Ok`, and a typed error only where a failure is recorded below.
//!
//! # Derived bounds
//!
//! Work in units of `s`. Let `Â = image / s` be what `T` actually holds
//! (exact in `f64`), `δ = ‖Â − A‖_F` the input rounding (zero wherever `s·A`
//! is representable; nonzero only in the subnormal binades), and `n²·ε(T)·‖Â‖_F`
//! the factorization's backward error (Higham 2002, Lemma 19.3: `O(n)`
//! orthogonal transformations of `γ̃_n` each). Restoring a result by `s` rounds
//! it once onto `T`'s grid; in the subnormal binades that costs at most half
//! the subnormal spacing, `0.5·2^(MIN−e)·ε` in units of `s`.
//!
//! - Singular values (Weyl): `|σ̂ − σ| ≤ δ + n²ε‖Â‖_F + 0.5·2^(MIN−e)·ε`.
//! - Eigenvalues of `A = S·diag(1, 2, 4)·S⁻¹` (Bauer–Fike):
//!   `|λ̂ − λ| ≤ κ(S)·(δ + n²ε‖Â‖_F) + 0.5·2^(MIN−e)·ε`, `κ(S) ≤ ‖S‖_F·‖S⁻¹‖_F = 3.68`.
//! - Pivoted QR: `‖Â·P − Q·R̂‖_F ≤ n²ε‖Â‖_F + n·0.5·2^(MIN−e)·ε`, and the rank
//!   is 3 whenever `δ < σ_min(A)/2`.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::format::{epsilon, Format};
use eunomia::{Bf16, F16};
use leto::{Array2, LetoError, Storage};
use leto_ops::{
    col_piv_qr, eigenvalues, schur, singular_values, svd_decompose, symmetric_eigen_qr,
};

const N: usize = 3;

/// A general matrix with distinct singular values; every entry is exact in
/// every supported format.
const GENERAL: [f64; 9] = [4.0, 1.0, 0.5, 1.0, 3.0, 1.0, 0.25, 1.0, 2.0];

/// `S·diag(1, 2, 4)·S⁻¹` with `S = [[1,1,0],[0,1,1],[1,0,1]]`,
/// `S⁻¹ = ½[[1,−1,1],[1,1,−1],[−1,1,1]]`: eigenvalues `{1, 2, 4}`, entries
/// multiples of `½` below 3 in magnitude.
const SIMILAR: [f64; 9] = [1.5, 0.5, -0.5, -1.0, 3.0, 1.0, -1.5, 1.5, 2.5];
const SIMILAR_EIGENVALUES: [f64; 3] = [1.0, 2.0, 4.0];
/// `‖S‖_F·‖S⁻¹‖_F = √6 · 1.5`.
const SIMILAR_CONDITION: f64 = 3.675;

fn frobenius(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// `x·2^k` in `f64`, exact for every `k` a supported format produces.
fn scale(x: f64, k: i32) -> f64 {
    let half = k / 2;
    x * 2.0_f64.powi(half) * 2.0_f64.powi(k - half)
}

/// One binade: `s = 2^e`, the matrix `T` holds, and its image in units of `s`.
struct Sample<T> {
    matrix: Array2<T>,
    unit: Vec<f64>,
    input_error: f64,
    backward: f64,
    grid: f64,
}

fn sample<T: Format>(base: &[f64], exponent: i32) -> Sample<T> {
    let s = T::ONE.scale_binary(exponent);
    let values: Vec<T> = base.iter().map(|&v| T::from_f64(v).mul(s)).collect();
    let unit: Vec<f64> = values
        .iter()
        .map(|v| scale(v.to_f64(), -exponent))
        .collect();
    let difference: Vec<f64> = unit.iter().zip(base).map(|(a, b)| a - b).collect();
    let epsilon = epsilon::<T>();
    Sample {
        matrix: Array2::from_shape_vec([N, N], values).unwrap(),
        input_error: frobenius(&difference),
        backward: (N * N) as f64 * epsilon * frobenius(&unit),
        grid: 0.5 * scale(1.0, T::MIN_EXPONENT - exponent) * epsilon,
        unit,
    }
}

/// Exponents from the smallest subnormal to `MAX − 3` (entries up to 4·s
/// stay finite, results up to ~6·s too).
fn exponents<T: Format>() -> std::ops::RangeInclusive<i32> {
    (T::MIN_EXPONENT - T::PRECISION + 1)..=(T::MAX_EXPONENT - 3)
}

/// Singular values of `GENERAL` by an independent route: `√λ(AᵀA)` through
/// the symmetric solver, in `f64`, descending.
fn general_singular_values() -> Vec<f64> {
    let mut gram = vec![0.0; N * N];
    for i in 0..N {
        for j in 0..N {
            gram[i * N + j] = (0..N)
                .map(|k| GENERAL[k * N + i] * GENERAL[k * N + j])
                .sum();
        }
    }
    let eigen = symmetric_eigen_qr(&Array2::from_shape_vec([N, N], gram).unwrap().view()).unwrap();
    eigen.eigenvalues.iter().rev().map(|l| l.sqrt()).collect()
}

fn check_singular_values<T: Format>() {
    let reference = general_singular_values();
    for exponent in exponents::<T>() {
        let sample = sample::<T>(&GENERAL, exponent);
        let bound = sample.input_error + sample.backward + sample.grid;
        let values = singular_values(&sample.matrix.view())
            .unwrap_or_else(|error| panic!("2^{exponent}: {error}"));
        let full = svd_decompose(&sample.matrix.view())
            .unwrap_or_else(|error| panic!("2^{exponent}: {error}"));
        for sigmas in [values, full.singular_values] {
            for (sigma, expected) in sigmas.iter().zip(&reference) {
                let sigma = scale(sigma.to_f64(), -exponent);
                assert!(
                    (sigma - expected).abs() <= bound,
                    "2^{exponent}: σ {sigma} vs {expected}, bound {bound:e}"
                );
            }
        }
    }
}

/// Real parts ascending, imaginary parts, in units of `s`.
fn sorted_unit<T: Format>(values: &[leto::Complex<T>], exponent: i32) -> Vec<(f64, f64)> {
    let mut parts: Vec<(f64, f64)> = values
        .iter()
        .map(|z| {
            (
                scale(z.re.to_f64(), -exponent),
                scale(z.im.to_f64(), -exponent),
            )
        })
        .collect();
    parts.sort_by(|a, b| a.0.total_cmp(&b.0));
    parts
}

/// A recorded failure: F16's Francis iteration stagnates on nonsymmetric input
/// at every scale (`LETO-F16-FRANCIS-2026-09-24`), so a typed
/// non-convergence is accepted there and nowhere else.
fn francis_may_fail<T: Format>() -> bool {
    T::PRECISION == 11
}

fn check_eigenvalues<T: Format>() {
    for exponent in exponents::<T>() {
        let sample = sample::<T>(&SIMILAR, exponent);
        let bound = SIMILAR_CONDITION * (sample.input_error + sample.backward) + sample.grid;
        let results = [
            eigenvalues(&sample.matrix.view()),
            schur(&sample.matrix.view()).map(|decomposition| decomposition.eigenvalues()),
        ];
        for result in results {
            match result {
                Ok(values) => {
                    for ((re, im), expected) in sorted_unit(&values, exponent)
                        .into_iter()
                        .zip(SIMILAR_EIGENVALUES)
                    {
                        assert!(
                            (re - expected).abs() <= bound && im.abs() <= bound,
                            "2^{exponent}: λ {re}+{im}i vs {expected}, bound {bound:e}"
                        );
                    }
                }
                Err(LetoError::StorageError { .. }) if francis_may_fail::<T>() => {}
                Err(error) => panic!("2^{exponent}: {error}"),
            }
        }
    }
}

fn check_pivoted_qr<T: Format>() {
    let sigma_min = *general_singular_values().last().unwrap();
    for exponent in exponents::<T>() {
        let sample = sample::<T>(&GENERAL, exponent);
        let decomposition = col_piv_qr(&sample.matrix.view())
            .unwrap_or_else(|error| panic!("2^{exponent}: {error}"));
        if sample.input_error < sigma_min / 2.0 {
            assert_eq!(decomposition.rank(), N, "2^{exponent}: rank");
        }
        let q = decomposition.q();
        let r = decomposition.r();
        let (q, r) = (q.storage().as_slice(), r.storage().as_slice());
        let permutation = decomposition.permutation();
        let mut residual = 0.0;
        for i in 0..N {
            for (k, &column) in permutation.iter().enumerate() {
                // R in units of s before multiplying, so no f64 product is subnormal.
                let qr: f64 = (0..N)
                    .map(|j| q[i * N + j].to_f64() * scale(r[j * N + k].to_f64(), -exponent))
                    .sum();
                residual += (sample.unit[i * N + column] - qr).powi(2);
            }
        }
        let bound = sample.backward + N as f64 * sample.grid;
        assert!(
            residual.sqrt() <= bound,
            "2^{exponent}: ‖ÂP − QR‖ {} > {bound:e}",
            residual.sqrt()
        );
    }
}

#[test]
fn singular_values_are_correct_across_each_exponent_range() {
    check_singular_values::<f64>();
    check_singular_values::<f32>();
    check_singular_values::<F16>();
    check_singular_values::<Bf16>();
}

#[test]
fn eigenvalues_and_schur_are_correct_across_each_exponent_range() {
    check_eigenvalues::<f64>();
    check_eigenvalues::<f32>();
    check_eigenvalues::<F16>();
    check_eigenvalues::<Bf16>();
}

#[test]
fn pivoted_qr_is_correct_across_each_exponent_range() {
    check_pivoted_qr::<f64>();
    check_pivoted_qr::<f32>();
    check_pivoted_qr::<F16>();
    check_pivoted_qr::<Bf16>();
}
