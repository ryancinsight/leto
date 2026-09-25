//! Exponent sweeps for the dense factorizations: `s·A` for every power of two
//! `s` from the smallest subnormal to three binades below the largest finite
//! value, in every supported format, must factor correctly — never a wrong
//! `Ok`, and never a typed error: the Francis and Golub–Kahan kernels form
//! their products scale-safely and the matrix-tier gate covers the rest
//! (`linalg::scaling`), so no exponent of any format is exempt.
//!
//! # Derived bounds
//!
//! Work in units of `s`. Let `Â = image / s` be what `T` actually holds
//! (exact in `f64`), `δ = ‖Â − A‖_F` the input rounding (zero wherever `s·A`
//! is representable; nonzero only in the subnormal binades and, for `Bf16`,
//! where an entry needs more than 8 significant bits), and `η·‖Â‖_F` the
//! factorization's backward error. `η` composes the per-transformation bounds
//! of Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.
//! (2002): a Householder reflector of length `m` applied to a column commits
//! `γ̃_m` (§19.3, Lemmas 19.2–19.3: `r` reflectors commit `r·γ̃_m`,
//! `γ̃_k = c·k·u/(1 − c·k·u)` with `c` a small integer constant Higham leaves
//! unspecified), and a Givens rotation `γ₆` (§19.6, Lemmas 19.7–19.8;
//! `u = ε/2`). The orthogonal transformation count is exact for the
//! reductions — `n − 2` Hessenberg reflectors, `2n − 1` bidiagonal ones — and
//! for the QR iterations it is sweeps × rotations per sweep: a Francis sweep
//! over an active block of order `k` applies `k − 1` reflectors of length
//! ≤ 3, a Golub–Kahan sweep `2(k − 1)` rotations. The sweep count is bounded
//! only by each kernel's own cap (`francis::MAX_ITER`, `bidiagonal_qr`'s
//! iteration cap); an exhausted cap is a typed error, never a value this
//! bound covers. At `n = 3` the observed sweeps are a handful per deflation
//! (Wilkinson shifts converge cubically once inside their basin, Golub &
//! Van Loan §8.3, §8.6), so `η = n²·ε = 9ε` is the first-order total for
//! `r ≈ n` transformations of order-`n` vectors at `c·u ≈ ε`; the constant
//! `c` is the one quantity not derived here, because Higham does not state it.
//! Restoring a result by `s` rounds it once onto `T`'s grid; in the subnormal
//! binades that costs at most half the subnormal spacing,
//! `0.5·2^(MIN−e)·ε` in units of `s`.
//!
//! - Singular values (Weyl): `|σ̂ − σ| ≤ δ + η‖Â‖_F + 0.5·2^(MIN−e)·ε`.
//! - Eigenvalues (Bauer–Fike): `|λ̂ − λ| ≤ κ·(δ + η‖Â‖_F) + ρ + 0.5·2^(MIN−e)·ε`,
//!   with `κ ≥ κ₂(V)`: for `SIMILAR = S·diag(1, 2, 4)·S⁻¹`,
//!   `κ = ‖S‖_F·‖S⁻¹‖_F = 3.675` in closed form; for `GENERAL`, `κ` from its
//!   `f64` left and right eigenvectors (`spectral_condition`), `ρ` the `f64`
//!   reference's own `κ·9ε₆₄·‖A‖_F`.
//! - Pivoted QR: `‖Â·P − Q·R̂‖_F ≤ η‖Â‖_F + n·0.5·2^(MIN−e)·ε`, and the rank
//!   is 3 whenever `δ < σ_min(A)/2`.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::format::{epsilon, Format};
use super::spectral_condition::bauer_fike_factor;
use eunomia::{Bf16, F16};
use leto::{Array2, Storage};
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

/// The spectrum a sweep checks: an eigenvalue-bearing base matrix, its
/// eigenvalues ascending, and the Bauer–Fike factor and reference error in
/// units of `s`.
struct Spectrum {
    matrix: [f64; 9],
    eigenvalues: [f64; 3],
    condition: f64,
    reference_error: f64,
}

fn similar_spectrum() -> Spectrum {
    Spectrum {
        matrix: SIMILAR,
        eigenvalues: SIMILAR_EIGENVALUES,
        condition: SIMILAR_CONDITION,
        reference_error: 0.0,
    }
}

/// `GENERAL`'s eigenvalues from the `f64` Schur form at unit scale, with the
/// Bauer–Fike factor from its eigenvectors — computed independently of
/// `schur` (`spectral_condition`).
fn general_spectrum() -> Spectrum {
    let reference = schur(
        &Array2::from_shape_vec([N, N], GENERAL.to_vec())
            .unwrap()
            .view(),
    )
    .unwrap()
    .eigenvalues();
    let mut eigenvalues = [0.0; 3];
    for (slot, z) in eigenvalues.iter_mut().zip(&reference) {
        assert_eq!(z.im, 0.0, "GENERAL has a real spectrum");
        *slot = z.re;
    }
    eigenvalues.sort_by(f64::total_cmp);
    let condition = bauer_fike_factor(&GENERAL, &eigenvalues);
    Spectrum {
        matrix: GENERAL,
        eigenvalues,
        condition,
        reference_error: condition * (N * N) as f64 * f64::EPSILON * frobenius(&GENERAL),
    }
}

fn check_eigenvalues<T: Format>(spectrum: &Spectrum) {
    for exponent in exponents::<T>() {
        let sample = sample::<T>(&spectrum.matrix, exponent);
        let bound = spectrum.condition * (sample.input_error + sample.backward)
            + spectrum.reference_error
            + sample.grid;
        let results = [
            eigenvalues(&sample.matrix.view()),
            schur(&sample.matrix.view()).map(|decomposition| decomposition.eigenvalues()),
        ];
        for result in results {
            let values = result.unwrap_or_else(|error| panic!("2^{exponent}: {error}"));
            for ((re, im), expected) in sorted_unit(&values, exponent)
                .into_iter()
                .zip(spectrum.eigenvalues)
            {
                assert!(
                    (re - expected).abs() <= bound && im.abs() <= bound,
                    "2^{exponent}: λ {re}+{im}i vs {expected}, bound {bound:e}"
                );
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
    // GENERAL covers the f32 band 2⁻³¹..2⁻²³ where the unscaled Francis step
    // used to stall; SIMILAR the known closed-form spectrum.
    for spectrum in [similar_spectrum(), general_spectrum()] {
        check_eigenvalues::<f64>(&spectrum);
        check_eigenvalues::<f32>(&spectrum);
        check_eigenvalues::<F16>(&spectrum);
        check_eigenvalues::<Bf16>(&spectrum);
    }
}

#[test]
fn pivoted_qr_is_correct_across_each_exponent_range() {
    check_pivoted_qr::<f64>();
    check_pivoted_qr::<f32>();
    check_pivoted_qr::<F16>();
    check_pivoted_qr::<Bf16>();
}

/// An input inside every gate is factored unscaled, so an entry at the
/// bottom of the subnormal range survives exactly: `diag(1, ½, 3·2^MINSUB)`
/// has an exact spectrum and singular values, and any power-of-two move
/// down would round `3·2^MINSUB` (an odd multiple of the subnormal spacing).
fn check_in_range_input_is_factored_unscaled<T: Format>() {
    let tiny = T::from_f64(3.0).mul(T::ONE.scale_binary(T::MIN_EXPONENT - T::PRECISION + 1));
    let mut values = vec![T::ZERO; N * N];
    values[0] = T::ONE;
    values[4] = T::from_f64(0.5);
    values[8] = tiny;
    let matrix = Array2::from_shape_vec([N, N], values).unwrap();
    let expected = [tiny.to_f64(), 0.5, 1.0];
    let mut sigmas: Vec<f64> = singular_values(&matrix.view())
        .unwrap()
        .iter()
        .map(|v| v.to_f64())
        .collect();
    sigmas.reverse();
    assert_eq!(sigmas, expected, "singular_values");
    let mut full: Vec<f64> = svd_decompose(&matrix.view())
        .unwrap()
        .singular_values
        .iter()
        .map(|v| v.to_f64())
        .collect();
    full.reverse();
    assert_eq!(full, expected, "svd_decompose");
    for values in [
        eigenvalues(&matrix.view()).unwrap(),
        schur(&matrix.view()).unwrap().eigenvalues(),
    ] {
        let spectrum: Vec<(f64, f64)> = sorted_unit(&values, 0);
        let expected_pairs: Vec<(f64, f64)> = expected.iter().map(|&v| (v, 0.0)).collect();
        assert_eq!(spectrum, expected_pairs, "Francis");
    }
    let r = col_piv_qr(&matrix.view()).unwrap().r();
    let mut diagonal: Vec<f64> = (0..N)
        .map(|i| r.storage().as_slice()[i * N + i].to_f64().abs())
        .collect();
    diagonal.sort_by(f64::total_cmp);
    assert_eq!(diagonal, expected, "col_piv_qr");
    let mut symmetric: Vec<f64> = symmetric_eigen_qr(&matrix.view())
        .unwrap()
        .eigenvalues
        .iter()
        .map(|v| v.to_f64())
        .collect();
    symmetric.sort_by(f64::total_cmp);
    assert_eq!(symmetric, expected, "symmetric_eigen_qr");
}

#[test]
fn in_range_inputs_are_factored_unscaled() {
    check_in_range_input_is_factored_unscaled::<f64>();
    check_in_range_input_is_factored_unscaled::<f32>();
    check_in_range_input_is_factored_unscaled::<F16>();
    check_in_range_input_is_factored_unscaled::<Bf16>();
}
