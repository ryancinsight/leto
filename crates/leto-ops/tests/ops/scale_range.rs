//! Exponent sweeps for the dense factorizations: `s·A` for every power of two
//! `s` from the smallest subnormal to three binades below the largest finite
//! value, in every supported format, must factor correctly — never a wrong
//! `Ok`, and never a typed error: the Francis and Golub–Kahan kernels form
//! their products scale-safely and the matrix-tier gate covers the rest
//! (`linalg::scaling`), so no exponent of any format is exempt.
//!
//! # Error bounds
//!
//! Work in units of `s`. Let `Â = image / s` be what `T` actually holds
//! (exact in `f64`), `δ = ‖Â − A‖_F` the input rounding (zero wherever `s·A`
//! is representable; nonzero only in the subnormal binades and, for `Bf16`,
//! where an entry needs more than 8 significant bits), and `η·‖Â‖_F` the
//! factorization's backward error, `η` derived in `backward_error.rs` for
//! each routine ([`svd`](super::backward_error::svd),
//! [`francis`](super::backward_error::francis),
//! [`col_piv_qr`](super::backward_error::col_piv_qr)) from Higham's `γ`
//! bounds over the enumerated transformations at the code's iteration caps.
//! These are worst cases at the caps — for `f64` about `10⁻¹⁰` relative,
//! vacuous (`η ≥ 1`) for `F16` and `Bf16` — and are asserted only where
//! informative ([`informative`](super::backward_error::informative)). Every
//! format is also checked a posteriori on the factors `svd_decompose` and
//! `schur` return (`a_posteriori.rs`): with `R` the residual measured in
//! `f64` and `δ_U, δ_V, δ_Q` the measured orthogonality defects, the Weyl
//! radius `‖R‖_F + (δ_U + δ_V + δ_Uδ_V)‖Σ̂‖₂` and the Bauer–Fike radius
//! `κ·(δ + ‖R‖_F + δ_Q(2 + δ_Q)‖T̂‖_F)` replace `η‖Â‖_F` below.
//! Restoring a result by `s` rounds it once onto `T`'s grid; in the subnormal
//! binades that costs at most half the subnormal spacing,
//! `0.5·2^(MIN−e)·ε` in units of `s`.
//!
//! - Singular values (Weyl): `|σ̂ − σ| ≤ δ + η‖Â‖_F + ρ + 0.5·2^(MIN−e)·ε`,
//!   `ρ` the error of the `f64` reference `√λ(AᵀA)`: `η_QL(ε₆₄)‖AᵀA‖_F/σ_min`.
//! - Eigenvalues (Bauer–Fike): `|λ̂ − λ| ≤ κ·(δ + η‖Â‖_F) + ρ + 0.5·2^(MIN−e)·ε`,
//!   with `κ ≥ κ₂(V)`: for `SIMILAR = S·diag(1, 2, 4)·S⁻¹`,
//!   `κ = ‖S‖_F·‖S⁻¹‖_F = 3.675` in closed form; for `GENERAL`, `κ` from its
//!   `f64` left and right eigenvectors (`spectral_condition`), `ρ` the `f64`
//!   reference's own `κ·η(ε₆₄)·‖A‖_F`.
//! - Pivoted QR: `‖Â·P − Q·R̂‖_F ≤ η‖Â‖_F + n·0.5·2^(MIN−e)·ε`, plus the `f64`
//!   evaluation's `γ_{n+1}·‖|Q||R|‖_F`, and the rank is 3 whenever
//!   `δ < σ_min(A)/2`.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::a_posteriori::{self, BlockEigenvalue};
use super::backward_error::{self, informative};
use super::format::{epsilon, Format};
use super::spectral_condition::bauer_fike_factor;
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
    norm: f64,
    epsilon: f64,
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
        norm: frobenius(&unit),
        epsilon,
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
/// the symmetric solver, in `f64`, descending, with their error bound
/// `|√λ̂ − √λ| ≤ |λ̂ − λ|/√λ ≤ η_QL(ε₆₄)·‖AᵀA‖_F/σ_min`.
fn general_singular_values() -> (Vec<f64>, f64) {
    let mut gram = vec![0.0; N * N];
    for i in 0..N {
        for j in 0..N {
            gram[i * N + j] = (0..N)
                .map(|k| GENERAL[k * N + i] * GENERAL[k * N + j])
                .sum();
        }
    }
    let bound = backward_error::ql(N, f64::EPSILON) * frobenius(&gram);
    let eigen = symmetric_eigen_qr(&Array2::from_shape_vec([N, N], gram).unwrap().view()).unwrap();
    let sigmas: Vec<f64> = eigen.eigenvalues.iter().rev().map(|l| l.sqrt()).collect();
    let error = bound / (sigmas[N - 1] * sigmas[N - 1] - bound).max(0.0).sqrt();
    (sigmas, error)
}

fn check_singular_values<T: Format>() {
    let (reference, reference_error) = general_singular_values();
    for exponent in exponents::<T>() {
        let sample = sample::<T>(&GENERAL, exponent);
        let values = singular_values(&sample.matrix.view())
            .unwrap_or_else(|error| panic!("2^{exponent}: {error}"));
        let full = svd_decompose(&sample.matrix.view())
            .unwrap_or_else(|error| panic!("2^{exponent}: {error}"));
        // A posteriori, on the returned factors (Σ̂ in units of s).
        let sigmas: Vec<f64> = full
            .singular_values
            .iter()
            .map(|v| scale(v.to_f64(), -exponent))
            .collect();
        let as_f64 = |m: &Array2<T>| -> Vec<f64> {
            m.storage().as_slice().iter().map(|v| v.to_f64()).collect()
        };
        let (u, v) = (
            as_f64(&full.left_singular_vectors),
            as_f64(&full.right_singular_vectors),
        );
        let mut diagonal = vec![0.0; N * N];
        for (i, sigma) in sigmas.iter().enumerate() {
            diagonal[i * N + i] = *sigma;
        }
        let residual = a_posteriori::residual(&sample.unit, &u, &diagonal, &v, N, N, N);
        let radius = a_posteriori::svd_certificate(
            residual,
            a_posteriori::gram_defect(&u, N, N),
            a_posteriori::gram_defect(&v, N, N),
            &sigmas,
        );
        let certified = sample.input_error + radius + reference_error;
        for (sigma, expected) in sigmas.iter().zip(&reference) {
            assert!(
                (sigma - expected).abs() <= certified,
                "2^{exponent}: σ {sigma} vs {expected}, certified {certified:e}"
            );
        }
        // A priori, both entry points, where informative.
        let Some(eta) = informative(backward_error::svd(N, N, sample.epsilon)) else {
            continue;
        };
        let bound = sample.input_error + eta * sample.norm + reference_error + sample.grid;
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
    let condition = bauer_fike_factor(&GENERAL, &eigenvalues.map(|re| (re, 0.0)));
    Spectrum {
        matrix: GENERAL,
        eigenvalues,
        condition,
        reference_error: condition * backward_error::francis(N, f64::EPSILON) * frobenius(&GENERAL),
    }
}

fn check_eigenvalues<T: Format>(spectrum: &Spectrum) {
    for exponent in exponents::<T>() {
        let sample = sample::<T>(&spectrum.matrix, exponent);
        let decomposition =
            schur(&sample.matrix.view()).unwrap_or_else(|error| panic!("2^{exponent}: {error}"));
        let spectrum_only = eigenvalues(&sample.matrix.view())
            .unwrap_or_else(|error| panic!("2^{exponent}: {error}"));
        // A posteriori: every eigenvalue of T̂ is one of Â − E = A + (Â − A) − E,
        // within κ·(δ + ‖E‖₂) of the spectrum of A (Bauer–Fike on A).
        let q: Vec<f64> = decomposition
            .q()
            .storage()
            .as_slice()
            .iter()
            .map(|v| v.to_f64())
            .collect();
        let t: Vec<f64> = decomposition
            .t()
            .storage()
            .as_slice()
            .iter()
            .map(|v| scale(v.to_f64(), -exponent))
            .collect();
        let defect = a_posteriori::gram_defect(&q, N, N);
        let residual = a_posteriori::residual(&sample.unit, &q, &t, &q, N, N, N);
        let radius = a_posteriori::schur_certificate(residual, defect, &t);
        for BlockEigenvalue { re, im, error } in a_posteriori::quasi_triangular_eigenvalues(&t, N) {
            let nearest = spectrum
                .eigenvalues
                .iter()
                .map(|e| (re - e).hypot(im))
                .fold(f64::INFINITY, f64::min);
            let certified = spectrum.condition * (sample.input_error + radius)
                + spectrum.reference_error
                + error;
            assert!(
                nearest <= certified,
                "2^{exponent}: λ(T̂) {re}+{im}i is {nearest:e} from the spectrum, certified {certified:e}"
            );
        }
        // A priori, both entry points, where informative.
        let Some(eta) = informative(backward_error::francis(N, sample.epsilon)) else {
            continue;
        };
        let bound = spectrum.condition * (sample.input_error + eta * sample.norm)
            + spectrum.reference_error
            + sample.grid;
        for values in [spectrum_only, decomposition.eigenvalues()] {
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
    let sigma_min = *general_singular_values().0.last().unwrap();
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
        let (mut residual, mut rounding) = (0.0, 0.0);
        for i in 0..N {
            for (k, &column) in permutation.iter().enumerate() {
                // R in units of s before multiplying, so no f64 product is subnormal.
                let terms =
                    (0..N).map(|j| q[i * N + j].to_f64() * scale(r[j * N + k].to_f64(), -exponent));
                let qr: f64 = terms.clone().sum();
                let magnitude: f64 = terms.map(f64::abs).sum();
                residual += (sample.unit[i * N + column] - qr).powi(2);
                rounding +=
                    (backward_error::gamma(N as f64 + 1.0, f64::EPSILON) * magnitude).powi(2);
            }
        }
        // A priori only (pivoted QR returns no certificate of its own), where
        // informative: every format but Bf16.
        let Some(eta) = informative(backward_error::col_piv_qr(N, N, sample.epsilon)) else {
            continue;
        };
        let bound = eta * sample.norm + N as f64 * sample.grid + rounding.sqrt();
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

/// The absolute deflation floor stays below `ε·‖A‖_max`
/// (`thresholds::homogeneous_safe_range`). `dbdsqr`'s `6k²·safmin` does not
/// in `F16`: at `k = 24` it is `6·24²·2⁻¹⁴ ≈ 0.21`, and would split every
/// superdiagonal below that of a unit-scale matrix. Twelve copies of the block
/// `[[1, a], [0, 1]]`, `a = 1/16` (already bidiagonal, so the reduction
/// applies identity reflectors; the gate moves `‖A‖_max = 1` up to its
/// raised lower end `2^5·smlnum = 2`, where `2a = ⅛` is still below `0.21`),
/// have singular values `√(1 + a²/4) ± a/2 ≈ 1.0317, 0.9692`; a split returns
/// `1` for both. The check separates the two structurally: every `σ̂` more
/// than `a/4` from `1` (the exact values are `≥ 0.030` away, the split `0`).
#[test]
fn deflation_floor_keeps_unit_scale_superdiagonals_in_f16() {
    let (k, a) = (24, 0.0625_f64);
    let mut values = vec![F16::from_f64(0.0); k * k];
    for block in 0..k / 2 {
        let i = 2 * block;
        values[i * k + i] = F16::from_f64(1.0);
        values[i * k + i + 1] = F16::from_f64(a);
        values[(i + 1) * k + i + 1] = F16::from_f64(1.0);
    }
    let matrix = Array2::from_shape_vec([k, k], values).unwrap();
    for sigmas in [
        singular_values(&matrix.view()).unwrap(),
        svd_decompose(&matrix.view()).unwrap().singular_values,
    ] {
        for sigma in sigmas {
            let sigma = f64::from(sigma.to_f32());
            assert!(
                (sigma - 1.0).abs() > a / 4.0,
                "σ {sigma}: the superdiagonal {a} was deflated"
            );
        }
    }
}

/// Where no scaling keeps the deflation floor `2^g·safmin` (`g = ⌈log₂ n⌉`)
/// at or below `ε·‖A‖_max`, the gate reports [`LetoError::Overflow`] instead
/// of factoring with a floor outside the backward error. Both gates are
/// degree 2 in `F16`: upper end `(Ω·2⁻ᶠ)^½ < 2^(8 − f/2)`, raised lower end
/// `2^g·smlnum = 2^(g − 4)`, so a gate empties once `g + f/2 ≥ 12`. With
/// `r = ⌈log₂(‖A‖_F/‖A‖_max)⌉ = log₂ n` for the all-ones matrix: the SVD
/// (`f = 2 + ⌈⌈log₂ n⌉/2⌉ + r`) at `n = 256` has `8 + 7 = 15`; Francis
/// (`f = 2r`) at `n = 128` has `7 + 7 = 14`; at `n = 32` they have
/// `5 + 5 = 10` and `5 + 5 = 10`, and factor.
#[test]
fn f16_orders_past_the_deflation_floor_are_typed_overflow() {
    let ones = |n: usize| Array2::from_shape_vec([n, n], vec![F16::from_f64(1.0); n * n]).unwrap();
    let floor = |result: Result<(), LetoError>| match result {
        Err(LetoError::Overflow { reason }) => {
            assert!(reason.contains("deflation floor"), "{reason}")
        }
        other => panic!("expected the deflation-floor overflow, got {other:?}"),
    };
    floor(singular_values(&ones(256).view()).map(drop));
    floor(svd_decompose(&ones(256).view()).map(drop));
    floor(schur(&ones(128).view()).map(drop));
    floor(eigenvalues(&ones(128).view()).map(drop));
    schur(&ones(32).view()).unwrap();
    svd_decompose(&ones(32).view()).unwrap();
}
