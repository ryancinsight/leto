#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

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

mod eigen;
mod pivoted_qr;
mod svd;
mod unscaled;

use super::backward_error;
use super::format::{epsilon, scale, Format};
use leto::Array2;
use leto_ops::symmetric_eigen_qr;

const N: usize = 3;

/// A general matrix with distinct singular values; every entry is exact in
/// every supported format.
const GENERAL: [f64; 9] = [4.0, 1.0, 0.5, 1.0, 3.0, 1.0, 0.25, 1.0, 2.0];

fn frobenius(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
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
