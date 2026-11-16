//! Seeded random scan at every binary exponent from each format's smallest
//! normal to three binades below its largest, over six families of order
//! 2 to 8:
//!
//! - **graded** — general, row `i` graded by `2⁻²ⁱ`, so the small singular
//!   values and the converging off-diagonals reach the subnormals at the
//!   bottom of the range;
//! - **clustered** — symmetric `Q·diag(1, 1, 1.001, 1.001, …, 0.5)·Qᵀ`, pairs
//!   of equal eigenvalues `10⁻³` apart;
//! - **rank-deficient** — general `X·Y` of rank `⌊n/2⌋`, and its symmetric
//!   counterpart `Q·diag(d, 0)·Qᵀ`;
//! - **skew-symmetric** — tridiagonal with off-diagonal magnitudes
//!   log-uniform over `[10⁻⁴, 1]` and random signs (the family on which
//!   `schur` failed to converge while `eigenvalues` did), and dense.
//!
//! Every routine must converge. Two kinds of check follow, on the image `Â`
//! of what `T` holds (so the input rounding is zero):
//!
//! - **A posteriori** (`a_posteriori.rs`), every format: the residuals of
//!   the returned factors, `‖Â − Q̂T̂Q̂ᵀ‖_F`, `‖Q̂ᵀQ̂ − I‖`,
//!   `‖Â − ÛΣ̂V̂ᵀ‖_F`, `‖ÛᵀÛ − I‖`, `‖V̂ᵀV̂ − I‖`, measured in `f64`; each
//!   eigenvalue of `T̂` within `κ·‖E‖₂` of the spectrum of `Â` (Bauer–Fike,
//!   `κ = 1` for the normal families, else from the reference's left and
//!   right eigenvectors, `spectral_condition`), each `σ̂` within the Weyl
//!   radius of `σ(Â)`, and every eigenvalue `schur` returns equal to the
//!   eigenvalue of its own `T̂` block up to the read-off's rounding. These
//!   hold for whatever factors come back: they certify the values against
//!   the factors, not the factors.
//! - **A priori** (`backward_error.rs`), only where the derived bound —
//!   deterministic at the iteration caps, else probabilistic at the
//!   instrumented counts — is informative (`η < 1`): the measured residuals
//!   within the derived `η_R·‖Â‖_F` and `η_O`, which certifies the factors; the
//!   factor-free entry points `eigenvalues` and `singular_values` within
//!   `(η(ε(T)) + η(ε₆₄))·‖Â‖_F` (Weyl, Bauer–Fike) of the `f64` reference;
//!   and `singular_values` within `2η(ε(T))·‖Â‖_F` of `svd_decompose`.
//!
//! `coverage.rs` tallies, per format and factor site, the cases asserted and
//! the cases not, each of which derives `η ≥ 1` from its rounding counts.
//!
//! The `f64` reference carries its own a-priori error `η(ε₆₄)` (always
//! informative). Restoring a result onto `T`'s grid adds half the subnormal
//! spacing per value.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::backward_error::{self, informative};
use super::format::{epsilon, scale, Format};
use super::spectral_condition::condition_by_inverse_iteration;
use case::{frobenius, nearest, Case};
use coverage::{Coverage, Row, Site};
use eunomia::{Bf16, F16};
use families::{place, unit, Family, FAMILIES};
use leto::Array2;
use leto_ops::{
    eigenvalues, schur, singular_values, svd_decompose, take_dbdsqr_counts, take_francis_steps,
    Xorshift64,
};

mod case;
mod coverage;
mod families;

fn check_case<T: Format>(
    coverage: &mut Coverage,
    family: Family,
    n: usize,
    unit_values: &[f64],
    exponent: i32,
) {
    let (matrix, image) = place::<T>(unit_values, n, exponent);
    let label = format!(
        "{} {family:?} 2^{exponent} n = {n}",
        std::any::type_name::<T>()
    );
    let eps = epsilon::<T>();
    let norm = frobenius(&image);
    let case = Case {
        label: &label,
        matrix: &matrix,
        image: &image,
        n,
        exponent,
        // Half the subnormal spacing, in units of 2^e.
        grid: 0.5 * scale(1.0, T::MIN_EXPONENT - T::PRECISION + 1 - exponent),
    };
    let reference = Array2::from_shape_vec([n, n], image.clone()).unwrap();

    // Singular values.
    let reference_sigmas = singular_values(&reference.view()).unwrap();
    let reference_error = backward_error::svd(n, n, f64::EPSILON) * norm;
    let full = case.svd_factors(coverage, &reference_sigmas, reference_error);
    let values = singular_values(&matrix.view())
        .unwrap_or_else(|error| panic!("{label}: singular_values: {error}"));
    let (sweeps, chases) = take_dbdsqr_counts();
    // Deterministic at caps, else probabilistic at the instrumented counts.
    let eta = informative(backward_error::svd(n, n, eps))
        .map(|eta| (eta, "a-priori"))
        .or_else(|| {
            informative(backward_error::svd_tilde(n, n, eps, sweeps, chases))
                .map(|eta| (eta, "probabilistic"))
        });
    if let Some((eta, tag)) = eta {
        let agreement = 2.0 * eta * norm + 2.0 * case.grid;
        let weyl = eta * norm + reference_error + case.grid;
        for ((a, b), r) in values.iter().zip(&full).zip(&reference_sigmas) {
            let a = scale(a.to_f64(), -exponent);
            assert!(
                (a - b).abs() <= agreement,
                "{label}: [{tag} {sweeps}+{chases}] singular_values {a:e} vs svd_decompose {b:e}, bound {agreement:e}"
            );
            assert!(
                (a - r).abs() <= weyl,
                "{label}: [{tag} {sweeps}+{chases}] σ {a:e} vs {r:e}, bound {weyl:e}"
            );
        }
    }

    // Eigenvalues. A repeated zero eigenvalue leaves the general
    // rank-deficient family without a Bauer–Fike factor; the symmetric one
    // covers it.
    let reference_spectrum: Vec<(f64, f64)> = eigenvalues(&reference.view())
        .unwrap()
        .iter()
        .map(|z| (z.re, z.im))
        .collect();
    let condition = match family {
        Family::RankDeficient => None,
        _ if family.normal() => Some(1.0),
        _ => Some(condition_by_inverse_iteration(
            &image,
            n,
            &reference_spectrum,
        )),
    };
    let reference_error =
        condition.unwrap_or(0.0) * backward_error::francis(n, f64::EPSILON) * norm;
    case.schur_factors(coverage, &reference_spectrum, condition, reference_error);
    let spectrum =
        eigenvalues(&matrix.view()).unwrap_or_else(|error| panic!("{label}: eigenvalues: {error}"));
    let steps = take_francis_steps();
    let eta = informative(backward_error::francis(n, eps))
        .map(|eta| (eta, "a-priori"))
        .or_else(|| {
            informative(backward_error::francis_tilde(n, eps, steps))
                .map(|eta| (eta, "probabilistic"))
        });
    let (Some(condition), Some((eta, tag))) = (condition, eta) else {
        return;
    };
    let bound = condition * eta * norm + reference_error + case.grid;
    for z in &spectrum {
        let (re, im) = (
            scale(z.re.to_f64(), -exponent),
            scale(z.im.to_f64(), -exponent),
        );
        let distance = nearest(&reference_spectrum, re, im);
        assert!(
            distance <= bound,
            "{label}: [{tag} {steps}] eigenvalues() {re:e}+{im:e}i is {distance:e} from the spectrum, bound {bound:e} (κ {condition:e})"
        );
    }
}

fn check_scan<T: Format>(coverage: &mut Coverage, seed: u64) {
    let mut rng = Xorshift64::new(seed);
    for exponent in T::MIN_EXPONENT..=(T::MAX_EXPONENT - 3) {
        for (index, family) in FAMILIES.into_iter().enumerate() {
            let n = 2 + (exponent.unsigned_abs() as usize * 3 + index) % 7;
            let unit_values = family.matrix(n, &mut rng);
            check_case::<T>(coverage, family, n, &unit_values, exponent);
        }
    }
}

#[test]
fn random_families_converge_accurately_at_every_normal_magnitude() {
    let mut coverage = Coverage::default();
    check_scan::<f64>(&mut coverage, 0x5EED_0064);
    check_scan::<f32>(&mut coverage, 0x5EED_0032);
    check_scan::<F16>(&mut coverage, 0x5EED_0016);
    check_scan::<Bf16>(&mut coverage, 0x5EED_BF16);
    coverage.assert_rows(&[
        // Cases per site: 2043 exponents x 6 families = 12258 in f64, 251 x 6 = 1506 in f32.
        (f64::PRECISION, Site::SvdResidual, Row::new(12258, 0)),
        (f64::PRECISION, Site::SvdOrthogonality, Row::new(12258, 0)),
        (f64::PRECISION, Site::SchurResidual, Row::new(12258, 0)),
        (f64::PRECISION, Site::SchurOrthogonality, Row::new(12258, 0)),
        (f32::PRECISION, Site::SvdResidual, Row::new(1506, 0)),
        (f32::PRECISION, Site::SvdOrthogonality, Row::new(1506, 0)),
        (f32::PRECISION, Site::SchurResidual, Row::new(1506, 0)),
        (f32::PRECISION, Site::SchurOrthogonality, Row::new(1506, 0)),
        // 27 exponents x 6 families = 162 cases per site.
        (F16::PRECISION, Site::SvdResidual, Row::new(162, 0)),
        (F16::PRECISION, Site::SvdOrthogonality, Row::new(162, 0)),
        (F16::PRECISION, Site::SchurResidual, Row::new(148, 14)),
        (F16::PRECISION, Site::SchurOrthogonality, Row::new(162, 0)),
        // 251 exponents x 6 families = 1506 cases per site.
        (Bf16::PRECISION, Site::SvdResidual, Row::new(0, 1506)),
        (Bf16::PRECISION, Site::SvdOrthogonality, Row::new(0, 1506)),
        (Bf16::PRECISION, Site::SchurResidual, Row::new(0, 1506)),
        (
            Bf16::PRECISION,
            Site::SchurOrthogonality,
            Row::new(272, 1234),
        ),
    ]);
}

/// Orders past where `dbdsqr`'s `maxitr·n²·unfl` floor passes the gate's
/// upper end in `F16` (`k ≳ 24`; the fifth review found `svd_decompose`
/// returning a false `Overflow` at order 48): three matrices per order,
/// entries uniform in `[−1, 1]`, must factor and agree with the `f64`
/// factorization of their image.
fn check_large_orders<T: Format>(coverage: &mut Coverage, orders: &[usize], seed: u64) {
    let mut rng = Xorshift64::new(seed);
    for &n in orders {
        for _ in 0..3 {
            let unit_values: Vec<f64> = (0..n * n).map(|_| unit(&mut rng)).collect();
            check_case::<T>(coverage, Family::RankDeficient, n, &unit_values, 0);
        }
    }
}

#[test]
fn large_orders_factor_in_the_narrow_formats() {
    let mut coverage = Coverage::default();
    check_large_orders::<F16>(&mut coverage, &[16, 24, 32, 48], 0xA11CE);
    check_large_orders::<Bf16>(&mut coverage, &[32, 64], 0xA11CE);
    // 3 matrices per order: 12 cases in F16 (orders 16 to 48), 6 in Bf16 (32, 64).
    coverage.assert_rows(&[
        (F16::PRECISION, Site::SvdResidual, Row::new(0, 12)),
        (F16::PRECISION, Site::SvdOrthogonality, Row::new(0, 12)),
        (F16::PRECISION, Site::SchurResidual, Row::new(0, 12)),
        (F16::PRECISION, Site::SchurOrthogonality, Row::new(0, 12)),
        (Bf16::PRECISION, Site::SvdResidual, Row::new(0, 6)),
        (Bf16::PRECISION, Site::SvdOrthogonality, Row::new(0, 6)),
        (Bf16::PRECISION, Site::SchurResidual, Row::new(0, 6)),
        (Bf16::PRECISION, Site::SchurOrthogonality, Row::new(0, 6)),
    ]);
}

/// `diag(1, B)` with `B` a random upper-Hessenberg block of subnormal
/// entries (integers up to 64 times the smallest subnormal, order 3 to 6): no
/// gate moves a matrix whose largest entry is `1`, so the block's converging
/// subdiagonals stay in the subnormals, where the relative deflation test
/// holds only for an exact zero and the sweep cycles. The absolute deflation
/// floor (`safmin` in both iterations) deflates them. Every singular value
/// and eigenvalue of `B` is at most `‖B‖_F`, and at most `k` floor
/// deflations perturb by at most `√k·safmin` jointly, so the results other
/// than the exact `1` stay within `‖B‖_F + √k·safmin`.
fn check_subnormal_block_deflates<T: Format>(seed: u64) {
    let unit_value = T::ONE.scale_binary(T::MIN_EXPONENT - T::PRECISION + 1);
    let mut rng = Xorshift64::new(seed);
    let label = std::any::type_name::<T>();
    for case in 0..40 {
        let k = 4 + case % 4;
        let mut values = vec![T::ZERO; k * k];
        values[0] = T::ONE;
        for i in 1..k {
            for j in (i - 1).max(1)..k {
                let integer = (64.0 * unit(&mut rng)).round();
                values[i * k + j] = T::from_f64(integer).mul(unit_value);
            }
        }
        let block_norm = values[1..]
            .iter()
            .map(|v| v.to_f64().powi(2))
            .sum::<f64>()
            .sqrt();
        let tolerance =
            block_norm + (k as f64).sqrt() * T::ONE.scale_binary(T::MIN_EXPONENT).to_f64();
        let matrix = Array2::from_shape_vec([k, k], values).unwrap();
        let case = format!("{label} case {case} (n = {k})");
        for sigmas in [
            singular_values(&matrix.view()).unwrap_or_else(|e| panic!("{case}: {e}")),
            svd_decompose(&matrix.view())
                .unwrap_or_else(|e| panic!("{case}: {e}"))
                .singular_values,
        ] {
            assert_eq!(sigmas[0].to_f64(), 1.0, "{case}: σ₁");
            for sigma in &sigmas[1..] {
                assert!(
                    sigma.to_f64() <= tolerance,
                    "{case}: σ {:e}",
                    sigma.to_f64()
                );
            }
        }
        for result in [
            eigenvalues(&matrix.view()),
            schur(&matrix.view()).map(|decomposition| decomposition.eigenvalues()),
        ] {
            let spectrum = result.unwrap_or_else(|e| panic!("{case}: {e}"));
            let parts: Vec<(f64, f64)> = spectrum
                .iter()
                .map(|z| (z.re.to_f64(), z.im.to_f64()))
                .collect();
            let is_unit = |&(re, im): &(f64, f64)| re == 1.0 && im == 0.0;
            let unit_eigenvalues = parts.iter().filter(|z| is_unit(z)).count();
            assert_eq!(unit_eigenvalues, 1, "{case}: λ = 1");
            for (re, im) in parts.iter().copied().filter(|z| !is_unit(z)) {
                let modulus = re.hypot(im);
                assert!(modulus <= tolerance, "{case}: |λ| {modulus:e}");
            }
        }
    }
}

#[test]
fn subnormal_blocks_deflate_at_the_absolute_floor() {
    check_subnormal_block_deflates::<f64>(0xB10C_0064);
    check_subnormal_block_deflates::<f32>(0xB10C_0032);
    check_subnormal_block_deflates::<F16>(0xB10C_0016);
    check_subnormal_block_deflates::<Bf16>(0xB10C_BF16);
}
