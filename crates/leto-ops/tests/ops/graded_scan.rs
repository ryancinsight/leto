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
//! - **A priori** (`backward_error.rs`), only where the derived bound is
//!   informative (`η < 1`: `f64` everywhere; `f32` for the SVD, the Schur
//!   residual to order 4 and the Francis eigenvalues to order 7; never
//!   `F16` or `Bf16`): the measured residuals within the
//!   derived `η_R·‖Â‖_F` and `η_O`, which certifies the factors; the
//!   factor-free entry points `eigenvalues` and `singular_values` within
//!   `(η(ε(T)) + η(ε₆₄))·‖Â‖_F` (Weyl, Bauer–Fike) of the `f64` reference;
//!   and `singular_values` within `2η(ε(T))·‖Â‖_F` of `svd_decompose`.
//!
//! The `f64` reference carries its own a-priori error `η(ε₆₄)` (always
//! informative). Restoring a result onto `T`'s grid adds half the subnormal
//! spacing per value.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::a_posteriori::{self, BlockEigenvalue};
use super::backward_error::{self, informative};
use super::format::{epsilon, Format};
use super::spectral_condition::condition_by_inverse_iteration;
use eunomia::{Bf16, F16};
use leto::{Array2, Storage};
use leto_ops::{eigenvalues, schur, singular_values, svd_decompose, Xorshift64};

/// `x·2^k` in `f64`, exact for every `k` a supported format produces.
fn scale(x: f64, k: i32) -> f64 {
    let half = k / 2;
    x * 2.0_f64.powi(half) * 2.0_f64.powi(k - half)
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

fn unit(rng: &mut Xorshift64) -> f64 {
    2.0 * rng.next_unit_f64() - 1.0
}

/// A random orthogonal `n × n` (Gram–Schmidt, twice, on uniform columns).
fn orthogonal(n: usize, rng: &mut Xorshift64) -> Vec<f64> {
    let mut q = vec![0.0; n * n];
    for j in 0..n {
        let mut v: Vec<f64> = (0..n).map(|_| unit(rng)).collect();
        for _ in 0..2 {
            for k in 0..j {
                let d: f64 = (0..n).map(|i| v[i] * q[i * n + k]).sum();
                for i in 0..n {
                    v[i] -= d * q[i * n + k];
                }
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for i in 0..n {
            q[i * n + j] = v[i] / norm;
        }
    }
    q
}

/// `Q·diag(d)·Qᵀ`, exactly symmetric.
fn symmetric_from(q: &[f64], d: &[f64], n: usize) -> Vec<f64> {
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v: f64 = (0..n).map(|k| q[i * n + k] * d[k] * q[j * n + k]).sum();
            s[i * n + j] = v;
            s[j * n + i] = v;
        }
    }
    s
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Family {
    Graded,
    Clustered,
    RankDeficient,
    SymmetricRankDeficient,
    SkewTridiagonal,
    SkewDense,
}

const FAMILIES: [Family; 6] = [
    Family::Graded,
    Family::Clustered,
    Family::RankDeficient,
    Family::SymmetricRankDeficient,
    Family::SkewTridiagonal,
    Family::SkewDense,
];

impl Family {
    /// Normal (`AAᵀ = AᵀA`): an orthogonal eigenbasis, `κ = 1`.
    fn normal(self) -> bool {
        !matches!(self, Self::Graded | Self::RankDeficient)
    }

    /// A seeded matrix of order `n` at unit scale.
    fn matrix(self, n: usize, rng: &mut Xorshift64) -> Vec<f64> {
        match self {
            Self::Graded => (0..n * n)
                .map(|k| unit(rng) * scale(1.0, -2 * i32::try_from(k / n).unwrap()))
                .collect(),
            Self::Clustered => {
                let q = orthogonal(n, rng);
                let d: Vec<f64> = (0..n)
                    .map(|i| {
                        if i + 1 == n {
                            0.5
                        } else {
                            1.0 + (i / 2) as f64 * 1e-3
                        }
                    })
                    .collect();
                symmetric_from(&q, &d, n)
            }
            Self::RankDeficient => {
                let r = (n / 2).max(1);
                let x: Vec<f64> = (0..n * r).map(|_| unit(rng)).collect();
                let y: Vec<f64> = (0..r * n).map(|_| unit(rng)).collect();
                (0..n * n)
                    .map(|k| (0..r).map(|t| x[(k / n) * r + t] * y[t * n + k % n]).sum())
                    .collect()
            }
            Self::SymmetricRankDeficient => {
                let q = orthogonal(n, rng);
                let r = (n / 2).max(1);
                let d: Vec<f64> = (0..n)
                    .map(|i| if i < r { 0.5 + unit(rng).abs() } else { 0.0 })
                    .collect();
                symmetric_from(&q, &d, n)
            }
            Self::SkewTridiagonal => {
                let mut m = vec![0.0; n * n];
                for i in 0..n - 1 {
                    let sign = if unit(rng) < 0.0 { -1.0 } else { 1.0 };
                    let magnitude = 10.0_f64.powf(-4.0 * rng.next_unit_f64());
                    m[i * n + i + 1] = sign * magnitude;
                    m[(i + 1) * n + i] = -sign * magnitude;
                }
                m
            }
            Self::SkewDense => {
                let mut m = vec![0.0; n * n];
                for i in 0..n {
                    for j in i + 1..n {
                        let v = unit(rng);
                        m[i * n + j] = v;
                        m[j * n + i] = -v;
                    }
                }
                m
            }
        }
    }
}

/// The `T` matrix `2^e·unit` and the exact image of what `T` holds, in units
/// of `2^e`.
fn place<T: Format>(unit_values: &[f64], n: usize, exponent: i32) -> (Array2<T>, Vec<f64>) {
    let s = T::ONE.scale_binary(exponent);
    let values: Vec<T> = unit_values.iter().map(|&v| T::from_f64(v).mul(s)).collect();
    let image = values
        .iter()
        .map(|v| scale(v.to_f64(), -exponent))
        .collect();
    (Array2::from_shape_vec([n, n], values).unwrap(), image)
}

/// The image of a `T` array, in units of `2^e`.
fn image_of<T: Format>(values: &Array2<T>, exponent: i32) -> Vec<f64> {
    values
        .storage()
        .as_slice()
        .iter()
        .map(|v| scale(v.to_f64(), -exponent))
        .collect()
}

/// Distance from `(re, im)` to the nearest point of `spectrum`.
fn nearest(spectrum: &[(f64, f64)], re: f64, im: f64) -> f64 {
    spectrum
        .iter()
        .map(|(r, i)| (re - r).hypot(im - i))
        .fold(f64::INFINITY, f64::min)
}

/// One scan case: the `T` matrix, the image `Â` of what it holds in units of
/// `2^e`, and half the subnormal spacing in the same units.
struct Case<'a, T> {
    label: &'a str,
    matrix: &'a Array2<T>,
    image: &'a [f64],
    n: usize,
    exponent: i32,
    grid: f64,
}

impl<T: Format> Case<'_, T> {
    fn norm(&self) -> f64 {
        frobenius(self.image)
    }

    /// `svd_decompose`: the a-posteriori Weyl certificate against the `f64`
    /// reference `σ(Â)` (within `reference_error`), and the a-priori residual
    /// and orthogonality where informative. Returns `σ̂` in units of `2^e`.
    fn svd_factors(&self, reference: &[f64], reference_error: f64) -> Vec<f64> {
        let (label, n, eps) = (self.label, self.n, epsilon::<T>());
        let full = svd_decompose(&self.matrix.view())
            .unwrap_or_else(|error| panic!("{label}: svd_decompose: {error}"));
        let sigmas: Vec<f64> = full
            .singular_values
            .iter()
            .map(|s| scale(s.to_f64(), -self.exponent))
            .collect();
        // Û and V̂ are scale-free; Σ̂ is in units of 2^e.
        let u = image_of(&full.left_singular_vectors, 0);
        let v = image_of(&full.right_singular_vectors, 0);
        let mut diagonal = vec![0.0; n * n];
        for (i, s) in sigmas.iter().enumerate() {
            diagonal[i * n + i] = *s;
        }
        // Restoring Σ̂ rounds each value once onto T's grid.
        let residual = a_posteriori::residual(self.image, &u, &diagonal, &v, n, n, n)
            + (n as f64).sqrt() * self.grid;
        let (left, right) = (
            a_posteriori::gram_defect(&u, n, n),
            a_posteriori::gram_defect(&v, n, n),
        );
        let (eta_r, eta_o) = backward_error::svd_factors(n, n, eps);
        if let Some(eta) = informative(eta_r) {
            let bound = eta * self.norm();
            assert!(
                residual <= bound,
                "{label}: ‖Â − ÛΣ̂V̂ᵀ‖_F {residual:e} > {bound:e}"
            );
        }
        if let Some(eta) = informative(eta_o) {
            assert!(
                left <= eta && right <= eta,
                "{label}: ‖ÛᵀÛ − I‖ {left:e}, ‖V̂ᵀV̂ − I‖ {right:e} > {eta:e}"
            );
        }
        let bound = a_posteriori::svd_certificate(residual, left, right, &sigmas) + reference_error;
        for (s, r) in sigmas.iter().zip(reference) {
            assert!(
                (s - r).abs() <= bound,
                "{label}: σ {s:e} vs {r:e}, certified {bound:e} (residual {residual:e}, δ_U {left:e}, δ_V {right:e})"
            );
        }
        sigmas
    }

    /// `schur`: the read-off of each returned eigenvalue from its own `T̂`
    /// block, the a-posteriori Bauer–Fike certificate against the `f64`
    /// reference spectrum (`condition` is `None` when `Â` has no eigenvector
    /// basis to bound), and the a-priori residual and orthogonality where
    /// informative.
    fn schur_factors(
        &self,
        reference: &[(f64, f64)],
        condition: Option<f64>,
        reference_error: f64,
    ) {
        let (label, n, eps) = (self.label, self.n, epsilon::<T>());
        let decomposition =
            schur(&self.matrix.view()).unwrap_or_else(|error| panic!("{label}: schur: {error}"));
        let q = image_of(&decomposition.q(), 0);
        let t = image_of(&decomposition.t(), self.exponent);
        let defect = a_posteriori::gram_defect(&q, n, n);
        // Restoring T̂ rounds each entry once; ‖Q̂‖₂² ≤ 1 + δ.
        let residual = a_posteriori::residual(self.image, &q, &t, &q, n, n, n)
            + (1.0 + defect) * n as f64 * self.grid;
        let (eta_r, eta_o) = backward_error::schur_factors(n, eps);
        if let Some(eta) = informative(eta_r) {
            let bound = eta * self.norm();
            assert!(
                residual <= bound,
                "{label}: ‖Â − Q̂T̂Q̂ᵀ‖_F {residual:e} > {bound:e}"
            );
        }
        if let Some(eta) = informative(eta_o) {
            assert!(defect <= eta, "{label}: ‖Q̂ᵀQ̂ − I‖ {defect:e} > {eta:e}");
        }
        let blocks = a_posteriori::quasi_triangular_eigenvalues(&t, n);
        // The returned eigenvalues read off the same blocks in T: real parts
        // exact up to restoring, imaginary parts `√|b|·√|c|` within γ₃(ε(T)).
        for (z, &BlockEigenvalue { re, im, error }) in
            decomposition.eigenvalues().iter().zip(&blocks)
        {
            let (got_re, got_im) = (
                scale(z.re.to_f64(), -self.exponent),
                scale(z.im.to_f64(), -self.exponent),
            );
            let read_off = backward_error::gamma(3.0, eps) * im.abs() + error + self.grid;
            assert!(
                (got_re - re).abs() <= self.grid && (got_im - im).abs() <= read_off,
                "{label}: eigenvalue {got_re:e}+{got_im:e}i read off as {re:e}+{im:e}i"
            );
        }
        let Some(condition) = condition else {
            return;
        };
        let radius = a_posteriori::schur_certificate(residual, defect, &t);
        for BlockEigenvalue { re, im, error } in blocks {
            let distance = nearest(reference, re, im);
            let bound = condition * radius + reference_error + error;
            assert!(
                distance <= bound,
                "{label}: λ(T̂) {re:e}+{im:e}i is {distance:e} from the spectrum, certified {bound:e} (κ {condition:e}, residual {residual:e}, δ {defect:e})"
            );
        }
    }
}

fn check_case<T: Format>(family: Family, n: usize, unit_values: &[f64], exponent: i32) {
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
    let full = case.svd_factors(&reference_sigmas, reference_error);
    let values = singular_values(&matrix.view())
        .unwrap_or_else(|error| panic!("{label}: singular_values: {error}"));
    if let Some(eta) = informative(backward_error::svd(n, n, eps)) {
        let agreement = 2.0 * eta * norm + 2.0 * case.grid;
        let weyl = eta * norm + reference_error + case.grid;
        for ((a, b), r) in values.iter().zip(&full).zip(&reference_sigmas) {
            let a = scale(a.to_f64(), -exponent);
            assert!(
                (a - b).abs() <= agreement,
                "{label}: singular_values {a:e} vs svd_decompose {b:e}, bound {agreement:e}"
            );
            assert!(
                (a - r).abs() <= weyl,
                "{label}: σ {a:e} vs {r:e}, bound {weyl:e}"
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
    case.schur_factors(&reference_spectrum, condition, reference_error);
    let spectrum =
        eigenvalues(&matrix.view()).unwrap_or_else(|error| panic!("{label}: eigenvalues: {error}"));
    let (Some(condition), Some(eta)) = (condition, informative(backward_error::francis(n, eps)))
    else {
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
            "{label}: eigenvalues() {re:e}+{im:e}i is {distance:e} from the spectrum, bound {bound:e} (κ {condition:e})"
        );
    }
}

fn check_scan<T: Format>(seed: u64) {
    let mut rng = Xorshift64::new(seed);
    for exponent in T::MIN_EXPONENT..=(T::MAX_EXPONENT - 3) {
        for (index, family) in FAMILIES.into_iter().enumerate() {
            let n = 2 + (exponent.unsigned_abs() as usize * 3 + index) % 7;
            let unit_values = family.matrix(n, &mut rng);
            check_case::<T>(family, n, &unit_values, exponent);
        }
    }
}

#[test]
fn random_families_converge_accurately_at_every_normal_magnitude() {
    check_scan::<f64>(0x5EED_0064);
    check_scan::<f32>(0x5EED_0032);
    check_scan::<F16>(0x5EED_0016);
    check_scan::<Bf16>(0x5EED_BF16);
}

/// Orders past where `dbdsqr`'s `maxitr·n²·unfl` floor passes the gate's
/// upper end in `F16` (`k ≳ 24`; the fifth review found `svd_decompose`
/// returning a false `Overflow` at order 48): three matrices per order,
/// entries uniform in `[−1, 1]`, must factor and agree with the `f64`
/// factorization of their image.
fn check_large_orders<T: Format>(orders: &[usize], seed: u64) {
    let mut rng = Xorshift64::new(seed);
    for &n in orders {
        for _ in 0..3 {
            let unit_values: Vec<f64> = (0..n * n).map(|_| unit(&mut rng)).collect();
            check_case::<T>(Family::RankDeficient, n, &unit_values, 0);
        }
    }
}

#[test]
fn large_orders_factor_in_the_narrow_formats() {
    check_large_orders::<F16>(&[16, 24, 32, 48], 0xA11CE);
    check_large_orders::<Bf16>(&[32, 64], 0xA11CE);
}

/// `diag(1, B)` with `B` a random upper-Hessenberg block of subnormal
/// entries (integers up to 64 times the smallest subnormal, order 3 to 6): no
/// gate moves a matrix whose largest entry is `1`, so the block's converging
/// subdiagonals stay in the subnormals, where the relative deflation test
/// holds only for an exact zero and the sweep cycles. The absolute deflation
/// floors (`2^⌈log₂ n⌉·safmin` in both iterations) deflate them. Every
/// singular value and eigenvalue of `B` is at most `‖B‖_F`, and deflating
/// perturbs it by at most the floor, so the results other than the exact `1`
/// stay within `‖B‖_F + 2³·safmin`.
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
        let tolerance = block_norm + T::ONE.scale_binary(T::MIN_EXPONENT + 3).to_f64();
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
