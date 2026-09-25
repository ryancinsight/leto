//! Seeded random scan at every binary exponent from each format's smallest
//! normal to three binades below its largest, over four families of order
//! 2 to 8:
//!
//! - **graded** — general, row `i` graded by `2⁻²ⁱ`, so the small singular
//!   values and the converging off-diagonals reach the subnormals at the
//!   bottom of the range;
//! - **clustered** — symmetric `Q·diag(1, 1, 1.001, 1.001, …, 0.5)·Qᵀ`, pairs
//!   of equal eigenvalues `10⁻³` apart;
//! - **rank-deficient** — general `X·Y` of rank `⌊n/2⌋`, and its symmetric
//!   counterpart `Q·diag(d, 0)·Qᵀ`.
//!
//! Every routine must converge, and every value is checked against the `f64`
//! factorization of the same image (so the input rounding is zero): singular
//! values and symmetric eigenvalues by Weyl within `(η(ε(T)) + η(ε₆₄))·‖Â‖_F`,
//! general eigenvalues by Bauer–Fike within `κ·(η(ε(T)) + η(ε₆₄))·‖Â‖_F` of
//! some reference eigenvalue, `κ` from the reference's left and right
//! eigenvectors (`spectral_condition`), `η` the derived backward errors of
//! `backward_error.rs`. `singular_values` and `svd_decompose` must agree
//! within `2η(ε(T))·‖Â‖_F`. Restoring a result onto `T`'s grid adds half the
//! subnormal spacing per value.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::backward_error;
use super::format::{epsilon, Format};
use super::spectral_condition::condition_by_inverse_iteration;
use eunomia::{Bf16, F16};
use leto::Array2;
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

#[derive(Clone, Copy, PartialEq)]
enum Family {
    Graded,
    Clustered,
    RankDeficient,
    SymmetricRankDeficient,
}

const FAMILIES: [Family; 4] = [
    Family::Graded,
    Family::Clustered,
    Family::RankDeficient,
    Family::SymmetricRankDeficient,
];

impl Family {
    fn symmetric(self) -> bool {
        matches!(self, Self::Clustered | Self::SymmetricRankDeficient)
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

fn check_case<T: Format>(family: Family, n: usize, unit_values: &[f64], exponent: i32) {
    let (matrix, image) = place::<T>(unit_values, n, exponent);
    let label = format!("2^{exponent} n = {n}");
    let eps = epsilon::<T>();
    let norm = frobenius(&image);
    // Half the subnormal spacing, in units of 2^e.
    let grid = 0.5 * scale(1.0, T::MIN_EXPONENT - T::PRECISION + 1 - exponent);
    let reference = Array2::from_shape_vec([n, n], image.clone()).unwrap();

    // Singular values: both entry points, against each other and the f64 image.
    let values = singular_values(&matrix.view())
        .unwrap_or_else(|error| panic!("{label}: singular_values: {error}"));
    let full = svd_decompose(&matrix.view())
        .unwrap_or_else(|error| panic!("{label}: svd_decompose: {error}"));
    let reference_sigmas = singular_values(&reference.view()).unwrap();
    let eta = backward_error::svd(n, n, eps);
    let agreement = 2.0 * eta * norm + 2.0 * grid;
    let weyl = (eta + backward_error::svd(n, n, f64::EPSILON)) * norm + grid;
    for ((a, b), r) in values
        .iter()
        .zip(&full.singular_values)
        .zip(&reference_sigmas)
    {
        let (a, b) = (scale(a.to_f64(), -exponent), scale(b.to_f64(), -exponent));
        assert!(
            (a - b).abs() <= agreement,
            "{label}: singular_values {a:e} vs svd_decompose {b:e}, bound {agreement:e}"
        );
        assert!(
            (a - r).abs() <= weyl,
            "{label}: σ {a:e} vs {r:e}, bound {weyl:e}"
        );
    }

    if family == Family::RankDeficient {
        // A repeated zero eigenvalue: no Bauer–Fike factor exists for the
        // general rank-deficient family; the symmetric one covers it.
        for result in [
            eigenvalues(&matrix.view()),
            schur(&matrix.view()).map(|decomposition| decomposition.eigenvalues()),
        ] {
            result.unwrap_or_else(|error| panic!("{label}: {error}"));
        }
        return;
    }
    let reference_spectrum: Vec<(f64, f64)> = eigenvalues(&reference.view())
        .unwrap()
        .iter()
        .map(|z| (z.re, z.im))
        .collect();
    let eta = backward_error::francis(n, eps) + backward_error::francis(n, f64::EPSILON);
    let condition = if family.symmetric() {
        1.0
    } else {
        condition_by_inverse_iteration(&image, n, &reference_spectrum)
    };
    let bound = condition * eta * norm + grid;
    for result in [
        eigenvalues(&matrix.view()),
        schur(&matrix.view()).map(|decomposition| decomposition.eigenvalues()),
    ] {
        let spectrum = result.unwrap_or_else(|error| panic!("{label}: {error}"));
        let mut computed: Vec<(f64, f64)> = spectrum
            .iter()
            .map(|z| {
                (
                    scale(z.re.to_f64(), -exponent),
                    scale(z.im.to_f64(), -exponent),
                )
            })
            .collect();
        if family.symmetric() {
            // Weyl: sorted real parts pair off; imaginary parts are errors.
            let mut expected: Vec<f64> = reference_spectrum.iter().map(|z| z.0).collect();
            computed.sort_by(|a, b| a.0.total_cmp(&b.0));
            expected.sort_by(f64::total_cmp);
            for ((re, im), e) in computed.iter().zip(&expected) {
                assert!(
                    (re - e).abs() <= bound && im.abs() <= bound,
                    "{label}: λ {re:e}+{im:e}i vs {e:e}, bound {bound:e}"
                );
            }
        } else {
            // Bauer–Fike: each computed eigenvalue near some reference one.
            for (re, im) in computed {
                let nearest = reference_spectrum
                    .iter()
                    .map(|(r, i)| (re - r).hypot(im - i))
                    .fold(f64::INFINITY, f64::min);
                assert!(
                    nearest <= bound,
                    "{label}: λ {re:e}+{im:e}i is {nearest:e} from the spectrum, bound {bound:e} (κ {condition:e})"
                );
            }
        }
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
