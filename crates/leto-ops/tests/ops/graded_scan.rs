//! Seeded graded random scan: at every binary exponent from each format's
//! smallest normal to three binades below its largest, random `n × n`
//! matrices (`n ∈ {2, 3, 4}`) whose row `i` is graded by `2⁻²ⁱ` must factor
//! without a non-convergence or a non-finite `Ok` in `singular_values`,
//! `svd_decompose`, `eigenvalues` and `schur`, and the two SVD entry points
//! must agree.
//!
//! Grading pushes the trailing rows — and with them the small singular
//! values and converging off-diagonals — up to `2^−(2n−2)` below the
//! magnitude, so at the bottom of the range they cross into the subnormals:
//! the case a purely relative deflation test cannot resolve (LAPACK `dbdsqr`
//! and `dlahqr` add an absolute underflow threshold for it).
//!
//! # Agreement bound
//!
//! Both SVD entry points run the same bidiagonal QR but reduce through
//! different bidiagonalizations (column-major values-only, row-major with
//! factors), so their singular values differ by rounding: by Weyl, each is
//! within its backward error `η·‖Â‖_F` of the exact `σ(Â)`, so they are within
//! `2η·‖Â‖_F` of each other, `η = n²·ε` the empirical envelope of
//! `scale_range.rs`. Each also carries the deflation floor, at most
//! `ε·‖Â‖_max` after the gate (`svd/bidiagonal_qr.rs`), once per deflation:
//! `n` deflations, `2n·ε·‖Â‖_max` for the pair. Restoring each result onto
//! `T`'s grid rounds it by at most half the subnormal spacing, one spacing
//! for the pair.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::format::{epsilon, Format};
use eunomia::{Bf16, F16};
use leto::{Array2, Storage};
use leto_ops::{eigenvalues, schur, singular_values, svd_decompose, Xorshift64};

/// Matrices per exponent.
const PER_EXPONENT: usize = 12;

fn check_graded_scan<T: Format>(seed: u64) {
    let mut rng = Xorshift64::new(seed);
    let eps = epsilon::<T>();
    // The subnormal spacing: restoring a result onto `T`'s grid rounds it by
    // at most half of it, once per entry point.
    let spacing = 2.0_f64.powi(T::MIN_EXPONENT - T::PRECISION + 1);
    for exponent in T::MIN_EXPONENT..=(T::MAX_EXPONENT - 3) {
        for case in 0..PER_EXPONENT {
            let n = 2 + case % 3;
            let values: Vec<T> = (0..n * n)
                .map(|k| {
                    let grade = -2 * i32::try_from(k / n).unwrap();
                    let unit = 2.0 * rng.next_unit_f64() - 1.0;
                    T::from_f64(unit).mul(T::ONE.scale_binary(exponent + grade))
                })
                .collect();
            let label = format!("2^{exponent} case {case} (n = {n})");
            let matrix = Array2::from_shape_vec([n, n], values).unwrap();
            let image: Vec<f64> = matrix
                .storage()
                .as_slice()
                .iter()
                .map(|v| v.to_f64())
                .collect();
            let largest = image.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
            // Scaled by the largest entry: the f64 squares of f64-subnormal
            // magnitudes would underflow.
            let frobenius = largest
                * image
                    .iter()
                    .map(|v| (v / largest).powi(2))
                    .sum::<f64>()
                    .sqrt();

            let values = singular_values(&matrix.view())
                .unwrap_or_else(|error| panic!("{label}: singular_values: {error}"));
            let full = svd_decompose(&matrix.view())
                .unwrap_or_else(|error| panic!("{label}: svd_decompose: {error}"));
            let nn = (n * n) as f64;
            let bound = 2.0 * nn * eps * frobenius + 2.0 * n as f64 * eps * largest + spacing;
            for (a, b) in values.iter().zip(&full.singular_values) {
                let (a, b) = (a.to_f64(), b.to_f64());
                assert!(a.is_finite() && b.is_finite(), "{label}: {a} vs {b}");
                assert!(
                    (a - b).abs() <= bound,
                    "{label}: singular_values {a:e} vs svd_decompose {b:e}, bound {bound:e}"
                );
            }
            for result in [
                eigenvalues(&matrix.view()),
                schur(&matrix.view()).map(|decomposition| decomposition.eigenvalues()),
            ] {
                let spectrum = result.unwrap_or_else(|error| panic!("{label}: {error} {image:?}"));
                assert!(
                    spectrum
                        .iter()
                        .all(|z| z.re.is_finite() && z.im.is_finite()),
                    "{label}: non-finite eigenvalue"
                );
            }
        }
    }
}

#[test]
fn graded_random_matrices_converge_at_every_normal_magnitude() {
    check_graded_scan::<f64>(0x5EED_0064);
    check_graded_scan::<f32>(0x5EED_0032);
    check_graded_scan::<F16>(0x5EED_0016);
    check_graded_scan::<Bf16>(0x5EED_BF16);
}

/// `diag(1, B)` with `B` a random upper-Hessenberg block of subnormal
/// entries (integers up to 64 times the smallest subnormal, order 3 to 6): no
/// gate moves a matrix whose largest entry is `1`, so the block's converging
/// subdiagonals stay in the subnormals, where the relative deflation test
/// holds only for an exact zero and the sweep cycles. The absolute deflation
/// floors (`dbdsqr`'s `maxitr·n²·unfl` in the SVD, `2^⌈log₂ n⌉·safmin` in
/// Francis) deflate them. Every singular value and eigenvalue of `B` is at
/// most `‖B‖_F`, and deflating perturbs it by at most the floor, so the
/// results other than the exact `1` stay within `‖B‖_F + 2⁹·safmin`
/// (`6·7² ≤ 2⁹`).
fn check_subnormal_block_deflates<T: Format>(seed: u64) {
    let unit = T::ONE.scale_binary(T::MIN_EXPONENT - T::PRECISION + 1);
    let mut rng = Xorshift64::new(seed);
    let label = std::any::type_name::<T>();
    for case in 0..40 {
        let k = 4 + case % 4;
        let mut values = vec![T::ZERO; k * k];
        values[0] = T::ONE;
        for i in 1..k {
            for j in (i - 1).max(1)..k {
                let integer = (64.0 * (2.0 * rng.next_unit_f64() - 1.0)).round();
                values[i * k + j] = T::from_f64(integer).mul(unit);
            }
        }
        let block_norm = values[1..]
            .iter()
            .map(|v| v.to_f64().powi(2))
            .sum::<f64>()
            .sqrt();
        let tolerance = block_norm + T::ONE.scale_binary(T::MIN_EXPONENT + 9).to_f64();
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
            let rest = parts.iter().copied().filter(|z| !is_unit(z));
            assert_eq!(unit_eigenvalues, 1, "{case}: λ = 1");
            for (re, im) in rest {
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
