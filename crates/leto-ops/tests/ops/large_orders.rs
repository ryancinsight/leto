//! Structured `F16` matrices at orders 64 to 640 — a single nonzero entry,
//! graded diagonals, `½·I`, a diagonal near `2⁻¹²`, and all-ones — where the
//! matrix-tier gates must neither refuse an input the kernels factor nor
//! let a floor deflation leave the backward error.
//!
//! The gates admit an input once `√k·safmin ≤ ε·‖A‖_F` is secured through
//! `2^l ≤ ‖A‖_F/‖A‖_max` (`linalg::thresholds::deflation_count_log2`), so
//! every entry point factors each family here. `schur` and `svd_decompose`
//! are certified a posteriori (`a_posteriori.rs`): every eigenvalue of the
//! returned `T̂` and every returned singular value within the radius the
//! measured residual and orthogonality give (`κ = 1`, the families are
//! symmetric) of the exact spectrum. `eigenvalues` and `singular_values`
//! return no factors to certify; they must return finite values.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::a_posteriori::{self, BlockEigenvalue};
use eunomia::F16;
use leto::{Array2, Storage};
use leto_ops::{eigenvalues, schur, singular_values, svd_decompose};

fn as_f64(m: &Array2<F16>) -> Vec<f64> {
    m.storage()
        .as_slice()
        .iter()
        .map(|v| f64::from(v.to_f32()))
        .collect()
}

/// A symmetric structured family of order `n` and its exact spectrum
/// (eigenvalues and singular values coincide in magnitude; all are `≥ 0`).
fn family(name: &str, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![0.0; n * n];
    let spectrum: Vec<f64> = match name {
        "single" => {
            a[0] = 1.5;
            (0..n).map(|i| if i == 0 { 1.5 } else { 0.0 }).collect()
        }
        "diagonal" => (0..n).map(|i| 1.0 + (i % 64) as f64 / 64.0).collect(),
        "half-identity" => vec![0.5; n],
        "small-diagonal" => (0..n)
            .map(|i| (1.0 + (i % 8) as f64 / 8.0) * 2.0_f64.powi(-12))
            .collect(),
        "ones" => {
            a.fill(1.0);
            (0..n)
                .map(|i| if i == 0 { n as f64 } else { 0.0 })
                .collect()
        }
        _ => unreachable!("unknown family {name}"),
    };
    if !matches!(name, "single" | "ones") {
        for (i, d) in spectrum.iter().enumerate() {
            a[i * n + i] = *d;
        }
    }
    (a, spectrum)
}

/// Distance from `x` to the nearest point of `spectrum`.
fn nearest(spectrum: &[f64], re: f64, im: f64) -> f64 {
    spectrum
        .iter()
        .map(|s| (re - s).hypot(im))
        .fold(f64::INFINITY, f64::min)
}

fn check(name: &str, n: usize) {
    let (values, spectrum) = family(name, n);
    let label = format!("{name} n = {n}");
    let narrowed: Vec<F16> = values.iter().map(|&v| F16::from_f64(v)).collect();
    let image: Vec<f64> = narrowed.iter().map(|v| f64::from(v.to_f32())).collect();
    assert_eq!(image, values, "{label}: every entry is exact in F16");
    let matrix = Array2::from_shape_vec([n, n], narrowed).unwrap();

    for z in eigenvalues(&matrix.view()).unwrap_or_else(|e| panic!("{label}: eigenvalues: {e}")) {
        assert!(
            z.re.to_f32().is_finite() && z.im.to_f32().is_finite(),
            "{label}"
        );
    }
    for s in
        singular_values(&matrix.view()).unwrap_or_else(|e| panic!("{label}: singular_values: {e}"))
    {
        assert!(s.to_f32().is_finite(), "{label}");
    }

    let decomposition = schur(&matrix.view()).unwrap_or_else(|e| panic!("{label}: schur: {e}"));
    let (q, t) = (as_f64(&decomposition.q()), as_f64(&decomposition.t()));
    let residual = a_posteriori::residual(&image, &q, &t, &q, n, n, n);
    let radius = a_posteriori::schur_certificate(residual, a_posteriori::gram_defect(&q, n, n), &t);
    for BlockEigenvalue { re, im, error } in a_posteriori::quasi_triangular_eigenvalues(&t, n) {
        let distance = nearest(&spectrum, re, im);
        assert!(
            distance <= radius + error,
            "{label}: λ(T̂) {re}+{im}i is {distance:e} from the spectrum, certified {radius:e}"
        );
    }

    let full = svd_decompose(&matrix.view()).unwrap_or_else(|e| panic!("{label}: svd: {e}"));
    let sigmas: Vec<f64> = full
        .singular_values
        .iter()
        .map(|v| f64::from(v.to_f32()))
        .collect();
    let (u, v) = (
        as_f64(&full.left_singular_vectors),
        as_f64(&full.right_singular_vectors),
    );
    let mut diagonal = vec![0.0; n * n];
    for (i, sigma) in sigmas.iter().enumerate() {
        diagonal[i * n + i] = *sigma;
    }
    let residual = a_posteriori::residual(&image, &u, &diagonal, &v, n, n, n);
    let radius = a_posteriori::svd_certificate(
        residual,
        a_posteriori::gram_defect(&u, n, n),
        a_posteriori::gram_defect(&v, n, n),
        &sigmas,
    );
    let mut exact = spectrum;
    exact.sort_by(|a, b| b.total_cmp(a));
    for (i, (sigma, expected)) in sigmas.iter().zip(&exact).enumerate() {
        assert!(
            (sigma - expected).abs() <= radius,
            "{label}: σ{i} {sigma} vs {expected}, certified {radius:e}"
        );
    }
}

const FAMILIES: [&str; 4] = ["single", "diagonal", "half-identity", "small-diagonal"];

#[test]
fn f16_all_ones_factor_at_orders_64_to_256() {
    for n in [64, 96, 128, 256] {
        check("ones", n);
    }
}

#[test]
fn f16_structured_families_factor_at_order_256() {
    for name in FAMILIES {
        check(name, 256);
    }
}

#[test]
fn f16_structured_families_factor_at_order_384() {
    for name in FAMILIES {
        check(name, 384);
    }
}

#[test]
fn f16_structured_families_factor_at_order_512() {
    for name in FAMILIES {
        check(name, 512);
    }
}

#[test]
fn f16_single_entry_and_diagonal_factor_at_order_640() {
    for name in &FAMILIES[..2] {
        check(name, 640);
    }
}

#[test]
fn f16_scaled_identities_factor_at_order_640() {
    for name in &FAMILIES[2..] {
        check(name, 640);
    }
}
