#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

//! Unscaled-factorization and deflation-floor checks.

use super::super::format::Format;
use super::{sorted_unit, N};
use eunomia::{Bf16, F16};
use leto::{Array2, Storage};
use leto_ops::{
    col_piv_qr, eigenvalues, schur, singular_values, svd_decompose, symmetric_eigen_qr,
};

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

/// The absolute deflation floor stays below `ε·‖A‖_F`
/// (`thresholds::homogeneous_safe_range`). `dbdsqr`'s `6k²·safmin` does not
/// in `F16`: at `k = 24` it is `6·24²·2⁻¹⁴ ≈ 0.21`, and would split every
/// superdiagonal below that of a unit-scale matrix. Twelve copies of the block
/// `[[1, a], [0, 1]]`, `a = 1/16` (already bidiagonal, so the reduction
/// applies identity reflectors; `‖A‖_max = 1` lies inside the gate's range,
/// lower end `√smlnum = ¼`, and is not moved), have singular values
/// `√(1 + a²/4) ± a/2 ≈ 1.0317, 0.9692`; a split returns `1` for both. The
/// check separates the two structurally: every `σ̂` more than `a/4` from `1`
/// (the exact values are `≥ 0.030` away, the split `0`).
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
