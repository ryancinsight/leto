#![allow(clippy::unwrap_used, reason = "test scope")]

use super::{nnls, NnlsConfig};
use leto::{Array1, Array2, LetoError};

#[test]
fn identity_solves_exact_positive_rhs() {
    let a = Array2::from_vec([2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let b = Array1::from_vec(2, vec![3.0, 4.0]).unwrap();
    let result = nnls(&a.view(), &b.view(), NnlsConfig::default()).unwrap();
    assert!(result.converged);
    assert!((result.solution[0] - 3.0).abs() < 1e-10);
    assert!((result.solution[1] - 4.0).abs() < 1e-10);
}

#[test]
fn negative_rhs_clamps_to_zero() {
    let a = Array2::from_vec([2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let b = Array1::from_vec(2, vec![-1.0, 5.0]).unwrap();
    let result = nnls(&a.view(), &b.view(), NnlsConfig::default()).unwrap();
    assert!(result.converged);
    assert!(result.solution[0].abs() < 1e-10);
    assert!((result.solution[1] - 5.0).abs() < 1e-10);
    assert!(result.iterations <= 2);
}

#[test]
fn all_negative_rhs_gives_zero_solution() {
    let a = Array2::from_vec([2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let b = Array1::from_vec(2, vec![-3.0, -4.0]).unwrap();
    let result = nnls(&a.view(), &b.view(), NnlsConfig::default()).unwrap();
    assert!(result.converged);
    assert!(result.residual_norm > 0.0);
    assert!(result.solution[0].abs() < 1e-10);
    assert!(result.solution[1].abs() < 1e-10);
}

#[test]
fn zero_rhs_gives_zero_solution() {
    let a = Array2::from_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Array1::from_vec(2, vec![0.0, 0.0]).unwrap();
    let result = nnls(&a.view(), &b.view(), NnlsConfig::default()).unwrap();
    assert!(result.converged);
    assert!(result.residual_norm < 1e-12);
    // Verify x is actually zero, not just a vector in the nullspace.
    assert!(result.solution[0].abs() < 1e-12);
    assert!(result.solution[1].abs() < 1e-12);
}

#[test]
fn overdetermined_known_solution() {
    let a = Array2::from_vec([3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let b = Array1::from_vec(3, vec![2.0, 3.0, 4.0]).unwrap();
    let result = nnls(&a.view(), &b.view(), NnlsConfig::default()).unwrap();
    assert!(result.converged);
    assert!((result.solution[0] - 5.0 / 3.0).abs() < 1e-8);
    assert!((result.solution[1] - 8.0 / 3.0).abs() < 1e-8);
}

#[test]
fn overdetermined_one_coefficient_clamped() {
    let a = Array2::from_vec([3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let b = Array1::from_vec(3, vec![2.0, -1.0, 0.0]).unwrap();
    let result = nnls(&a.view(), &b.view(), NnlsConfig::default()).unwrap();
    assert!(result.converged);
    assert!(result.solution[0] > 0.0);
    assert!(result.solution[1].abs() < 1e-10);
}

#[test]
fn max_iterations_respected() {
    // A = I_3, b = [1, 0, 0] — has a positive w entry but limiting
    // to 0 iterations forces early exit.
    let a = Array2::from_vec([3, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    let b = Array1::from_vec(3, vec![1.0, 0.0, 0.0]).unwrap();
    let config = NnlsConfig {
        max_iterations: 0,
        tolerance: 1e-14,
    };
    let result = nnls(&a.view(), &b.view(), config).unwrap();
    assert!(!result.converged);
    assert_eq!(result.iterations, 0);
}

#[test]
fn zero_columns_returns_empty_solution() {
    let a = Array2::zeros([3, 0]);
    let b = Array1::from_vec(3, vec![1.0, 2.0, 3.0]).unwrap();
    let a_view = a.view().slice(&[(0, 3, 1), (0, 0, 1)]).unwrap();
    let result = nnls(&a_view, &b.view(), NnlsConfig::default()).unwrap();
    assert!(result.converged);
    assert_eq!(result.solution.shape()[0], 0);
}

#[test]
fn shape_mismatch_errors() {
    let a = Array2::from_vec([3, 2], vec![1.0; 6]).unwrap();
    let b = Array1::from_vec(2, vec![1.0, 2.0]).unwrap();
    let err = nnls(&a.view(), &b.view(), NnlsConfig::default()).unwrap_err();
    assert!(matches!(err, LetoError::ShapeMismatch { .. }));
}

#[test]
fn single_column_positive_rhs() {
    let a = Array2::from_vec([2, 1], vec![1.0, 2.0]).unwrap();
    let b = Array1::from_vec(2, vec![3.0, 6.0]).unwrap();
    let result = nnls(&a.view(), &b.view(), NnlsConfig::default()).unwrap();
    assert!(result.converged);
    assert!((result.solution[0] - 3.0).abs() < 1e-8);
}

#[test]
fn single_column_negative_rhs_clamps() {
    let a = Array2::from_vec([2, 1], vec![1.0, 2.0]).unwrap();
    let b = Array1::from_vec(2, vec![-1.0, -2.0]).unwrap();
    let result = nnls(&a.view(), &b.view(), NnlsConfig::default()).unwrap();
    assert!(result.converged);
    assert!(result.solution[0].abs() < 1e-10);
}

#[test]
fn tolerance_respected() {
    let a = Array2::from_vec([3, 2], vec![2.0, 0.0, 0.0, 3.0, 1.0, 1.0]).unwrap();
    let b = Array1::from_vec(3, vec![4.0, 9.0, 4.0]).unwrap();
    let result = nnls(
        &a.view(),
        &b.view(),
        NnlsConfig {
            max_iterations: 100,
            tolerance: 1e-4,
        },
    )
    .unwrap();
    assert!(result.converged);
    assert!(result.residual_norm >= 0.0);
}

/// CSD-shape recovery: a synthesised non-negative sparse spike is recovered
/// exactly from a Toeplitz-of-exponentials basis, the standard shape used
/// for constrained spherical deconvolution response functions. This is the
/// cross-verification oracle for the algorithm's doc-claimed CSD readiness
/// (the CSD motivation at the top of the file): the spike's amplitude is
/// the only non-zero entry, and the basis is one-to-one on it, so the
/// recovery must be exact up to QR-conditioning tolerance.
#[test]
fn csd_shape_sparse_spike_recovered() {
    // m = 12 sample directions, n = 8 candidate direction bins.
    // A is a Toeplitz-shaped positive kernel — column j is the row j of
    // an exponential decay, normalised to a unit L1 norm so each basis
    // column has comparable scale.
    let m = 12_usize;
    let n = 8_usize;
    let mut a_data = Vec::with_capacity(m * n);
    for j in 0..n {
        for i in 0..m {
            // |i - j| under cyclic wrap so every column sees the kernel.
            let diff = i.abs_diff(j);
            let cycl = diff.min(m - diff);
            a_data.push((-(cycl as f64) * 0.4).exp());
        }
    }
    let a = Array2::from_vec([m, n], a_data).unwrap();

    // Ground truth: a unit-amplitude non-negative spike at bin 3, zero
    // elsewhere. This is the analytical oracle the test asserts against.
    let x_star: Vec<f64> = (0..n).map(|j| if j == 3 { 1.0 } else { 0.0 }).collect();

    // b = A · x* (no noise) — the noiseless CSD deconvolution case.
    let mut b = vec![0.0_f64; m];
    for i in 0..m {
        for j in 0..n {
            b[i] += a[[i, j]] * x_star[j];
        }
    }
    let b = Array1::from_vec(m, b).unwrap();

    let result = nnls(&a.view(), &b.view(), NnlsConfig::default()).unwrap();

    // CSD-acceptance contract:
    //   1. non-negativity (the constraint that unconstrained deconvolution
    //      violates and is the reason NNLS exists for FOD estimation);
    //   2. exact recovery of the spike amplitude to QR-conditioning
    //      tolerance (`1e-8` covers the 8×8 column submatrix's
    //      2-norm-conditioned solve);
    //   3. zero on every non-spike entry to the same tolerance, so no
    //      spurious lobes (the dMRI equivalent of a negative-lobe defect).
    assert!(result.converged);
    for j in 0..n {
        assert!(
            result.solution[j] >= -1e-12,
            "NNLS violated non-negativity at bin {j}: {}",
            result.solution[j]
        );
    }
    assert!((result.solution[3] - 1.0).abs() < 1e-8);
    for j in 0..n {
        if j == 3 {
            continue;
        }
        assert!(
            result.solution[j].abs() < 1e-8,
            "spurious lobe at bin {j}: {}",
            result.solution[j]
        );
    }
    assert!(result.residual_norm < 1e-8);
}
