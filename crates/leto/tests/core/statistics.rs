//! Covariance / correlation: closed-form oracles + variance cross-checks.
//!
//! Convention: rows are variables, columns observations (ndarray-stats /
//! numpy `rowvar = true`). Expected values are derived by hand from
//! `C[i,j] = (1/(n−ddof)) Σₖ (xᵢₖ−x̄ᵢ)(xⱼₖ−x̄ⱼ)`.

use leto::application::reduction::var_axis;
use leto::application::statistics::{covariance, pearson_correlation};
use leto::{Array2, LetoError, Storage};

#[track_caller]
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
        "actual {actual} expected {expected}"
    );
}

fn leto(shape: [usize; 2], data: Vec<f64>) -> Array2<f64> {
    Array2::from_shape_vec(shape, data).unwrap()
}

#[test]
fn covariance_closed_form_sample_and_population() {
    // 2 variables, 3 observations: A=[1,2,3] (mean 2), B=[4,6,8] (mean 6).
    // centered A=[-1,0,1], B=[-2,0,2].
    let a = leto([2, 3], vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0]);

    // Sample (ddof=1): C_AA=2/2=1, C_BB=8/2=4, C_AB=(2+0+2)/2=2.
    let c1 = covariance(&a, 1.0).unwrap();
    let d1 = c1.storage().as_slice();
    assert_close(d1[0], 1.0); // [0,0]
    assert_close(d1[1], 2.0); // [0,1]
    assert_close(d1[2], 2.0); // [1,0] (symmetric)
    assert_close(d1[3], 4.0); // [1,1]

    // Population (ddof=0): divide by 3 instead of 2.
    let c0 = covariance(&a, 0.0).unwrap();
    let d0 = c0.storage().as_slice();
    assert_close(d0[0], 2.0 / 3.0);
    assert_close(d0[1], 4.0 / 3.0);
    assert_close(d0[3], 8.0 / 3.0);
}

#[test]
fn covariance_diagonal_equals_variance() {
    // Diagonal of the covariance matrix is the per-variable variance.
    // var_axis along axis 1 reduces over observations → one variance per row.
    let a = leto(
        [3, 4],
        vec![1.0, 3.0, 2.0, 5.0, 2.0, 2.0, 2.0, 2.0, 9.0, 1.0, 4.0, 7.0],
    );
    for &ddof in &[0.0_f64, 1.0] {
        let cov = covariance(&a, ddof).unwrap();
        let cd = cov.storage().as_slice();
        let var = var_axis::<f64, _, 2, 1>(&a, 1, ddof).unwrap();
        let vd = var.storage().as_slice();
        for i in 0..3usize {
            assert_close(cd[i * 3 + i], vd[i]);
        }
    }
}

#[test]
fn covariance_is_symmetric() {
    let a = leto([3, 5], (0..15).map(f64::from).collect());
    let cov = covariance(&a, 1.0).unwrap();
    let cd = cov.storage().as_slice();
    for i in 0..3usize {
        for j in 0..3usize {
            assert_close(cd[i * 3 + j], cd[j * 3 + i]);
        }
    }
}

#[test]
fn correlation_diagonal_unit_and_perfect_linear() {
    // B = 2A + 2 ⇒ perfectly positively correlated ⇒ R_AB = 1.
    let a = leto([2, 3], vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0]);
    let r = pearson_correlation(&a).unwrap();
    let rd = r.storage().as_slice();
    assert_close(rd[0], 1.0); // R_AA
    assert_close(rd[3], 1.0); // R_BB
    assert_close(rd[1], 1.0); // R_AB
    assert_close(rd[2], 1.0); // R_BA
}

#[test]
fn correlation_perfect_negative() {
    // B = -A ⇒ R_AB = -1.
    let a = leto([2, 3], vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
    let r = pearson_correlation(&a).unwrap();
    let rd = r.storage().as_slice();
    assert_close(rd[1], -1.0);
}

#[test]
fn correlation_bounded_and_matches_normalized_covariance() {
    // Independent-looking data: R[i,j] = C[i,j]/(σᵢσⱼ), |R| ≤ 1.
    let a = leto(
        [3, 4],
        vec![1.0, 2.0, 4.0, 3.0, 2.0, 1.0, 3.0, 5.0, 7.0, 1.0, 2.0, 4.0],
    );
    let cov = covariance(&a, 0.0).unwrap();
    let cd = cov.storage().as_slice();
    let r = pearson_correlation(&a).unwrap();
    let rd = r.storage().as_slice();
    for i in 0..3usize {
        for j in 0..3usize {
            let expected = cd[i * 3 + j] / (cd[i * 3 + i].sqrt() * cd[j * 3 + j].sqrt());
            assert_close(rd[i * 3 + j], expected);
            assert!(rd[i * 3 + j].abs() <= 1.0 + 1.0e-12);
        }
    }
}

#[test]
fn covariance_rejects_empty_and_excess_ddof() {
    let a = leto([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // n − ddof ≤ 0 with n=3.
    match covariance(&a, 3.0) {
        Err(LetoError::StorageError { reason }) => {
            assert_eq!(
                reason,
                "variance degrees of freedom (n - ddof) must be positive"
            );
        }
        other => panic!("expected dof error, got {other:?}"),
    }
    let empty: Array2<f64> = Array2::from_shape_vec([0, 0], vec![]).unwrap();
    match covariance(&empty, 0.0) {
        Err(LetoError::StorageError { reason }) => {
            assert_eq!(reason, "covariance over an empty matrix is undefined");
        }
        other => panic!("expected empty error, got {other:?}"),
    }
    match pearson_correlation(&empty) {
        Err(LetoError::StorageError { reason }) => {
            assert_eq!(reason, "covariance over an empty matrix is undefined");
        }
        other => panic!("expected propagated covariance error, got {other:?}"),
    }
}
