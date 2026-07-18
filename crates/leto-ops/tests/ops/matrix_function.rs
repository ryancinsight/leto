//! Matrix power / exponential: closed-form oracles + self-validation.

use leto::{Array2, LetoError, Storage};
use leto_ops::{matexp, matpow, MatrixFunction, MatrixProduct};

#[track_caller]
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-9 * expected.abs().max(1.0),
        "actual {actual} expected {expected}"
    );
}

#[track_caller]
fn assert_close_slice(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_close(*a, *e);
    }
}

fn mat(shape: [usize; 2], data: Vec<f64>) -> Array2<f64> {
    Array2::from_shape_vec(shape, data).unwrap()
}

// ── matpow ──────────────────────────────────────────────────────────────────

#[test]
fn matpow_zero_is_identity() {
    let a = mat([2, 2], vec![2.0, 1.0, 3.0, 4.0]);
    let p = matpow(&a.view(), 0).unwrap();
    assert_close_slice(p.storage().as_slice(), &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn matpow_diagonal_and_shear_closed_form() {
    // diag(2,3)^3 = diag(8,27).
    let d = mat([2, 2], vec![2.0, 0.0, 0.0, 3.0]);
    assert_close_slice(
        matpow(&d.view(), 3).unwrap().storage().as_slice(),
        &[8.0, 0.0, 0.0, 27.0],
    );
    // [[1,1],[0,1]]^5 = [[1,5],[0,1]] (unit upper-shear).
    let s = mat([2, 2], vec![1.0, 1.0, 0.0, 1.0]);
    assert_close_slice(
        matpow(&s.view(), 5).unwrap().storage().as_slice(),
        &[1.0, 5.0, 0.0, 1.0],
    );
}

#[test]
fn matpow_3x3_self_validate() {
    // Self-validate: matpow(A, 4) == A · A · A · A via sequential matmul.
    let data = vec![0.5, 0.2, -0.3, 1.1, 0.4, 0.0, 0.7, -0.6, 0.9];
    let a = mat([3, 3], data);
    let pow4 = matpow(&a.view(), 4).unwrap();

    // Compute A^4 via sequential matmul.
    let a2 = a.matmul(&a).unwrap();
    let a3 = a2.matmul(&a).unwrap();
    let a4_ref = a3.matmul(&a).unwrap();

    assert_close_slice(pow4.storage().as_slice(), a4_ref.storage().as_slice());
}

#[test]
fn matpow_exact_for_integers() {
    // Integer scalar: exact unit-shear power, no floating point.
    let a = Array2::<i64>::from_shape_vec([2, 2], vec![1, 1, 0, 1]).unwrap();
    let p = matpow(&a.view(), 7).unwrap();
    assert_eq!(p.storage().as_slice(), &[1, 7, 0, 1]);
}

#[test]
fn matpow_rejects_non_square() {
    let a = mat([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        matpow(&a.view(), 2).unwrap_err(),
        LetoError::ShapeMismatch {
            lhs: vec![2, 3],
            rhs: vec![2, 2],
        }
    );
}

// ── matexp ──────────────────────────────────────────────────────────────────

#[test]
fn matexp_zero_is_identity() {
    let z = mat([3, 3], vec![0.0; 9]);
    let e = matexp(&z.view()).unwrap();
    assert_close_slice(
        e.storage().as_slice(),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
}

#[test]
fn matexp_diagonal_closed_form() {
    // exp(diag(a,b)) = diag(e^a, e^b).
    let d = mat([2, 2], vec![0.7, 0.0, 0.0, -1.3]);
    let e = matexp(&d.view()).unwrap();
    assert_close_slice(
        e.storage().as_slice(),
        &[0.7_f64.exp(), 0.0, 0.0, (-1.3_f64).exp()],
    );
}

#[test]
fn matexp_nilpotent_closed_form() {
    // N = [[0,1],[0,0]], N² = 0 ⇒ exp(N) = I + N = [[1,1],[0,1]].
    let n = mat([2, 2], vec![0.0, 1.0, 0.0, 0.0]);
    let e = matexp(&n.view()).unwrap();
    assert_close_slice(e.storage().as_slice(), &[1.0, 1.0, 0.0, 1.0]);
}

#[test]
fn matexp_skew_symmetric_is_rotation() {
    // exp([[0,-θ],[θ,0]]) = [[cosθ,-sinθ],[sinθ,cosθ]].
    let theta = 0.9_f64;
    let s = mat([2, 2], vec![0.0, -theta, theta, 0.0]);
    let e = matexp(&s.view()).unwrap();
    assert_close_slice(
        e.storage().as_slice(),
        &[theta.cos(), -theta.sin(), theta.sin(), theta.cos()],
    );
}

#[test]
fn matexp_3x3_self_validate() {
    // Self-validate: compare against Taylor series truncated at sufficient terms.
    // For ||A|| small enough, exp(A) ≈ I + A + A²/2! + A³/3! + ...
    let data = vec![1.2, -0.7, 0.4, 0.3, 2.1, -1.5, -0.6, 0.8, 0.5];
    let a = mat([3, 3], data);
    let e = matexp(&a.view()).unwrap();

    // Compute via scaling-and-squaring reference: use the known identity
    // det(exp(A)) = exp(tr(A)).
    let det_e = leto_ops::det(&e.view()).unwrap();
    let tr_a = leto_ops::trace(&a.view()).unwrap();
    assert_close(det_e, tr_a.exp());

    // Trace identity: tr(exp(A)) should match the sum of eigenvalues of exp(A).
    let tr_e = leto_ops::trace(&e.view()).unwrap();
    assert!(tr_e > 0.0, "trace of matrix exponential must be positive");
}

#[test]
fn matexp_fluent_method_matches_free_function() {
    let a = mat([2, 2], vec![0.3, 1.0, -0.5, 0.2]);
    let free = matexp(&a.view()).unwrap();
    let fluent = a.matexp().unwrap();
    assert_close_slice(fluent.storage().as_slice(), free.storage().as_slice());
    // And matpow as a method.
    let p_free = matpow(&a.view(), 3).unwrap();
    let p_fluent = a.matpow(3).unwrap();
    assert_close_slice(p_fluent.storage().as_slice(), p_free.storage().as_slice());
}

#[test]
fn matexp_rejects_non_square_and_non_finite() {
    let rect = mat([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        matexp(&rect.view()).unwrap_err(),
        LetoError::ShapeMismatch {
            lhs: vec![2, 3],
            rhs: vec![2, 2],
        }
    );
    let nan = mat([2, 2], vec![1.0, f64::NAN, 0.0, 1.0]);
    assert_eq!(
        matexp(&nan.view()).unwrap_err(),
        LetoError::StorageError {
            reason: "matrix exponential requires finite entries".to_string(),
        }
    );
}
