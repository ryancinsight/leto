//! Tests for symmetric indefinite unpivoted `A = U D Uᵀ`.

use leto::{Array, Array2, Storage};
use leto_ops::{udu_decompose, MatrixDecompose, MatrixProduct};

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

fn diagonal_matrix(values: &[f64]) -> Array2<f64> {
    let n = values.len();
    let mut out = vec![0.0; n * n];
    for (i, value) in values.iter().enumerate() {
        out[i * n + i] = *value;
    }
    Array2::from_shape_vec([n, n], out).unwrap()
}

#[test]
fn udu_reconstructs_symmetric_indefinite_matrix() {
    let n = 3;
    let values = vec![4.0, 2.0, -2.0, 2.0, -3.0, 1.0, -2.0, 1.0, 2.0];
    let a = Array2::from_shape_vec([n, n], values.clone()).unwrap();
    let f = udu_decompose(&a.view()).unwrap();

    let u = f.u();
    let d = diagonal_matrix(f.diagonal());
    let u_transposed = u.transpose([1, 0]).unwrap();
    let reconstructed = u.matmul(&d).unwrap().matmul(&u_transposed).unwrap();
    assert_close_slice(reconstructed.storage().as_slice(), &values);

    // Self-validate det via product of diagonal entries (U is unit-triangular, det(U)=1).
    let det_from_diag: f64 = f.diagonal().iter().product();
    assert_close(f.det(), det_from_diag);
}

#[test]
fn udu_solve_and_inverse_self_consistent() {
    let n = 3;
    let values = vec![4.0, 2.0, -2.0, 2.0, -3.0, 1.0, -2.0, 1.0, 2.0];
    let rhs_values = vec![3.0, -1.0, 2.0];
    let a = Array2::from_shape_vec([n, n], values.clone()).unwrap();
    let rhs = Array::from_shape_vec([n], rhs_values.clone()).unwrap();
    let f = a.udu().unwrap();

    let x = f.solve(&rhs.view()).unwrap();

    // Self-validate: A · x = b (element-wise since x is rank-1).
    for (i, &rhs_value) in rhs_values.iter().enumerate().take(n) {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a.get([i, j]).unwrap() * x.get([j]).unwrap();
        }
        assert_close(sum, rhs_value);
    }

    let inv = f.inv().unwrap();

    // Self-validate: A · A⁻¹ = I.
    let product = a.matmul(&inv).unwrap();
    for r in 0..n {
        for c in 0..n {
            let expected = if r == c { 1.0 } else { 0.0 };
            assert_close(*product.get([r, c]).unwrap(), expected);
        }
    }

    // Self-validate: A⁻¹ · A = I.
    let product_rev = inv.matmul(&a).unwrap();
    for r in 0..n {
        for c in 0..n {
            let expected = if r == c { 1.0 } else { 0.0 };
            assert_close(*product_rev.get([r, c]).unwrap(), expected);
        }
    }
}

#[test]
fn udu_rejects_invalid_contracts() {
    let non_square = Array2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 2.0, 4.0, 5.0]).unwrap();
    assert!(udu_decompose(&non_square.view()).is_err());

    let nonsymmetric = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(udu_decompose(&nonsymmetric.view()).is_err());

    let zero_pivot = Array2::from_shape_vec([2, 2], vec![1.0, 1.0, 1.0, 0.0]).unwrap();
    assert!(udu_decompose(&zero_pivot.view()).is_err());
}
