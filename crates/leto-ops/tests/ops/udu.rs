//! Tests for symmetric indefinite unpivoted `A = U D Uᵀ`.

use leto::{Array, Array2, Storage};
use leto_ops::{udu_decompose, MatrixDecompose, MatrixProduct};
use nalgebra::{DMatrix, DVector};

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

    let na_det = DMatrix::from_row_slice(n, n, &values).determinant();
    assert_close(f.det(), na_det);
}

#[test]
fn udu_solve_and_inverse_match_nalgebra_oracles() {
    let n = 3;
    let values = vec![4.0, 2.0, -2.0, 2.0, -3.0, 1.0, -2.0, 1.0, 2.0];
    let rhs_values = vec![3.0, -1.0, 2.0];
    let a = Array2::from_shape_vec([n, n], values.clone()).unwrap();
    let rhs = Array::from_shape_vec([n], rhs_values.clone()).unwrap();
    let f = a.udu().unwrap();

    let x = f.solve(&rhs.view()).unwrap();
    let na = DMatrix::from_row_slice(n, n, &values);
    let na_x = na
        .clone()
        .lu()
        .solve(&DVector::from_vec(rhs_values))
        .unwrap();
    assert_close_slice(x.storage().as_slice(), na_x.as_slice());

    let inv = f.inv().unwrap();
    let na_inv = na.try_inverse().unwrap();
    let mut expected = Vec::with_capacity(n * n);
    for r in 0..n {
        for c in 0..n {
            expected.push(na_inv[(r, c)]);
        }
    }
    assert_close_slice(inv.storage().as_slice(), &expected);
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
