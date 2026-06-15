//! Differential tests for the fluent rank-2 LA trait layer (ADR 0003).
//!
//! Each test asserts the trait-method surface is identical to (a) the
//! authoritative free-function kernel it delegates to and (b) the nalgebra /
//! ndarray oracle. A transposed-receiver case proves arbitrary-layout support
//! flows through the `AsMatrixView` bridge unchanged.

use leto::{Array, Array2, Storage};
use leto_ops::{det, matmul, norm_l2, MatrixDecompose, MatrixNorm, MatrixProduct, MatrixSolve};
use nalgebra::{Cholesky, DMatrix, DVector, SymmetricEigen};
use ndarray::Array2 as NdArray2;

const EPS: f64 = 1.0e-9;

#[track_caller]
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= EPS * expected.abs().max(1.0),
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

fn dmatrix(rows: usize, cols: usize, values: &[f64]) -> DMatrix<f64> {
    DMatrix::from_row_slice(rows, cols, values)
}

/// Flatten a nalgebra (column-major) matrix to row-major for slice comparison.
fn dmatrix_row_major(m: &DMatrix<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.nrows() * m.ncols());
    for r in 0..m.nrows() {
        for c in 0..m.ncols() {
            out.push(m[(r, c)]);
        }
    }
    out
}

#[test]
fn pinv_method_matches_nalgebra_and_moore_penrose() {
    // Tall (full column rank), wide (full row rank), and square invertible.
    let cases: [(usize, usize, Vec<f64>); 3] = [
        (4, 2, vec![1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0]),
        (2, 4, vec![1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0]),
        (3, 3, vec![4.0, -2.0, 1.0, -2.0, 4.0, -2.0, 1.0, -2.0, 4.0]),
    ];

    for (rows, cols, values) in cases {
        let a = Array2::from_shape_vec([rows, cols], values.clone()).unwrap();
        let leto_pinv = a.pinv().unwrap();
        assert_eq!(leto_pinv.shape(), [cols, rows]);

        // Differential vs nalgebra's SVD-based pseudo_inverse.
        let na = dmatrix(rows, cols, &values);
        let na_pinv = na.clone().pseudo_inverse(1.0e-12).unwrap();
        assert_close_slice(leto_pinv.storage().as_slice(), &dmatrix_row_major(&na_pinv));

        // Moore-Penrose condition 1: A A⁺ A == A (oracle-independent).
        let mut a_pinv = Array2::zeros([rows, cols]);
        {
            let mut tmp = Array2::zeros([rows, rows]);
            matmul(&a.view(), &leto_pinv.view(), &mut tmp.view_mut()).unwrap();
            matmul(&tmp.view(), &a.view(), &mut a_pinv.view_mut()).unwrap();
        }
        assert_close_slice(a_pinv.storage().as_slice(), &values);
    }
}

#[test]
fn matmul_method_matches_kernel_and_ndarray() {
    let (m, k, n) = (4usize, 3, 5);
    let a_vals: Vec<f64> = (0..m * k).map(|i| i as f64 * 0.3 + 1.0).collect();
    let b_vals: Vec<f64> = (0..k * n).map(|i| i as f64 * 0.2 - 1.0).collect();
    let a = Array2::from_shape_vec([m, k], a_vals.clone()).unwrap();
    let b = Array2::from_shape_vec([k, n], b_vals.clone()).unwrap();

    // Fluent method.
    let fluent = a.matmul(&b).unwrap();

    // Authoritative caller-owned kernel.
    let mut kernel_out = Array2::from_shape_vec([m, n], vec![0.0; m * n]).unwrap();
    matmul(&a.view(), &b.view(), &mut kernel_out.view_mut()).unwrap();

    // ndarray oracle.
    let nd = NdArray2::from_shape_vec((m, k), a_vals)
        .unwrap()
        .dot(&NdArray2::from_shape_vec((k, n), b_vals).unwrap());

    assert_close_slice(fluent.storage().as_slice(), kernel_out.storage().as_slice());
    assert_close_slice(fluent.storage().as_slice(), nd.as_slice().unwrap());
}

#[test]
fn solve_det_inv_methods_match_nalgebra() {
    let values = vec![4.0, -2.0, 1.0, -2.0, 4.0, -2.0, 1.0, -2.0, 4.0];
    let rhs_values = vec![11.0, -16.0, 17.0];
    let a = Array2::from_shape_vec([3, 3], values.clone()).unwrap();
    let rhs = Array::from_shape_vec([3], rhs_values.clone()).unwrap();

    let x = a.solve(&rhs.view()).unwrap();
    let d = a.det().unwrap();
    let inverse = a.inv().unwrap();

    let na = dmatrix(3, 3, &values);
    let na_rhs = DVector::from_vec(rhs_values);
    let na_lu = na.clone().lu();
    let expected_x = na_lu.solve(&na_rhs).unwrap();
    let expected_det = na_lu.determinant();
    let expected_inv = na.try_inverse().unwrap();

    assert_close_slice(x.storage().as_slice(), expected_x.as_slice());
    assert_close(d, expected_det);
    let mut expected_inv_rows = Vec::with_capacity(9);
    for r in 0..3 {
        for c in 0..3 {
            expected_inv_rows.push(expected_inv[(r, c)]);
        }
    }
    assert_close_slice(inverse.storage().as_slice(), &expected_inv_rows);
}

#[test]
fn cholesky_and_eigen_methods_match_nalgebra() {
    let values = vec![6.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
    let a = Array2::from_shape_vec([3, 3], values.clone()).unwrap();
    let na = dmatrix(3, 3, &values);

    let mut eig = a.symmetric_eigenvalues().unwrap();
    let mut expected_eig = SymmetricEigen::new(na.clone())
        .eigenvalues
        .as_slice()
        .to_vec();
    eig.sort_by(|x, y| x.total_cmp(y));
    expected_eig.sort_by(|x, y| x.total_cmp(y));
    assert_close_slice(&eig, &expected_eig);

    let chol = a.cholesky().unwrap();
    let lower = Cholesky::new(na).unwrap().l();
    let mut expected_lower = Vec::with_capacity(9);
    for r in 0..3 {
        for c in 0..3 {
            expected_lower.push(lower[(r, c)]);
        }
    }
    assert_close_slice(chol.lower().storage().as_slice(), &expected_lower);
}

#[test]
fn singular_values_and_least_squares_methods_match_nalgebra() {
    // Singular values of a tall matrix.
    let sv_vals = vec![1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0];
    let a = Array2::from_shape_vec([4, 2], sv_vals.clone()).unwrap();
    let mut sv = a.singular_values().unwrap();
    let mut expected_sv = dmatrix(4, 2, &sv_vals)
        .svd(false, false)
        .singular_values
        .as_slice()
        .to_vec();
    sv.sort_by(|x, y| y.total_cmp(x));
    expected_sv.sort_by(|x, y| y.total_cmp(x));
    assert_close_slice(&sv, &expected_sv);

    // Overdetermined least squares vs normal equations.
    let ls_vals = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
    let rhs_vals = vec![6.0, 5.0, 7.0, 10.0];
    let ls = Array2::from_shape_vec([4, 2], ls_vals.clone()).unwrap();
    let rhs = Array::from_shape_vec([4], rhs_vals.clone()).unwrap();
    let x = ls.solve_least_squares(&rhs.view()).unwrap();

    let na = dmatrix(4, 2, &ls_vals);
    let na_rhs = DVector::from_vec(rhs_vals);
    let expected = (na.transpose() * &na)
        .lu()
        .solve(&(na.transpose() * &na_rhs))
        .unwrap();
    assert_close_slice(x.storage().as_slice(), expected.as_slice());
}

#[test]
fn norm_methods_match_kernel_and_nalgebra() {
    let values = vec![3.0, -4.0, 12.0, 0.0, -5.0, 0.0];
    let a = Array2::from_shape_vec([2, 3], values.clone()).unwrap();

    assert_close(a.norm_l2().unwrap(), norm_l2(&a.view()).unwrap());
    // Frobenius norm parity with nalgebra.
    assert_close(a.norm_l2().unwrap(), dmatrix(2, 3, &values).norm());
    // L1 / max entrywise references.
    assert_close(a.norm_l1().unwrap(), values.iter().map(|v| v.abs()).sum());
    assert_close(
        a.norm_max().unwrap(),
        values.iter().map(|v| v.abs()).fold(0.0, f64::max),
    );
}

#[test]
fn strided_transposed_receiver_matches_contiguous() {
    // A transposed view is strided (non-C-contiguous); the trait must produce
    // the same result as the same op on its contiguous materialization.
    let values = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0];
    let a = Array2::from_shape_vec([3, 3], values).unwrap();
    let transposed = a.transpose([1, 0]).unwrap();
    let materialized = transposed.to_contiguous();

    // det through the strided view == det of the dense copy (free fn).
    assert_close(
        transposed.det().unwrap(),
        det(&materialized.view()).unwrap(),
    );
    // norm through the strided view == norm of the dense copy.
    assert_close(
        transposed.norm_l2().unwrap(),
        materialized.norm_l2().unwrap(),
    );
}
