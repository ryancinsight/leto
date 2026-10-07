#![allow(clippy::unwrap_used, reason = "test scope")]

use super::CscMatrix;
use crate::application::sparse::{CsrMatrix, CsrView};

#[test]
fn from_dense_round_trips() {
    let dense =
        leto::Array2::from_shape_vec([3, 3], vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0])
            .unwrap();
    let csc = CscMatrix::from_dense(&dense.view());
    assert_eq!(csc.shape(), (3, 3));
    assert_eq!(csc.nnz(), 5);
    let round = csc.to_dense();
    assert_eq!(round, dense);
}

#[test]
fn csc_identity_matrix() {
    let dense =
        leto::Array2::from_shape_vec([3, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
    let csc = CscMatrix::from_dense(&dense.view());
    assert_eq!(csc.shape(), (3, 3));
    assert_eq!(csc.nnz(), 3);
    let (values, row_indices, col_ptr) = csc.as_parts();
    assert_eq!(values, &[1.0, 1.0, 1.0]);
    assert_eq!(row_indices, &[0, 1, 2]);
    assert_eq!(col_ptr, &[0, 1, 2, 3]);
}

#[test]
fn csc_column_access() {
    let dense = leto::Array2::from_shape_vec([3, 2], vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0]).unwrap();
    let csc = CscMatrix::from_dense(&dense.view());
    let col0 = csc.column(0);
    assert_eq!(col0.values(), &[1.0, 3.0]);
    assert_eq!(col0.row_indices(), &[0, 2]);
    let col1 = csc.column(1);
    assert_eq!(col1.values(), &[2.0]);
    assert_eq!(col1.row_indices(), &[1]);
}

#[test]
fn csc_transpose_is_csr_of_transpose() {
    let dense = leto::Array2::from_shape_vec([2, 3], vec![1.0, 0.0, 2.0, 3.0, 4.0, 0.0]).unwrap();
    let csc = CscMatrix::from_dense(&dense.view());
    let csr_via_transpose = csc.transpose();
    let expected_csr = CsrMatrix::from_dense(&dense.view().transpose([1, 0]).unwrap());
    assert_eq!(csr_via_transpose, expected_csr);
}

#[test]
fn csc_from_csr_round_trips() {
    let dense =
        leto::Array2::from_shape_vec([3, 3], vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0])
            .unwrap();
    let csr = CsrMatrix::from_dense(&dense.view());
    let csc = CscMatrix::from_csr(&csr.as_view());
    assert_eq!(csc.to_dense(), dense);
}

#[test]
fn csc_from_csr_accepts_borrowed_view() {
    let dense =
        leto::Array2::from_shape_vec([3, 3], vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0])
            .unwrap();
    let owned = CsrMatrix::from_dense(&dense.view());
    let view = CsrView::from_slices(
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[0, 2, 1, 0, 2],
        &[0, 2, 3, 5],
        3,
        3,
    )
    .unwrap();
    assert_eq!(
        CscMatrix::from_csr(&view),
        CscMatrix::from_csr(&owned.as_view())
    );
}

#[test]
fn csc_zeros() {
    let csc = CscMatrix::<f64>::zeros(3, 4);
    assert_eq!(csc.shape(), (3, 4));
    assert_eq!(csc.nnz(), 0);
}

#[test]
fn csc_diagonal() {
    let dense =
        leto::Array2::from_shape_vec([3, 3], vec![1.0, 0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 5.0])
            .unwrap();
    let csc = CscMatrix::from_dense(&dense.view());
    assert_eq!(csc.diagonal(), vec![1.0, 4.0, 5.0]);
}

#[test]
fn csc_scale_values() {
    let dense = leto::Array2::from_shape_vec([3, 2], vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0]).unwrap();
    let mut csc = CscMatrix::from_dense(&dense.view());
    csc.scale_values(2.0);
    let expected =
        leto::Array2::from_shape_vec([3, 2], vec![2.0, 0.0, 0.0, 4.0, 6.0, 0.0]).unwrap();
    assert_eq!(csc.to_dense(), expected);
}

#[test]
fn csc_frobenius_norm() {
    let dense = leto::Array2::from_shape_vec([2, 2], vec![3.0f64, 0.0, 0.0, 4.0]).unwrap();
    let csc = CscMatrix::from_dense(&dense.view());
    assert!((csc.frobenius_norm() - 5.0f64).abs() < 1e-12);
}

#[test]
fn csc_diagonally_dominant() {
    let dd =
        leto::Array2::from_shape_vec([3, 3], vec![4.0, 1.0, 0.0, 1.0, 5.0, 2.0, 0.0, 2.0, 6.0])
            .unwrap();
    let csc = CscMatrix::from_dense(&dd.view());
    assert!(csc.is_strictly_diagonally_dominant());

    let non_dd = leto::Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 1.0]).unwrap();
    let csc2 = CscMatrix::from_dense(&non_dd.view());
    assert!(!csc2.is_strictly_diagonally_dominant());
}

#[test]
fn from_parts_validates_invariants() {
    assert!(CscMatrix::<f64>::from_parts(
        vec![1.0, 2.0, 3.0],
        vec![0, 1, 2],
        vec![0, 2, 2, 3],
        3,
        3
    )
    .is_ok());
    assert!(CscMatrix::<f64>::from_parts(vec![1.0], vec![0, 1], vec![0, 1], 2, 1).is_err());
    assert!(CscMatrix::<f64>::from_parts(vec![1.0], vec![5], vec![0, 1], 2, 1).is_err());
    assert!(CscMatrix::<f64>::from_parts(vec![1.0], vec![0], vec![1, 1], 2, 1).is_err());
}
