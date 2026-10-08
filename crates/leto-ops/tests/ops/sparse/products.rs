#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array, Array2, SliceArg, Storage};
use leto_ops::{spgemm, spmm, spmm_into, spmv, spmv_into, CsrMatrix};

#[test]
fn spmv_matches_closed_form_and_overwrites_output() {
    let a = CsrMatrix::from_parts(
        vec![2.0f64, -1.0, 3.0, 4.0],
        vec![0, 2, 1, 2],
        vec![0, 2, 3, 4],
        3,
        3,
    )
    .unwrap();
    let x_base = Array::from_shape_vec([4], vec![9.0f64, 1.0, 2.0, 3.0]).unwrap();
    let x = x_base
        .slice_with::<1>(&[SliceArg::range(Some(3), Some(0), -1)])
        .unwrap();
    let mut y = vec![99.0; 3];

    spmv_into(&a, &x, &mut y).unwrap();
    let allocated = spmv(&a, &x).unwrap();

    assert_eq!(y, &[5.0, 6.0, 4.0]);
    assert_eq!(allocated.storage().as_slice(), &[5.0, 6.0, 4.0]);
}

#[test]
fn spmm_matches_closed_form_with_strided_dense_rhs() {
    let a =
        CsrMatrix::from_parts(vec![2.0f64, -1.0, 3.0], vec![0, 2, 1], vec![0, 2, 3], 2, 3).unwrap();
    let b_base = Array2::from_shape_vec([3, 2], vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = b_base
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();
    let mut c = vec![99.0; 4];

    spmm_into(&a, &b, &mut c).unwrap();
    let allocated = spmm(&a, &b).unwrap();

    assert_eq!(c, &[-2.0, -3.0, 12.0, 9.0]);
    assert_eq!(allocated.storage().as_slice(), &[-2.0, -3.0, 12.0, 9.0]);
}

#[test]
fn spgemm_matches_closed_form_and_sorted_csr_rows() {
    let a =
        CsrMatrix::from_parts(vec![2.0f64, -1.0, 3.0], vec![0, 2, 1], vec![0, 2, 3], 2, 3).unwrap();
    let b = CsrMatrix::from_parts(
        vec![4.0f64, 5.0, 6.0, 7.0],
        vec![1, 0, 0, 1],
        vec![0, 1, 2, 4],
        3,
        2,
    )
    .unwrap();

    let c = spgemm(&a, &b).unwrap();

    assert_eq!(c.shape(), (2, 2));
    assert_eq!(c.row_ptr(), &[0, 2, 3]);
    assert_eq!(c.col_indices(), &[0, 1, 0]);
    assert_eq!(c.values(), &[-6.0, 1.0, 15.0]);
    assert_eq!(c.to_dense().storage().as_slice(), &[-6.0, 1.0, 15.0, 0.0]);
}

#[test]
fn spgemm_drops_exact_zero_cancellation() {
    let a = CsrMatrix::from_parts(vec![1.0f64, 1.0], vec![0, 1], vec![0, 2], 1, 2).unwrap();
    let b =
        CsrMatrix::from_parts(vec![3.0f64, -3.0, 2.0], vec![0, 0, 1], vec![0, 1, 3], 2, 2).unwrap();

    let c = spgemm(&a, &b).unwrap();

    assert_eq!(c.shape(), (1, 2));
    assert_eq!(c.row_ptr(), &[0, 1]);
    assert_eq!(c.col_indices(), &[1]);
    assert_eq!(c.values(), &[2.0]);
}

#[test]
fn sparse_products_reject_shape_mismatch() {
    let a = CsrMatrix::from_parts(vec![1.0f64], vec![0], vec![0, 1], 1, 1).unwrap();
    let bad_x = Array::from_shape_vec([2], vec![1.0f64, 2.0]).unwrap();
    let bad_b = Array2::from_shape_vec([2, 1], vec![1.0f64, 2.0]).unwrap();
    let good_b = Array2::from_shape_vec([1, 1], vec![1.0f64]).unwrap();

    assert!(spmv(&a, &bad_x.view()).is_err());
    assert!(spmm(&a, &bad_b.view()).is_err());
    assert!(spmm_into(&a, &good_b.view(), &mut [0.0; 2]).is_err());
    assert!(spgemm(&a, &CsrMatrix::zeros(2, 1)).is_err());
}
