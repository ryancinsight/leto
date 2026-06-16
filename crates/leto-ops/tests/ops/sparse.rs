use leto::{Array, Array2, SliceArg, Storage};
use leto_ops::{spmm, spmm_into, spmv, spmv_into, CsrMatrix};

#[test]
fn csr_from_dense_round_trips_nonzeros() {
    let dense = Array2::from_shape_vec(
        [3, 4],
        vec![
            1.0f64, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0, 0.0,
        ],
    )
    .unwrap();

    let csr = CsrMatrix::from_dense(&dense.view());

    assert_eq!(csr.shape(), (3, 4));
    assert_eq!(csr.nnz(), 5);
    assert_eq!(csr.as_parts().0, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(csr.as_parts().1, &[0, 3, 1, 0, 2]);
    assert_eq!(csr.as_parts().2, &[0, 2, 3, 5]);
    assert_eq!(
        csr.to_dense().storage().as_slice(),
        dense.storage().as_slice()
    );
    assert_eq!(csr.density(), 5.0 / 12.0);
}

#[test]
fn csr_from_dense_handles_negative_stride_view() {
    let base =
        Array2::from_shape_vec([2, 4], vec![0.0f64, 1.0, 0.0, 2.0, 3.0, 0.0, 4.0, 0.0]).unwrap();
    let reversed = base
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();

    let csr = CsrMatrix::from_dense(&reversed);

    assert_eq!(csr.shape(), (2, 4));
    assert_eq!(csr.as_parts().0, &[2.0, 1.0, 4.0, 3.0]);
    assert_eq!(csr.as_parts().1, &[0, 2, 1, 3]);
    assert_eq!(
        csr.to_dense().storage().as_slice(),
        &[2.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 3.0]
    );
}

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
fn sparse_products_reject_shape_mismatch() {
    let a = CsrMatrix::from_parts(vec![1.0f64], vec![0], vec![0, 1], 1, 1).unwrap();
    let bad_x = Array::from_shape_vec([2], vec![1.0f64, 2.0]).unwrap();
    let bad_b = Array2::from_shape_vec([2, 1], vec![1.0f64, 2.0]).unwrap();
    let good_b = Array2::from_shape_vec([1, 1], vec![1.0f64]).unwrap();

    assert!(spmv(&a, &bad_x.view()).is_err());
    assert!(spmm(&a, &bad_b.view()).is_err());
    assert!(spmm_into(&a, &good_b.view(), &mut [0.0; 2]).is_err());
}

#[test]
fn csr_from_parts_rejects_invalid_structure() {
    assert!(CsrMatrix::from_parts(vec![1.0f64], vec![0], vec![0], 1, 1).is_err());
    assert!(CsrMatrix::from_parts(vec![1.0f64], vec![], vec![0, 1], 1, 1).is_err());
    assert!(CsrMatrix::from_parts(vec![1.0f64], vec![1], vec![0, 1], 1, 1).is_err());
    assert!(CsrMatrix::from_parts(vec![1.0f64], vec![0], vec![1, 1], 1, 1).is_err());
    assert!(CsrMatrix::from_parts(vec![1.0f64, 2.0], vec![0, 0], vec![0, 2], 1, 1).is_err());
    assert!(CsrMatrix::from_parts(vec![1.0f64, 2.0], vec![1, 0], vec![0, 2], 1, 2).is_err());
}

#[test]
fn empty_width_dense_matrix_has_empty_csr_storage() {
    let dense = Array2::<f64>::from_shape_vec([2, 0], vec![]).unwrap();

    let csr = CsrMatrix::from_dense(&dense.view());

    assert_eq!(csr.shape(), (2, 0));
    assert_eq!(csr.nnz(), 0);
    assert_eq!(csr.density(), 0.0);
    assert_eq!(csr.as_parts().2, &[0, 0, 0]);
    assert_eq!(csr.to_dense().storage().as_slice(), &[]);
}
