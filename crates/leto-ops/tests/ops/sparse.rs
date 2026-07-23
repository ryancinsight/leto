use leto::{Array, Array2, SliceArg, Storage};
use leto_ops::{spgemm, spmm, spmm_into, spmv, spmv_into, CsrMatrix};

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
fn csr_zero_matrix_has_valid_empty_rows() {
    let csr = CsrMatrix::<f64>::zeros(3, 5);

    assert_eq!(csr.shape(), (3, 5));
    assert_eq!(csr.nrows(), 3);
    assert_eq!(csr.ncols(), 5);
    assert_eq!(csr.nnz(), 0);
    assert_eq!(csr.values(), &[] as &[f64]);
    assert_eq!(csr.col_indices(), &[] as &[usize]);
    assert_eq!(csr.row_ptr(), &[0, 0, 0, 0]);
    for row in 0..csr.nrows() {
        let view = csr.row(row);
        assert_eq!(view.nnz(), 0);
        assert_eq!(view.values(), &[] as &[f64]);
        assert_eq!(view.col_indices(), &[] as &[usize]);
    }
}

#[test]
fn csr_row_views_and_value_mutation_preserve_structure() {
    let mut csr = CsrMatrix::from_parts(
        vec![2.0f64, -1.0, 3.0, 4.0],
        vec![0, 2, 1, 2],
        vec![0, 2, 3, 4],
        3,
        3,
    )
    .unwrap();

    let row0 = csr.row(0);
    assert_eq!(row0.nnz(), 2);
    assert_eq!(row0.values(), &[2.0, -1.0]);
    assert_eq!(row0.col_indices(), &[0, 2]);

    for value in csr.values_mut() {
        *value *= 2.0;
    }

    assert_eq!(csr.values(), &[4.0, -2.0, 6.0, 8.0]);
    assert_eq!(csr.col_indices(), &[0, 2, 1, 2]);
    assert_eq!(csr.row_ptr(), &[0, 2, 3, 4]);
    assert_eq!(csr.row(1).values(), &[6.0]);
    assert_eq!(csr.row(1).col_indices(), &[1]);
}

#[test]
fn csr_diagonal_norm_and_dominance_are_value_semantic() {
    let dominant = CsrMatrix::from_parts(
        vec![4.0f64, -1.0, 5.0, 1.5, -2.0, 6.0],
        vec![0, 2, 1, 0, 1, 2],
        vec![0, 2, 3, 6],
        3,
        3,
    )
    .unwrap();

    assert_eq!(dominant.diagonal(), vec![4.0, 5.0, 6.0]);
    assert_eq!(
        dominant.frobenius_norm(),
        (4.0f64 * 4.0 + 1.0 + 25.0 + 2.25 + 4.0 + 36.0).sqrt()
    );
    assert!(dominant.is_strictly_diagonally_dominant());
    assert_eq!(dominant.condition_estimate().unwrap(), 19.0 / 12.0);

    let non_dominant =
        CsrMatrix::from_parts(vec![1.0f64, 2.0], vec![0, 1], vec![0, 2, 2], 2, 2).unwrap();
    assert!(!non_dominant.is_strictly_diagonally_dominant());

    let rectangular =
        CsrMatrix::from_parts(vec![7.0f64, 8.0], vec![1, 2], vec![0, 1, 2], 2, 4).unwrap();
    assert_eq!(rectangular.diagonal(), vec![0.0, 0.0]);
    assert!(!rectangular.is_strictly_diagonally_dominant());
    assert!(rectangular.condition_estimate().is_err());

    let singular_diagonal =
        CsrMatrix::from_parts(vec![1.0e-13f64], vec![0], vec![0, 1], 1, 1).unwrap();
    assert_eq!(
        singular_diagonal.condition_estimate().unwrap(),
        f64::INFINITY
    );
}

#[test]
fn csr_value_row_and_column_scaling_preserve_structure() {
    let mut csr = CsrMatrix::from_parts(
        vec![2.0f64, -1.0, 3.0, 4.0],
        vec![0, 2, 1, 2],
        vec![0, 2, 3, 4],
        3,
        3,
    )
    .unwrap();

    csr.scale_values(0.5);
    assert_eq!(csr.values(), &[1.0, -0.5, 1.5, 2.0]);
    assert_eq!(csr.col_indices(), &[0, 2, 1, 2]);
    assert_eq!(csr.row_ptr(), &[0, 2, 3, 4]);

    csr.scale_rows(&[2.0, 3.0, 4.0]).unwrap();
    assert_eq!(csr.values(), &[2.0, -1.0, 4.5, 8.0]);

    csr.scale_columns(&[10.0, 20.0, 30.0]).unwrap();
    assert_eq!(csr.values(), &[20.0, -30.0, 90.0, 240.0]);
    assert_eq!(
        csr.to_dense().storage().as_slice(),
        &[20.0, 0.0, -30.0, 0.0, 90.0, 0.0, 0.0, 0.0, 240.0]
    );

    assert!(csr.scale_rows(&[1.0, 2.0]).is_err());
    assert!(csr.scale_columns(&[1.0, 2.0]).is_err());
}

#[test]
fn csr_transpose_matches_dense_transpose_and_preserves_sorted_rows() {
    let csr = CsrMatrix::from_parts(
        vec![1.0f64, 2.0, 3.0, 4.0, 5.0],
        vec![0, 3, 1, 0, 2],
        vec![0, 2, 3, 5],
        3,
        4,
    )
    .unwrap();

    let transposed = csr.transpose();

    assert_eq!(transposed.shape(), (4, 3));
    assert_eq!(transposed.row_ptr(), &[0, 2, 3, 4, 5]);
    assert_eq!(transposed.col_indices(), &[0, 2, 1, 2, 0]);
    assert_eq!(transposed.values(), &[1.0, 4.0, 3.0, 5.0, 2.0]);
    assert_eq!(
        transposed.to_dense().storage().as_slice(),
        &[1.0, 0.0, 4.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0, 2.0, 0.0, 0.0]
    );
    assert_eq!(transposed.transpose(), csr);
}

#[test]
fn csr_zero_transpose_swaps_shape_without_storage() {
    let csr = CsrMatrix::<f64>::zeros(2, 5);
    let transposed = csr.transpose();

    assert_eq!(transposed.shape(), (5, 2));
    assert_eq!(transposed.nnz(), 0);
    assert_eq!(transposed.values(), &[] as &[f64]);
    assert_eq!(transposed.col_indices(), &[] as &[usize]);
    assert_eq!(transposed.row_ptr(), &[0, 0, 0, 0, 0, 0]);
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
    assert_eq!(csr.to_dense().storage().as_slice(), &[] as &[f64]);
}
