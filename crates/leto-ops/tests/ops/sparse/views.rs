//! Zero-copy CSR/CSC views over caller slices.

use super::*;

#[test]
fn csr_view_from_slices_validates_like_from_parts() {
    let values = vec![2.0f64, -1.0, 3.0, 4.0];
    let cols = vec![0usize, 2, 1, 2];
    let row_ptr = vec![0usize, 2, 3, 4];

    // Each broken triple must fail identically through both constructors
    // (shared validator SSOT).
    let broken: Vec<(Vec<f64>, Vec<usize>, Vec<usize>)> = vec![
        (values.clone(), cols.clone(), vec![0, 2, 3]),
        (values.clone(), cols.clone(), vec![1, 2, 3, 4]),
        (values.clone(), cols.clone(), vec![0, 2, 3, 3]),
        (values.clone(), cols.clone(), vec![0, 3, 2, 4]),
        (values.clone(), vec![0, 3, 1, 2], row_ptr.clone()),
        (values.clone(), vec![0, 0, 1, 2], row_ptr.clone()),
        (values.clone(), vec![0, 2, 1], row_ptr.clone()),
    ];
    for (v, c, r) in &broken {
        assert!(CsrView::from_slices(v, c, r, 3, 3).is_err());
        assert!(CsrMatrix::from_parts(v.clone(), c.clone(), r.clone(), 3, 3).is_err());
    }
    // Control: the valid triple passes through both.
    let view = CsrView::from_slices(&values, &cols, &row_ptr, 3, 3).unwrap();
    assert_eq!(view.shape(), (3, 3));
    assert_eq!(view.nnz(), 4);
    assert!(CsrMatrix::from_parts(values, cols, row_ptr, 3, 3).is_ok());
}

#[test]
fn csr_view_aliases_inputs_zero_copy() {
    let values = vec![2.0f64, -1.0, 3.0, 4.0];
    let cols = vec![0usize, 2, 1, 2];
    let row_ptr = vec![0usize, 2, 3, 4];
    let view = CsrView::from_slices(&values, &cols, &row_ptr, 3, 3).unwrap();
    let (v, c, r) = view.as_parts();
    assert_eq!(v.as_ptr(), values.as_ptr());
    assert_eq!(c.as_ptr(), cols.as_ptr());
    assert_eq!(r.as_ptr(), row_ptr.as_ptr());

    let owned = CsrMatrix::from_parts(values, cols, row_ptr, 3, 3).unwrap();
    let owned_view = owned.as_view();
    let (ov, oc, orp) = owned_view.as_parts();
    let (ev, ec, er) = owned.as_parts();
    assert_eq!(ov.as_ptr(), ev.as_ptr());
    assert_eq!(oc.as_ptr(), ec.as_ptr());
    assert_eq!(orp.as_ptr(), er.as_ptr());
}

#[test]
fn spmv_view_matches_owned() {
    let values = vec![2.0f64, -1.0, 3.0, 4.0];
    let cols = vec![0usize, 2, 1, 2];
    let row_ptr = vec![0usize, 2, 3, 4];
    let owned = CsrMatrix::from_parts(values.clone(), cols.clone(), row_ptr.clone(), 3, 3).unwrap();
    let view = CsrView::from_slices(&values, &cols, &row_ptr, 3, 3).unwrap();
    let x = Array::from_shape_vec([3], vec![1.0f64, 2.0, 3.0]).unwrap();
    let mut y_owned = vec![0.0; 3];
    let mut y_view = vec![0.0; 3];
    spmv_into(&owned, &x.view(), &mut y_owned).unwrap();
    spmv_view_into(&view, &x.view(), &mut y_view).unwrap();
    assert_eq!(y_owned, y_view);
    assert_eq!(y_owned, vec![-1.0, 6.0, 12.0]);
}

#[test]
fn spmm_view_matches_owned() {
    let values = vec![2.0f64, -1.0, 3.0, 4.0];
    let cols = vec![0usize, 2, 1, 2];
    let row_ptr = vec![0usize, 2, 3, 4];
    let owned = CsrMatrix::from_parts(values.clone(), cols.clone(), row_ptr.clone(), 3, 3).unwrap();
    let view = CsrView::from_slices(&values, &cols, &row_ptr, 3, 3).unwrap();
    let b = Array2::from_shape_vec([3, 2], vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mut c_owned = vec![0.0; 6];
    let mut c_view = vec![0.0; 6];
    spmm_into(&owned, &b.view(), &mut c_owned).unwrap();
    spmm_view_into(&view, &b.view(), &mut c_view).unwrap();
    assert_eq!(c_owned, c_view);
    assert_eq!(c_owned, vec![-3.0, -2.0, 9.0, 12.0, 20.0, 24.0]);
}

#[test]
fn csc_view_from_slices_validates_like_from_parts() {
    let values = vec![1.0f64, 3.0, 2.0];
    let rows = vec![0usize, 2, 1];
    let col_ptr = vec![0usize, 2, 3];

    // Each broken triple must fail identically through both constructors
    // (shared validator SSOT).
    let broken: Vec<(Vec<f64>, Vec<usize>, Vec<usize>)> = vec![
        (values.clone(), rows.clone(), vec![0, 2]),
        (values.clone(), rows.clone(), vec![1, 2, 3]),
        (values.clone(), rows.clone(), vec![0, 2, 2]),
        (values.clone(), rows.clone(), vec![0, 4, 3]),
        (values.clone(), vec![0, 3, 1], col_ptr.clone()),
        (values.clone(), vec![0, 0, 1], col_ptr.clone()),
        (values.clone(), vec![0, 2], col_ptr.clone()),
    ];
    for (v, r, c) in &broken {
        assert!(CscView::from_slices(v, r, c, 3, 2).is_err());
        assert!(CscMatrix::from_parts(v.clone(), r.clone(), c.clone(), 3, 2).is_err());
    }
    // Control: the valid triple passes through both.
    let view = CscView::from_slices(&values, &rows, &col_ptr, 3, 2).unwrap();
    assert_eq!(view.shape(), (3, 2));
    assert_eq!(view.nnz(), 3);
    assert!(CscMatrix::from_parts(values, rows, col_ptr, 3, 2).is_ok());
}

#[test]
fn csc_view_aliases_inputs_zero_copy() {
    let values = vec![1.0f64, 3.0, 2.0];
    let rows = vec![0usize, 2, 1];
    let col_ptr = vec![0usize, 2, 3];
    let view = CscView::from_slices(&values, &rows, &col_ptr, 3, 2).unwrap();
    let (v, r, c) = view.as_parts();
    assert_eq!(v.as_ptr(), values.as_ptr());
    assert_eq!(r.as_ptr(), rows.as_ptr());
    assert_eq!(c.as_ptr(), col_ptr.as_ptr());

    let owned = CscMatrix::from_parts(values, rows, col_ptr, 3, 2).unwrap();
    let owned_view = owned.as_view();
    let (ov, orw, ocp) = owned_view.as_parts();
    let (ev, er, ec) = owned.as_parts();
    assert_eq!(ov.as_ptr(), ev.as_ptr());
    assert_eq!(orw.as_ptr(), er.as_ptr());
    assert_eq!(ocp.as_ptr(), ec.as_ptr());
}

#[test]
fn csc_spmv_view_matches_owned() {
    let values = vec![1.0f64, 3.0, 2.0];
    let rows = vec![0usize, 2, 1];
    let col_ptr = vec![0usize, 2, 3];
    let owned = CscMatrix::from_parts(values.clone(), rows.clone(), col_ptr.clone(), 3, 2).unwrap();
    let view = CscView::from_slices(&values, &rows, &col_ptr, 3, 2).unwrap();
    let x = Array::from_shape_vec([2], vec![2.0f64, 5.0]).unwrap();
    let mut y_owned = vec![0.0; 3];
    let mut y_view = vec![0.0; 3];
    csc_spmv_into(&owned, &x.view(), &mut y_owned).unwrap();
    csc_spmv_view_into(&view, &x.view(), &mut y_view).unwrap();
    assert_eq!(y_owned, y_view);
    assert_eq!(y_owned, vec![2.0, 10.0, 6.0]);
}
