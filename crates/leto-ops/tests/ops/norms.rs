use leto::{Array, Layout, SliceArg, VecStorage};
use leto_ops::{norm_l1, norm_l2, norm_max};

const EPS: f64 = 1e-12;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= EPS * expected.abs().max(1.0),
        "actual {actual} expected {expected}"
    );
}

#[test]
fn vector_norms_match_nalgebra() {
    let values = vec![3.0f64, -4.0, 12.0, -0.5, 2.25];
    let array = Array::from_shape_vec([5], values.clone()).unwrap();
    let reference = nalgebra::DVector::from_vec(values);

    assert_close(norm_l2(&array.view()).unwrap(), reference.norm());
    assert_close(norm_l1(&array.view()).unwrap(), reference.lp_norm(1));
    assert_close(
        norm_max(&array.view()).unwrap(),
        reference.amax(), // max absolute element
    );
}

#[test]
fn frobenius_norm_matches_nalgebra_rank2() {
    let values = vec![1.0f64, -2.0, 3.5, 4.25, -5.5, 6.75];
    let array = Array::from_shape_vec([2, 3], values.clone()).unwrap();
    let reference = nalgebra::DMatrix::from_row_slice(2, 3, &values);

    // norm_l2 over rank-2 is the Frobenius norm; one generic entry point.
    assert_close(norm_l2(&array.view()).unwrap(), reference.norm());
}

#[test]
fn norms_are_layout_independent_on_strided_views() {
    // A transposed view must produce the same norms as its source: the
    // elementwise norms are traversal-order independent and the strided
    // fallback visits each logical element exactly once.
    let values = vec![1.0f64, -2.0, 3.0, -4.0, 5.0, -6.0];
    let array = Array::from_shape_vec([2, 3], values).unwrap();
    let transposed = array.transpose([1, 0]).unwrap();

    assert_close(
        norm_l2(&transposed).unwrap(),
        norm_l2(&array.view()).unwrap(),
    );
    assert_close(
        norm_l1(&transposed).unwrap(),
        norm_l1(&array.view()).unwrap(),
    );
    assert_close(
        norm_max(&transposed).unwrap(),
        norm_max(&array.view()).unwrap(),
    );

    // Every-other-column strided slice: norms over the logical selection only.
    let strided = array
        .view()
        .slice_with::<2>(&[leto::SliceArg::All, leto::SliceArg::range(Some(0), None, 2)])
        .unwrap();
    // columns 0 and 2: values 1, 3, -4, -6 -> L1 = 14, max = 6
    assert_close(norm_l1(&strided).unwrap(), 14.0);
    assert_close(norm_max(&strided).unwrap(), 6.0);
    assert_close(
        norm_l2(&strided).unwrap(),
        (1.0f64 + 9.0 + 16.0 + 36.0).sqrt(),
    );

    let reversed = array
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();
    assert_close(norm_l2(&reversed).unwrap(), norm_l2(&array.view()).unwrap());
    assert_close(norm_l1(&reversed).unwrap(), norm_l1(&array.view()).unwrap());
    assert_close(
        norm_max(&reversed).unwrap(),
        norm_max(&array.view()).unwrap(),
    );
}

#[test]
fn empty_view_norms_are_zero() {
    let array: Array<f64, VecStorage<f64>, 1> =
        Array::new(Layout::c_contiguous([0]).unwrap(), VecStorage::new(vec![])).unwrap();
    assert_eq!(norm_l1(&array.view()).unwrap(), 0.0);
    assert_eq!(norm_l2(&array.view()).unwrap(), 0.0);
    assert_eq!(norm_max(&array.view()).unwrap(), 0.0);
}

#[test]
fn norms_run_at_reduced_precision() {
    use half::f16;
    let values: Vec<f16> = [3.0f32, 4.0].iter().map(|&v| f16::from_f32(v)).collect();
    let array = Array::from_shape_vec([2], values).unwrap();
    // 3-4-5 triangle is exactly representable in f16.
    assert_eq!(norm_l2(&array.view()).unwrap(), f16::from_f32(5.0));
    assert_eq!(norm_l1(&array.view()).unwrap(), f16::from_f32(7.0));
    assert_eq!(norm_max(&array.view()).unwrap(), f16::from_f32(4.0));
}
