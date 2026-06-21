use leto::{Array, Layout, Storage, VecStorage};
use leto_ops::{
    dot, map_inplace, scalar_map, scalar_map_into, unary_map, unary_map_into, AbsOp, AddOp, ExpOp,
    MulOp, NegOp, PowfOp, SqrtOp, l2_normalize_into, jaccard_distance, hamming_distance,
};

const EPS: f64 = 1e-12;

fn assert_close_slice(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() <= EPS, "actual {a} expected {e}");
    }
}

#[test]
fn test_unary_map_allocating_sqrt() {
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let array = Array::new(layout, VecStorage::new(vec![1.0f64, 4.0, 9.0, 16.0])).unwrap();

    let out = unary_map(SqrtOp, &array.view()).unwrap();
    assert_eq!(out.storage().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_unary_map_into_neg_and_abs() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let array = Array::new(layout, VecStorage::new(vec![-1.0f64, 2.0, -3.0, 4.0])).unwrap();

    let mut neg_out = Array::new(layout, VecStorage::fill(4, 0.0f64)).unwrap();
    unary_map_into(NegOp, &array.view(), &mut neg_out.view_mut()).unwrap();
    assert_eq!(neg_out.storage().as_slice(), &[1.0, -2.0, 3.0, -4.0]);

    let abs_out = unary_map(AbsOp, &array.view()).unwrap();
    assert_eq!(abs_out.storage().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_unary_map_exp_matches_reference() {
    let layout = Layout::c_contiguous([3]).unwrap();
    let values = vec![0.0f64, 1.0, -1.0];
    let array = Array::new(layout, VecStorage::new(values.clone())).unwrap();

    let out = unary_map(ExpOp, &array.view()).unwrap();
    let expected: Vec<f64> = values.iter().map(|x| x.exp()).collect();
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_powf_op_carries_exponent() {
    let layout = Layout::c_contiguous([3]).unwrap();
    let array = Array::new(layout, VecStorage::new(vec![1.0f64, 2.0, 3.0])).unwrap();

    let out = unary_map(PowfOp { exponent: 2.0 }, &array.view()).unwrap();
    assert_eq!(out.storage().as_slice(), &[1.0, 4.0, 9.0]);
}

#[test]
fn test_map_inplace_mutates_in_place() {
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let mut array = Array::new(layout, VecStorage::new(vec![1.0f64, 2.0, 3.0, 4.0])).unwrap();

    map_inplace(&mut array.view_mut(), |x| x * 10.0).unwrap();
    assert_eq!(array.storage().as_slice(), &[10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_map_inplace_on_transposed_view() {
    // Transposed (F-order) view: contiguous in memory order, so the fast path
    // applies, and every logical element is touched exactly once.
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let mut array = Array::new(
        layout,
        VecStorage::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .unwrap();

    {
        let mut transposed = array.view_mut().transpose_mut([1, 0]).unwrap();
        map_inplace(&mut transposed, |x| x + 100.0).unwrap();
    }
    assert_eq!(
        array.storage().as_slice(),
        &[101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    );
}

#[test]
fn test_scalar_map_add_and_mul() {
    let layout = Layout::c_contiguous([3]).unwrap();
    let array = Array::new(layout, VecStorage::new(vec![1.0f64, 2.0, 3.0])).unwrap();

    let added = scalar_map::<AddOp, _, 1>(&array.view(), 10.0).unwrap();
    assert_eq!(added.storage().as_slice(), &[11.0, 12.0, 13.0]);

    let mut out = Array::new(layout, VecStorage::fill(3, 0.0f64)).unwrap();
    scalar_map_into::<MulOp, _, 1>(&array.view(), 2.0, &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice(), &[2.0, 4.0, 6.0]);
}

#[test]
fn test_dot_contiguous_and_strided() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let a = Array::new(layout, VecStorage::new(vec![1.0f64, 2.0, 3.0, 4.0])).unwrap();
    let b = Array::new(layout, VecStorage::new(vec![5.0f64, 6.0, 7.0, 8.0])).unwrap();

    // 1*5 + 2*6 + 3*7 + 4*8 = 70
    assert_eq!(dot(&a.view(), &b.view()).unwrap(), 70.0);

    // Strided: a row of a transposed 2x2 matrix.
    let m = Array::new(
        Layout::c_contiguous([2, 2]).unwrap(),
        VecStorage::new(vec![1.0f64, 2.0, 3.0, 4.0]),
    )
    .unwrap();
    let transposed = m.transpose([1, 0]).unwrap();
    let col0 = transposed
        .slice_with::<1>(&[leto::SliceArg::Index(0), leto::SliceArg::All])
        .unwrap();
    // column 0 of [[1,2],[3,4]] is [1,3]
    let ones = Array::new(
        Layout::c_contiguous([2]).unwrap(),
        VecStorage::new(vec![1.0f64, 1.0]),
    )
    .unwrap();
    assert_eq!(dot(&col0, &ones.view()).unwrap(), 4.0);
}

#[test]
fn test_dot_shape_mismatch_rejected() {
    let a = Array::new(
        Layout::c_contiguous([3]).unwrap(),
        VecStorage::new(vec![1.0f64, 2.0, 3.0]),
    )
    .unwrap();
    let b = Array::new(
        Layout::c_contiguous([2]).unwrap(),
        VecStorage::new(vec![1.0f64, 2.0]),
    )
    .unwrap();
    assert!(dot(&a.view(), &b.view()).is_err());
}

#[test]
fn test_l2_normalize() {
    let layout = Layout::c_contiguous([3]).unwrap();
    let array = Array::new(layout, VecStorage::new(vec![3.0f64, 0.0, 4.0])).unwrap();
    let mut out = Array::new(layout, VecStorage::fill(3, 0.0f64)).unwrap();
    l2_normalize_into(&array.view(), &mut out.view_mut(), 0.0).unwrap();
    assert_close_slice(out.storage().as_slice(), &[0.6, 0.0, 0.8]);
}

#[test]
fn test_jaccard_distance() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let a = Array::new(layout, VecStorage::new(vec![0b1100u32, 0b1010, 0b1111, 0b0000])).unwrap();
    let b = Array::new(layout, VecStorage::new(vec![0b1010u32, 0b1100, 0b1111, 0b0000])).unwrap();
    let dist = jaccard_distance(&a.view(), &b.view()).unwrap();
    assert!((dist - 0.4).abs() <= EPS);
}

#[test]
fn test_hamming_distance() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let a = Array::new(layout, VecStorage::new(vec![0b1100u32, 0b1010, 0b1111, 0b0000])).unwrap();
    let b = Array::new(layout, VecStorage::new(vec![0b1010u32, 0b1100, 0b1111, 0b0000])).unwrap();
    let dist = hamming_distance(&a.view(), &b.view()).unwrap();
    assert_eq!(dist, 4);
}
