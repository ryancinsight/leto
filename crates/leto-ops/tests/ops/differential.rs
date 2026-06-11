use leto::{concat, stack, Array, SliceArg, Storage};
use leto_ops::{
    add, batched_matmul, cumsum, mapv, matmul, max_axis, mean_axis, min_axis, scalar_map, sum,
    sum_axis, unary_map, AddOp, ExpOp, SqrtOp,
};
use ndarray::{s, Array2, Array3, Axis};

fn assert_close_slice(lhs: &[f32], rhs: &[f32]) {
    assert_eq!(lhs.len(), rhs.len());
    for (left, right) in lhs.iter().zip(rhs.iter()) {
        assert!(
            (*left - *right).abs() <= 1.0e-5,
            "left {left} differs from right {right}"
        );
    }
}

#[test]
fn test_matmul_differential_matches_ndarray_contiguous() {
    let lhs_values = vec![1.0f32, -2.0, 3.0, 4.5, 0.25, -6.0];
    let rhs_values = vec![7.0f32, 8.0, -9.0, 10.0, 11.0, -12.0];
    let lhs = Array::from_shape_vec([2, 3], lhs_values.clone()).unwrap();
    let rhs = Array::from_shape_vec([3, 2], rhs_values.clone()).unwrap();
    let mut out = Array::zeros([2, 2]);

    matmul(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();

    let ndarray_lhs = Array2::from_shape_vec((2, 3), lhs_values).unwrap();
    let ndarray_rhs = Array2::from_shape_vec((3, 2), rhs_values).unwrap();
    let expected = ndarray_lhs.dot(&ndarray_rhs);
    assert_close_slice(out.storage().as_slice(), expected.as_slice().unwrap());
}

#[test]
fn test_matmul_differential_matches_ndarray_transposed_views() {
    let lhs_base_values = vec![1.0f32, 4.5, -2.0, 0.25, 3.0, -6.0];
    let rhs_base_values = vec![7.0f32, -9.0, 11.0, 8.0, 10.0, -12.0];
    let lhs_base = Array::from_shape_vec([3, 2], lhs_base_values.clone()).unwrap();
    let rhs_base = Array::from_shape_vec([2, 3], rhs_base_values.clone()).unwrap();
    let lhs = lhs_base.transpose([1, 0]).unwrap();
    let rhs = rhs_base.transpose([1, 0]).unwrap();
    let mut out = Array::zeros([2, 2]);

    matmul(&lhs, &rhs, &mut out.view_mut()).unwrap();

    let ndarray_lhs_base = Array2::from_shape_vec((3, 2), lhs_base_values).unwrap();
    let ndarray_rhs_base = Array2::from_shape_vec((2, 3), rhs_base_values).unwrap();
    let expected = ndarray_lhs_base.t().dot(&ndarray_rhs_base.t());
    let expected_values: Vec<f32> = expected.iter().copied().collect();
    assert_close_slice(out.storage().as_slice(), &expected_values);
}

#[test]
fn test_map_differential_matches_ndarray_contiguous_and_transposed_views() {
    let values = vec![1.0f32, -2.0, 3.5, 4.25, -5.5, 6.75];
    let input = Array::from_shape_vec([2, 3], values.clone()).unwrap();

    let mapped = mapv(&input.view(), |value| value.mul_add(value, -1.0)).unwrap();

    let expected = Array2::from_shape_vec((2, 3), values.clone())
        .unwrap()
        .mapv(|value| value.mul_add(value, -1.0));
    assert_close_slice(mapped.storage().as_slice(), expected.as_slice().unwrap());

    let transposed = input.transpose([1, 0]).unwrap();
    let mapped_transposed = mapv(&transposed, |value| value + 0.5).unwrap();
    let ndarray_input = Array2::from_shape_vec((2, 3), values).unwrap();
    let expected_transposed = ndarray_input.t().mapv(|value| value + 0.5);
    let expected_values: Vec<f32> = expected_transposed.iter().copied().collect();

    assert_eq!(mapped_transposed.shape(), [3, 2]);
    assert_close_slice(mapped_transposed.storage().as_slice(), &expected_values);
}

#[test]
fn test_binary_broadcast_differential_matches_ndarray() {
    let lhs_values = vec![1.0f32, 10.0];
    let rhs_values = vec![2.0f32, -3.0, 4.5];
    let lhs = Array::from_shape_vec([2, 1], lhs_values.clone()).unwrap();
    let rhs = Array::from_shape_vec([1, 3], rhs_values.clone()).unwrap();
    let mut out = Array::zeros([2, 3]);

    add(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();

    let ndarray_lhs = Array2::from_shape_vec((2, 1), lhs_values).unwrap();
    let ndarray_rhs = Array2::from_shape_vec((1, 3), rhs_values).unwrap();
    let expected =
        ndarray_lhs.broadcast((2, 3)).unwrap().to_owned() + ndarray_rhs.broadcast((2, 3)).unwrap();
    assert_close_slice(out.storage().as_slice(), expected.as_slice().unwrap());
}

#[test]
fn test_row_walk_maps_negative_last_axis_stride() {
    let values = vec![1.0f32, -2.0, 3.5, 4.25, -5.5, 6.75, 7.5, -8.25];
    let input = Array::from_shape_vec([2, 4], values.clone()).unwrap();
    let reversed = input
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();

    let mapped = mapv(&reversed, |value| value.mul_add(2.0, 1.0)).unwrap();

    let ndarray_input = Array2::from_shape_vec((2, 4), values).unwrap();
    let expected = ndarray_input
        .slice(s![.., ..;-1])
        .mapv(|value| value.mul_add(2.0, 1.0));
    let expected_values: Vec<f32> = expected.iter().copied().collect();
    assert_eq!(mapped.shape(), [2, 4]);
    assert_close_slice(mapped.storage().as_slice(), &expected_values);
}

#[test]
fn test_row_walk_binary_map_negative_last_axis_stride() {
    let lhs_values = vec![1.0f32, -2.0, 3.5, 4.25, -5.5, 6.75, 7.5, -8.25];
    let rhs_values = vec![0.5f32, 1.5, -3.0, 2.0, 4.0, -1.0, 6.0, 8.0];
    let lhs_base = Array::from_shape_vec([2, 4], lhs_values.clone()).unwrap();
    let rhs_base = Array::from_shape_vec([2, 4], rhs_values.clone()).unwrap();
    let lhs = lhs_base
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();
    let rhs = rhs_base
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();
    let mut out = Array::zeros([2, 4]);

    add(&lhs, &rhs, &mut out.view_mut()).unwrap();

    let ndarray_lhs = Array2::from_shape_vec((2, 4), lhs_values).unwrap();
    let ndarray_rhs = Array2::from_shape_vec((2, 4), rhs_values).unwrap();
    let expected =
        ndarray_lhs.slice(s![.., ..;-1]).to_owned() + ndarray_rhs.slice(s![.., ..;-1]).to_owned();
    let expected_values: Vec<f32> = expected.iter().copied().collect();
    assert_close_slice(out.storage().as_slice(), &expected_values);
}

#[test]
fn test_row_walk_sum_negative_last_axis_stride() {
    let values = vec![1.0f32, -2.0, 3.5, 4.25, -5.5, 6.75, 7.5, -8.25];
    let input = Array::from_shape_vec([2, 4], values.clone()).unwrap();
    let reversed = input
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();

    let actual = sum(&reversed);

    let ndarray_input = Array2::from_shape_vec((2, 4), values).unwrap();
    let expected = ndarray_input.slice(s![.., ..;-1]).sum();
    assert!(
        (actual - expected).abs() <= 1.0e-5,
        "actual {actual} expected {expected}"
    );
}

#[test]
fn test_axis_reductions_differential_match_ndarray_keepdim() {
    let values = vec![1.0f32, -2.0, 3.5, 4.25, -5.5, 6.75];
    let input = Array::from_shape_vec([2, 3], values.clone()).unwrap();
    let ndarray_input = Array2::from_shape_vec((2, 3), values).unwrap();

    let row_sum = sum_axis(&input.view(), 1).unwrap();
    let row_mean = mean_axis(&input.view(), 1).unwrap();
    let row_min = min_axis(&input.view(), 1).unwrap();
    let row_max = max_axis(&input.view(), 1).unwrap();

    let expected_row_sum = ndarray_input.sum_axis(Axis(1)).insert_axis(Axis(1));
    let expected_row_mean = ndarray_input
        .mean_axis(Axis(1))
        .unwrap()
        .insert_axis(Axis(1));
    let expected_row_min = ndarray_input
        .map_axis(Axis(1), |row| {
            row.iter().copied().fold(f32::INFINITY, f32::min)
        })
        .insert_axis(Axis(1));
    let expected_row_max = ndarray_input
        .map_axis(Axis(1), |row| {
            row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        })
        .insert_axis(Axis(1));

    assert_eq!(row_sum.shape(), [2, 1]);
    assert_close_slice(
        row_sum.storage().as_slice(),
        expected_row_sum.as_slice().unwrap(),
    );
    assert_close_slice(
        row_mean.storage().as_slice(),
        expected_row_mean.as_slice().unwrap(),
    );
    assert_close_slice(
        row_min.storage().as_slice(),
        expected_row_min.as_slice().unwrap(),
    );
    assert_close_slice(
        row_max.storage().as_slice(),
        expected_row_max.as_slice().unwrap(),
    );

    let col_sum = sum_axis(&input.view(), 0).unwrap();
    let expected_col_sum = ndarray_input.sum_axis(Axis(0)).insert_axis(Axis(0));
    assert_eq!(col_sum.shape(), [1, 3]);
    assert_close_slice(
        col_sum.storage().as_slice(),
        expected_col_sum.as_slice().unwrap(),
    );
}

#[test]
fn test_unary_map_differential_matches_ndarray() {
    let values = vec![0.25f32, 1.0, 2.5, 4.0, 0.5, 9.0];
    let input = Array::from_shape_vec([2, 3], values.clone()).unwrap();
    let ndarray_input = Array2::from_shape_vec((2, 3), values).unwrap();

    let exp = unary_map(ExpOp, &input.view()).unwrap();
    let expected_exp = ndarray_input.mapv(|v| v.exp());
    assert_close_slice(exp.storage().as_slice(), expected_exp.as_slice().unwrap());

    let sqrt = unary_map(SqrtOp, &input.view()).unwrap();
    let expected_sqrt = ndarray_input.mapv(|v| v.sqrt());
    assert_close_slice(sqrt.storage().as_slice(), expected_sqrt.as_slice().unwrap());
}

#[test]
fn test_scalar_map_differential_matches_ndarray() {
    let values = vec![1.0f32, -2.0, 3.5, 4.25];
    let input = Array::from_shape_vec([2, 2], values.clone()).unwrap();
    let ndarray_input = Array2::from_shape_vec((2, 2), values).unwrap();

    let shifted = scalar_map::<AddOp, _, 2>(&input.view(), 10.0).unwrap();
    let expected = ndarray_input.mapv(|v| v + 10.0);
    assert_close_slice(shifted.storage().as_slice(), expected.as_slice().unwrap());
}

#[test]
fn test_concat_differential_matches_ndarray() {
    let a_values = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_values = vec![5.0f32, 6.0];
    let a = Array::from_shape_vec([2, 2], a_values.clone()).unwrap();
    let b = Array::from_shape_vec([1, 2], b_values.clone()).unwrap();

    let out = concat(&[a.view(), b.view()], 0).unwrap();

    let nd_a = Array2::from_shape_vec((2, 2), a_values).unwrap();
    let nd_b = Array2::from_shape_vec((1, 2), b_values).unwrap();
    let expected = ndarray::concatenate(Axis(0), &[nd_a.view(), nd_b.view()]).unwrap();
    assert_eq!(out.shape(), [3, 2]);
    assert_close_slice(out.storage().as_slice(), expected.as_slice().unwrap());
}

#[test]
fn test_stack_differential_matches_ndarray() {
    let a_values = vec![1.0f32, 2.0, 3.0];
    let b_values = vec![4.0f32, 5.0, 6.0];
    let a = Array::from_shape_vec([3], a_values.clone()).unwrap();
    let b = Array::from_shape_vec([3], b_values.clone()).unwrap();

    let out = stack::<f32, 1, 2>(&[a.view(), b.view()], 1).unwrap();

    let nd_a = ndarray::Array1::from_vec(a_values);
    let nd_b = ndarray::Array1::from_vec(b_values);
    let expected = ndarray::stack(Axis(1), &[nd_a.view(), nd_b.view()]).unwrap();
    let expected_values: Vec<f32> = expected.iter().copied().collect();
    assert_eq!(out.shape(), [3, 2]);
    assert_close_slice(out.storage().as_slice(), &expected_values);
}

#[test]
fn test_batched_matmul_differential_matches_ndarray_per_batch() {
    let lhs_values: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let rhs_values: Vec<f32> = (1..=12).map(|x| (x as f32) * 0.5).collect();
    // [2, 2, 3] x [2, 3, 2] -> [2, 2, 2]
    let lhs = Array::from_shape_vec([2, 2, 3], lhs_values.clone()).unwrap();
    let rhs = Array::from_shape_vec([2, 3, 2], rhs_values.clone()).unwrap();
    let mut out = Array::zeros([2, 2, 2]);

    batched_matmul(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();

    let nd_lhs = Array3::from_shape_vec((2, 2, 3), lhs_values).unwrap();
    let nd_rhs = Array3::from_shape_vec((2, 3, 2), rhs_values).unwrap();
    let mut expected = Vec::with_capacity(8);
    for b in 0..2 {
        let l = nd_lhs.index_axis(Axis(0), b).to_owned();
        let r = nd_rhs.index_axis(Axis(0), b).to_owned();
        let prod = l.dot(&r);
        expected.extend(prod.iter().copied());
    }
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_cumsum_differential_matches_reference_accumulate() {
    let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Array::from_shape_vec([2, 3], values.clone()).unwrap();

    let out = cumsum(&input.view(), 1).unwrap();

    // Reference: running sum along axis 1 of each row.
    let mut expected = vec![0.0f32; 6];
    for row in 0..2 {
        let mut acc = 0.0f32;
        for col in 0..3 {
            acc += values[row * 3 + col];
            expected[row * 3 + col] = acc;
        }
    }
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_axis_reductions_differential_match_ndarray_transposed_views() {
    let values = vec![1.0f32, 4.25, -2.0, -5.5, 3.5, 6.75];
    let input = Array::from_shape_vec([3, 2], values.clone()).unwrap();
    let transposed = input.transpose([1, 0]).unwrap();
    let reduced = sum_axis(&transposed, 1).unwrap();

    let ndarray_input = Array2::from_shape_vec((3, 2), values).unwrap();
    let expected = ndarray_input.t().sum_axis(Axis(1)).insert_axis(Axis(1));
    let expected_values: Vec<f32> = expected.iter().copied().collect();

    assert_eq!(reduced.shape(), [2, 1]);
    assert_close_slice(reduced.storage().as_slice(), &expected_values);
}
