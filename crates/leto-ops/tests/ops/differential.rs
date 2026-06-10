use leto::{Array, Storage};
use leto_ops::{add, mapv, matmul, max_axis, mean_axis, min_axis, sum_axis};
use ndarray::{Array2, Axis};

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
