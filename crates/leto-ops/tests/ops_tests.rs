use leto::{Array, Layout, Storage, VecStorage};
use leto_ops::{
    add, binary_map, div, map, map_into, mapv, matmul, max_axis_into, mean_axis_into,
    min_axis_into, mul, sub, sum, sum_axis_into, zip_mut_with, AddOp, MulOp,
};
use ndarray::Array2;

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
fn test_elementwise_binary_ops() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let a_storage = VecStorage::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b_storage = VecStorage::new(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let out_storage = VecStorage::fill(6, 0.0f32);

    let a = Array::new(layout, a_storage).unwrap();
    let b = Array::new(layout, b_storage).unwrap();
    let mut out = Array::new(layout, out_storage).unwrap();

    add(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_eq!(
        out.storage().as_slice(),
        &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
    );

    // For subtraction, write into out2
    let out2_storage = VecStorage::fill(6, 0.0f32);
    let mut out2 = Array::new(layout, out2_storage).unwrap();
    sub(&out.view(), &a.view(), &mut out2.view_mut()).unwrap();
    assert_eq!(
        out2.storage().as_slice(),
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    );

    // For multiplication, write into out3
    let out3_storage = VecStorage::fill(6, 0.0f32);
    let mut out3 = Array::new(layout, out3_storage).unwrap();
    mul(&out2.view(), &a.view(), &mut out3.view_mut()).unwrap();
    assert_eq!(
        out3.storage().as_slice(),
        &[10.0, 40.0, 90.0, 160.0, 250.0, 360.0]
    );

    // For division, write into out4
    let out4_storage = VecStorage::fill(6, 0.0f32);
    let mut out4 = Array::new(layout, out4_storage).unwrap();
    div(&out3.view(), &a.view(), &mut out4.view_mut()).unwrap();
    assert_eq!(
        out4.storage().as_slice(),
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    );
}

#[test]
fn test_binary_map_zst_operation_entry_point() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let a = Array::new(layout, VecStorage::new(vec![1.0f32, 2.0, 3.0, 4.0])).unwrap();
    let b = Array::new(layout, VecStorage::new(vec![5.0f32, 6.0, 7.0, 8.0])).unwrap();
    let mut out = Array::new(layout, VecStorage::fill(4, 0.0f32)).unwrap();

    binary_map::<AddOp, _, 1>(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice(), &[6.0, 8.0, 10.0, 12.0]);

    binary_map::<MulOp, _, 1>(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice(), &[5.0, 12.0, 21.0, 32.0]);
}

#[test]
fn test_binary_map_strided_transposed_views() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let a = Array::new(
        layout,
        VecStorage::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .unwrap();
    let b = Array::new(
        layout,
        VecStorage::new(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0]),
    )
    .unwrap();
    let out_layout = Layout::c_contiguous([3, 2]).unwrap();
    let mut out = Array::new(out_layout, VecStorage::fill(6, 0.0f32)).unwrap();

    let a_t = a.transpose([1, 0]).unwrap();
    let b_t = b.transpose([1, 0]).unwrap();
    add(&a_t, &b_t, &mut out.view_mut()).unwrap();

    assert_eq!(
        out.storage().as_slice(),
        &[11.0, 44.0, 22.0, 55.0, 33.0, 66.0]
    );
}

#[test]
fn test_sum_reduction() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let storage = VecStorage::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let arr = Array::new(layout, storage).unwrap();

    let total = sum(&arr.view());
    assert_eq!(total, 21.0f64);
}

#[test]
fn test_map_into_uses_caller_owned_output() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let input = Array::new(layout, VecStorage::new(vec![1.0f32, -2.0, 3.5, 4.0])).unwrap();
    let mut output = Array::new(layout, VecStorage::fill(4, 0.0f32)).unwrap();

    map_into(&input.view(), &mut output.view_mut(), |value| value * value).unwrap();

    assert_eq!(output.storage().as_slice(), &[1.0, 4.0, 12.25, 16.0]);
}

#[test]
fn test_mapv_allocates_c_contiguous_output_with_explicit_conversion() {
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let input = Array::new(layout, VecStorage::new(vec![1.25f64, 2.5, 3.75, 4.0])).unwrap();

    let output = mapv(&input.view(), |value| value as f32).unwrap();

    assert_eq!(output.shape(), [2, 2]);
    assert!(output.layout().is_c_contiguous());
    assert_eq!(output.storage().as_slice(), &[1.25f32, 2.5, 3.75, 4.0]);
}

#[test]
fn test_map_into_handles_strided_transposed_input() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let input = Array::new(layout, VecStorage::new(vec![1i32, 2, 3, 4, 5, 6])).unwrap();
    let transposed = input.transpose([1, 0]).unwrap();
    let out_layout = Layout::c_contiguous([3, 2]).unwrap();
    let mut output = Array::new(out_layout, VecStorage::fill(6, 0i32)).unwrap();

    map_into(&transposed, &mut output.view_mut(), |value| value * 10).unwrap();

    assert_eq!(output.storage().as_slice(), &[10, 40, 20, 50, 30, 60]);
}

#[test]
fn test_axis_reductions_keep_reduced_dimension() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let input = Array::new(
        layout,
        VecStorage::new(vec![1.0f32, -2.0, 3.0, 4.0, 5.0, -6.0]),
    )
    .unwrap();

    let row_layout = Layout::c_contiguous([2, 1]).unwrap();
    let mut row_sum = Array::new(row_layout, VecStorage::fill(2, 0.0f32)).unwrap();
    let mut row_mean = Array::new(row_layout, VecStorage::fill(2, 0.0f32)).unwrap();
    let mut row_min = Array::new(row_layout, VecStorage::fill(2, 0.0f32)).unwrap();
    let mut row_max = Array::new(row_layout, VecStorage::fill(2, 0.0f32)).unwrap();

    sum_axis_into(&input.view(), 1, &mut row_sum.view_mut()).unwrap();
    mean_axis_into(&input.view(), 1, &mut row_mean.view_mut()).unwrap();
    min_axis_into(&input.view(), 1, &mut row_min.view_mut()).unwrap();
    max_axis_into(&input.view(), 1, &mut row_max.view_mut()).unwrap();

    assert_eq!(row_sum.storage().as_slice(), &[2.0, 3.0]);
    assert_eq!(row_mean.storage().as_slice(), &[2.0 / 3.0, 1.0]);
    assert_eq!(row_min.storage().as_slice(), &[-2.0, -6.0]);
    assert_eq!(row_max.storage().as_slice(), &[3.0, 5.0]);

    let col_layout = Layout::c_contiguous([1, 3]).unwrap();
    let mut col_sum = Array::new(col_layout, VecStorage::fill(3, 0.0f32)).unwrap();
    sum_axis_into(&input.view(), 0, &mut col_sum.view_mut()).unwrap();
    assert_eq!(col_sum.storage().as_slice(), &[5.0, 3.0, -3.0]);
}

#[test]
fn test_axis_reduction_rejects_output_shape_mismatch() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let input = Array::new(layout, VecStorage::new(vec![1.0f32; 6])).unwrap();
    let wrong_layout = Layout::c_contiguous([1, 3]).unwrap();
    let mut wrong = Array::new(wrong_layout, VecStorage::fill(3, 0.0f32)).unwrap();

    let result = sum_axis_into(&input.view(), 1, &mut wrong.view_mut());
    assert!(matches!(result, Err(leto::LetoError::ShapeMismatch { .. })));
}

#[test]
fn test_axis_reduction_strided_transposed_input() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let input = Array::new(
        layout,
        VecStorage::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .unwrap();
    let transposed = input.transpose([1, 0]).unwrap();
    let out_layout = Layout::c_contiguous([3, 1]).unwrap();
    let mut output = Array::new(out_layout, VecStorage::fill(3, 0.0f64)).unwrap();

    sum_axis_into(&transposed, 1, &mut output.view_mut()).unwrap();

    assert_eq!(output.storage().as_slice(), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_empty_axis_sum_is_zero_and_mean_is_rejected() {
    let layout = Layout::c_contiguous([2, 0]).unwrap();
    let input = Array::new(layout, VecStorage::new(Vec::<f32>::new())).unwrap();
    let out_layout = Layout::c_contiguous([2, 1]).unwrap();
    let mut sum_out = Array::new(out_layout, VecStorage::fill(2, 1.0f32)).unwrap();
    let mut mean_out = Array::new(out_layout, VecStorage::fill(2, 1.0f32)).unwrap();

    sum_axis_into(&input.view(), 1, &mut sum_out.view_mut()).unwrap();
    assert_eq!(sum_out.storage().as_slice(), &[0.0, 0.0]);

    let result = mean_axis_into(&input.view(), 1, &mut mean_out.view_mut());
    assert!(matches!(result, Err(leto::LetoError::StorageError { .. })));
}

#[test]
fn test_matmul() {
    // 2x3 matrix
    let lhs_layout = Layout::c_contiguous([2, 3]).unwrap();
    let lhs_storage = VecStorage::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let lhs = Array::new(lhs_layout, lhs_storage).unwrap();

    // 3x2 matrix
    let rhs_layout = Layout::c_contiguous([3, 2]).unwrap();
    let rhs_storage = VecStorage::new(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let rhs = Array::new(rhs_layout, rhs_storage).unwrap();

    // 2x2 output matrix
    let out_layout = Layout::c_contiguous([2, 2]).unwrap();
    let out_storage = VecStorage::fill(4, 0.0f32);
    let mut out = Array::new(out_layout, out_storage).unwrap();

    matmul(&lhs.view(), &rhs.view(), &mut out.view_mut()).unwrap();

    assert_eq!(out.storage().as_slice(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_mapping_and_zipping() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let arr = Array::new(layout, VecStorage::new(vec![1, 2, 3, 4, 5, 6])).unwrap();

    // Map by reference
    let mapped = map(&arr.view(), |x| x * 10).unwrap();
    assert_eq!(mapped.storage().as_slice(), &[10, 20, 30, 40, 50, 60]);
    assert!(mapped.layout().is_c_contiguous());

    // Map on a transposed strided view
    let transposed = arr.transpose([1, 0]).unwrap();
    let mapped_t = map(&transposed, |x| x + 1).unwrap();
    assert_eq!(mapped_t.storage().as_slice(), &[2, 5, 3, 6, 4, 7]);
    assert!(mapped_t.layout().is_c_contiguous());

    // Zip-mapping in place
    let mut dest = Array::new(layout, VecStorage::fill(6, 100)).unwrap();
    zip_mut_with(&mut dest.view_mut(), &arr.view(), |d, &s| {
        *d += s;
    })
    .unwrap();
    assert_eq!(dest.storage().as_slice(), &[101, 102, 103, 104, 105, 106]);

    // Shape mismatch validation
    let wrong_layout = Layout::c_contiguous([3, 2]).unwrap();
    let wrong_arr = Array::new(wrong_layout, VecStorage::fill(6, 0)).unwrap();
    let mut dest_mut = dest.view_mut();
    assert!(zip_mut_with(&mut dest_mut, &wrong_arr.view(), |_, _| {}).is_err());
}

#[test]
fn test_zip_mut_with_handles_strided_transposed_views() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let lhs_base = Array::new(layout, VecStorage::new(vec![1i32, 2, 3, 4, 5, 6])).unwrap();
    let rhs_base = Array::new(layout, VecStorage::new(vec![10i32, 20, 30, 40, 50, 60])).unwrap();
    let mut lhs_storage = lhs_base.into_vec();
    let mut lhs_view = leto::ArrayViewMut::try_new(
        Layout::c_contiguous([2, 3]).unwrap(),
        lhs_storage.as_mut_slice(),
    )
    .unwrap()
    .transpose_mut([1, 0])
    .unwrap();
    let rhs_view = rhs_base.transpose([1, 0]).unwrap();

    zip_mut_with(&mut lhs_view, &rhs_view, |left, right| {
        *left += *right;
    })
    .unwrap();

    assert_eq!(lhs_storage.as_slice(), &[11, 22, 33, 44, 55, 66]);
}

#[test]
fn test_matmul_strided_and_transposed() {
    // 2x3 matrix
    let lhs_storage = VecStorage::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let lhs = Array::new(Layout::c_contiguous([2, 3]).unwrap(), lhs_storage).unwrap();

    // 2x3 matrix to be transposed into 3x2
    let rhs_storage = VecStorage::new(vec![7.0f32, 9.0, 11.0, 8.0, 10.0, 12.0]);
    let rhs_t = Array::new(Layout::c_contiguous([2, 3]).unwrap(), rhs_storage).unwrap();
    let rhs = rhs_t.transpose([1, 0]).unwrap(); // shape [3, 2] with strides [1, 3]

    // 2x2 output matrix with stride 1 column
    let mut out = Array::new(
        Layout::c_contiguous([2, 2]).unwrap(),
        VecStorage::fill(4, 0.0f32),
    )
    .unwrap();

    matmul(&lhs.view(), &rhs, &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice(), &[58.0, 64.0, 139.0, 154.0]);

    // Transposed lhs and transposed rhs
    let lhs_storage2 = VecStorage::new(vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let lhs2_t = Array::new(Layout::c_contiguous([3, 2]).unwrap(), lhs_storage2).unwrap();
    let lhs2 = lhs2_t.transpose([1, 0]).unwrap(); // shape [2, 3] with strides [1, 2]

    let rhs2 = Array::new(
        Layout::c_contiguous([3, 2]).unwrap(),
        VecStorage::new(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]),
    )
    .unwrap();

    let mut out2 = Array::new(
        Layout::c_contiguous([2, 2]).unwrap(),
        VecStorage::fill(4, 0.0f32),
    )
    .unwrap();
    matmul(&lhs2, &rhs2.view(), &mut out2.view_mut()).unwrap();
    assert_eq!(out2.storage().as_slice(), &[58.0, 64.0, 139.0, 154.0]);
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
