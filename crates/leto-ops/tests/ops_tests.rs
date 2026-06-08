use leto::{Array, Layout, Storage, VecStorage};
use leto_ops::{
    add, binary_map, div, matmul, max_axis_into, mean_axis_into, min_axis_into, mul, sub, sum,
    sum_axis_into, AddOp, MulOp,
};

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
