#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array, Layout, SliceArg, Storage, VecStorage};
use leto_ops::{
    max_axis, max_axis_into, mean_axis, mean_axis_into, min_axis, min_axis_into, product_axis,
    product_axis_into, sum, sum_axis, sum_axis_into,
};

#[test]
fn test_sum_reduction() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let storage = VecStorage::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let arr = Array::new(layout, storage).unwrap();

    let total = sum(&arr.view());
    assert_eq!(total, 21.0f64);
}

#[test]
fn whole_reduction_preserves_non_unit_stride_values() {
    let input = Array::from_shape_vec(
        [2, 4],
        vec![1.0f64, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0],
    )
    .unwrap();
    let selected = input
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(Some(0), None, 2)])
        .unwrap();

    assert_eq!(sum(&selected), 10.0);
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
    let mut row_product = Array::new(row_layout, VecStorage::fill(2, 0.0f32)).unwrap();

    sum_axis_into(&input.view(), 1, &mut row_sum.view_mut()).unwrap();
    mean_axis_into(&input.view(), 1, &mut row_mean.view_mut()).unwrap();
    min_axis_into(&input.view(), 1, &mut row_min.view_mut()).unwrap();
    max_axis_into(&input.view(), 1, &mut row_max.view_mut()).unwrap();
    product_axis_into(&input.view(), 1, &mut row_product.view_mut()).unwrap();

    assert_eq!(row_sum.storage().as_slice(), &[2.0, 3.0]);
    assert_eq!(row_mean.storage().as_slice(), &[2.0 / 3.0, 1.0]);
    assert_eq!(row_min.storage().as_slice(), &[-2.0, -6.0]);
    assert_eq!(row_max.storage().as_slice(), &[3.0, 5.0]);
    assert_eq!(row_product.storage().as_slice(), &[-6.0, -120.0]);

    let col_layout = Layout::c_contiguous([1, 3]).unwrap();
    let mut col_sum = Array::new(col_layout, VecStorage::fill(3, 0.0f32)).unwrap();
    sum_axis_into(&input.view(), 0, &mut col_sum.view_mut()).unwrap();
    assert_eq!(col_sum.storage().as_slice(), &[5.0, 3.0, -3.0]);
}

#[test]
fn test_allocating_axis_reductions_keep_reduced_dimension() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let input = Array::new(
        layout,
        VecStorage::new(vec![1.0f32, -2.0, 3.0, 4.0, 5.0, -6.0]),
    )
    .unwrap();

    let row_sum = sum_axis(&input.view(), 1).unwrap();
    let row_mean = mean_axis(&input.view(), 1).unwrap();
    let row_min = min_axis(&input.view(), 1).unwrap();
    let row_max = max_axis(&input.view(), 1).unwrap();
    let row_product = product_axis(&input.view(), 1).unwrap();

    assert_eq!(row_sum.shape(), [2, 1]);
    assert!(row_sum.layout().is_c_contiguous());
    assert_eq!(row_sum.storage().as_slice(), &[2.0, 3.0]);
    assert_eq!(row_mean.storage().as_slice(), &[2.0 / 3.0, 1.0]);
    assert_eq!(row_min.storage().as_slice(), &[-2.0, -6.0]);
    assert_eq!(row_max.storage().as_slice(), &[3.0, 5.0]);
    assert_eq!(row_product.shape(), [2, 1]);
    assert_eq!(row_product.storage().as_slice(), &[-6.0, -120.0]);

    let col_sum = sum_axis(&input.view(), 0).unwrap();
    assert_eq!(col_sum.shape(), [1, 3]);
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
fn test_allocating_axis_reduction_handles_strided_transposed_input() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let input = Array::new(
        layout,
        VecStorage::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .unwrap();
    let transposed = input.transpose([1, 0]).unwrap();

    let output = sum_axis(&transposed, 1).unwrap();

    assert_eq!(output.shape(), [3, 1]);
    assert!(output.layout().is_c_contiguous());
    assert_eq!(output.storage().as_slice(), &[5.0, 7.0, 9.0]);
}

#[test]
fn axis_reductions_handle_negative_stride_input() {
    let input = Array::from_shape_vec([2, 3], vec![1i32, 2, 3, 4, 5, 6]).unwrap();
    let reversed_cols = input
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(Some(-1), None, -1)])
        .unwrap();

    let row_sum = sum_axis(&reversed_cols, 1).unwrap();
    let row_min = min_axis(&reversed_cols, 1).unwrap();
    let col_max = max_axis(&reversed_cols, 0).unwrap();

    assert_eq!(row_sum.storage().as_slice(), &[6, 15]);
    assert_eq!(row_min.storage().as_slice(), &[1, 4]);
    assert_eq!(col_max.storage().as_slice(), &[6, 5, 4]);
}

#[test]
fn test_empty_axis_sum_is_zero_and_mean_is_rejected() {
    let layout = Layout::c_contiguous([2, 0]).unwrap();
    let input = Array::new(layout, VecStorage::new(Vec::<f32>::new())).unwrap();
    let out_layout = Layout::c_contiguous([2, 1]).unwrap();
    let mut sum_out = Array::new(out_layout, VecStorage::fill(2, 1.0f32)).unwrap();
    let mut product_out = Array::new(out_layout, VecStorage::fill(2, 0.0f32)).unwrap();
    let mut mean_out = Array::new(out_layout, VecStorage::fill(2, 1.0f32)).unwrap();

    sum_axis_into(&input.view(), 1, &mut sum_out.view_mut()).unwrap();
    product_axis_into(&input.view(), 1, &mut product_out.view_mut()).unwrap();
    assert_eq!(sum_out.storage().as_slice(), &[0.0, 0.0]);
    assert_eq!(product_out.storage().as_slice(), &[1.0, 1.0]);

    let result = mean_axis_into(&input.view(), 1, &mut mean_out.view_mut());
    assert!(matches!(result, Err(leto::LetoError::StorageError { .. })));
}

#[test]
fn test_allocating_empty_axis_sum_is_zero_and_mean_is_rejected() {
    let layout = Layout::c_contiguous([2, 0]).unwrap();
    let input = Array::new(layout, VecStorage::new(Vec::<f32>::new())).unwrap();

    let sum_output = sum_axis(&input.view(), 1).unwrap();
    assert_eq!(sum_output.shape(), [2, 1]);
    assert_eq!(sum_output.storage().as_slice(), &[0.0, 0.0]);

    let result = mean_axis(&input.view(), 1);
    assert!(matches!(result, Err(leto::LetoError::StorageError { .. })));
}

#[test]
fn integer_scalar_reductions_are_value_semantic() {
    let input = Array::from_shape_vec([2, 3], vec![3i32, -7, 11, 13, -17, 19]).unwrap();

    let total = sum(&input.view());
    let row_sum = sum_axis(&input.view(), 1).unwrap();
    let col_min = min_axis(&input.view(), 0).unwrap();
    let row_max = max_axis(&input.view(), 1).unwrap();

    assert_eq!(total, 22);
    assert_eq!(row_sum.storage().as_slice(), &[7, 15]);
    assert_eq!(col_min.storage().as_slice(), &[3, -17, 11]);
    assert_eq!(row_max.storage().as_slice(), &[11, 19]);
}

#[test]
fn unsigned_integer_reduction_handles_strided_transposed_input() {
    let input = Array::from_shape_vec([2, 3], vec![2u64, 3, 5, 7, 11, 13]).unwrap();
    let transposed = input.transpose([1, 0]).unwrap();

    let row_sum = sum_axis(&transposed, 1).unwrap();
    let row_max = max_axis(&transposed, 1).unwrap();

    assert_eq!(row_sum.shape(), [3, 1]);
    assert_eq!(row_sum.storage().as_slice(), &[9, 14, 18]);
    assert_eq!(row_max.storage().as_slice(), &[7, 11, 13]);
}

/// NaN lanes are ignored by `min_axis`/`max_axis` on both routes — the
/// contiguous axis delegates to the SIMD `min_slice`/`max_slice`, the strided
/// axis runs the scalar fold — and an all-NaN axis reduces to the identity
/// (`+∞` for min, `−∞` for max), so the result never depends on layout or on
/// where the NaN sits.
fn nan_lanes_are_ignored_on_both_routes<T>()
where
    T: leto_ops::Scalar + From<f32> + PartialEq + core::fmt::Debug,
{
    let v = |x: f32| T::from(x);
    let nan = f32::NAN;
    #[rustfmt::skip]
    let rows = [
        [nan, 3.0, -1.0, 2.0],
        [3.0, nan, -1.0, 2.0],
        [3.0, -1.0, 2.0, nan],
        [nan, nan, nan, nan],
        [3.0, -1.0, 2.0, 0.5],
    ];
    let input =
        Array::from_shape_vec([5, 4], rows.iter().flatten().map(|&x| v(x)).collect()).unwrap();

    let row_min = min_axis(&input.view(), 1).unwrap();
    let row_max = max_axis(&input.view(), 1).unwrap();
    let inf = v(f32::INFINITY);
    let neg_inf = v(f32::NEG_INFINITY);
    assert_eq!(
        row_min.storage().as_slice(),
        &[v(-1.0), v(-1.0), v(-1.0), inf, v(-1.0)],
        "contiguous min ignores NaN wherever it sits and returns +inf for an all-NaN row"
    );
    assert_eq!(
        row_max.storage().as_slice(),
        &[v(3.0), v(3.0), v(3.0), neg_inf, v(3.0)],
        "contiguous max ignores NaN wherever it sits and returns -inf for an all-NaN row"
    );

    let col_min = min_axis(&input.view(), 0).unwrap();
    let col_max = max_axis(&input.view(), 0).unwrap();
    assert_eq!(
        col_min.storage().as_slice(),
        &[v(3.0), v(-1.0), v(-1.0), v(0.5)],
        "strided min ignores a leading, interior, or trailing NaN"
    );
    assert_eq!(
        col_max.storage().as_slice(),
        &[v(3.0), v(3.0), v(2.0), v(2.0)],
        "strided max ignores a leading, interior, or trailing NaN"
    );
}

#[test]
fn min_max_axis_ignore_nan_lanes_on_both_routes() {
    nan_lanes_are_ignored_on_both_routes::<f32>();
    nan_lanes_are_ignored_on_both_routes::<f64>();
}
