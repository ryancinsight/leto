use leto::{
    domain::{RankMarker, RemoveAxis},
    Array, Array1, Array2, ArrayView, ArrayViewMut, AxisIter, Layout, LetoError, SliceArg, Storage,
    VecStorage,
};

#[test]
fn test_c_contiguous_layout() {
    let layout = Layout::c_contiguous([2, 3, 4]).unwrap();
    assert_eq!(layout.shape, [2, 3, 4]);
    assert_eq!(layout.strides, [12, 4, 1]);
    assert_eq!(layout.offset, 0);
    assert_eq!(layout.size(), 24);
    assert!(layout.is_c_contiguous());
    assert!(!layout.is_f_contiguous());

    let (min_off, max_off) = layout.min_max_offsets();
    assert_eq!(min_off, 0);
    assert_eq!(max_off, 23);
}

#[test]
fn test_f_contiguous_layout() {
    let layout = Layout::f_contiguous([2, 3, 4]).unwrap();
    assert_eq!(layout.shape, [2, 3, 4]);
    assert_eq!(layout.strides, [1, 2, 6]);
    assert_eq!(layout.offset, 0);
    assert_eq!(layout.size(), 24);
    assert!(!layout.is_c_contiguous());
    assert!(layout.is_f_contiguous());

    let (min_off, max_off) = layout.min_max_offsets();
    assert_eq!(min_off, 0);
    assert_eq!(max_off, 23);
}

#[test]
fn test_offset_calculation() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    assert_eq!(layout.offset_of([0, 0]).unwrap(), 0);
    assert_eq!(layout.offset_of([0, 1]).unwrap(), 1);
    assert_eq!(layout.offset_of([1, 0]).unwrap(), 3);
    assert_eq!(layout.offset_of([1, 2]).unwrap(), 5);

    assert!(matches!(
        layout.offset_of([2, 0]),
        Err(LetoError::OutOfBounds { .. })
    ));
}

#[test]
fn test_rank_marker_removes_axis_shape_and_strides() {
    let marker = RankMarker::<3>;

    assert_eq!(marker.remove_shape([2, 3, 4], 0).unwrap(), [3, 4]);
    assert_eq!(marker.remove_shape([2, 3, 4], 1).unwrap(), [2, 4]);
    assert_eq!(marker.remove_shape([2, 3, 4], 2).unwrap(), [2, 3]);

    assert_eq!(marker.remove_strides([12, 4, 1], 0).unwrap(), [4, 1]);
    assert_eq!(marker.remove_strides([12, 4, 1], 1).unwrap(), [12, 1]);
    assert_eq!(marker.remove_strides([12, 4, 1], 2).unwrap(), [12, 4]);
}

#[test]
fn test_rank_marker_rejects_out_of_bounds_axis() {
    let marker = RankMarker::<2>;

    assert!(matches!(
        marker.remove_shape([2, 3], 2),
        Err(LetoError::StorageError { .. })
    ));
    assert!(matches!(
        marker.remove_strides([3, 1], 2),
        Err(LetoError::StorageError { .. })
    ));
}

#[test]
fn test_negative_stride_offsets_are_checked_before_unsigned_conversion() {
    let layout = Layout::new([3], [-1], 2);

    assert_eq!(layout.checked_min_max_offsets().unwrap(), (0, 2));
    assert_eq!(layout.offset_of([0]).unwrap(), 2);
    assert_eq!(layout.offset_of([2]).unwrap(), 0);

    let invalid = Layout::new([3], [-1], 0);
    assert!(matches!(
        invalid.checked_min_max_offsets(),
        Err(LetoError::StorageError { .. })
    ));
    assert!(matches!(
        invalid.offset_of([1]),
        Err(LetoError::StorageError { .. })
    ));
}

#[test]
fn test_array_rejects_layout_that_reaches_one_past_storage() {
    let layout = Layout::new([2], [1], 1);
    let storage = VecStorage::new(vec![10, 20]);

    assert!(matches!(
        Array::new(layout, storage),
        Err(LetoError::StorageError { .. })
    ));
}

#[test]
fn test_legacy_slice_empty_range_has_zero_extent_without_underflow() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let sliced = layout.slice(&[(2, 2, 1)]).unwrap();

    assert_eq!(sliced.shape, [0]);
    assert_eq!(sliced.strides, [1]);
    assert_eq!(sliced.offset, 2);
    assert_eq!(sliced.checked_size().unwrap(), 0);
}

#[test]
fn test_view_try_new_rejects_external_out_of_bounds_layout() {
    let data = [10, 20];
    let invalid = Layout::new([2], [1], 1);

    assert!(matches!(
        ArrayView::try_new(invalid, &data),
        Err(LetoError::StorageError { .. })
    ));

    let valid = ArrayView::try_new(Layout::new([2], [1], 0), &data).unwrap();
    assert_eq!(*valid.get([1]).unwrap(), 20);
}

#[test]
fn test_view_mut_try_new_rejects_external_out_of_bounds_layout() {
    let mut data = [10, 20];
    let invalid = Layout::new([2], [1], 1);

    assert!(matches!(
        ArrayViewMut::try_new(invalid, &mut data),
        Err(LetoError::StorageError { .. })
    ));
}

#[test]
fn test_array_creation_and_indexing() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let storage = VecStorage::new(vec![10, 11, 12, 20, 21, 22]);
    let array = Array::new(layout, storage).unwrap();

    assert_eq!(*array.get([0, 0]).unwrap(), 10);
    assert_eq!(*array.get([0, 2]).unwrap(), 12);
    assert_eq!(*array.get([1, 1]).unwrap(), 21);
}

#[test]
fn test_rank_aliases_construct_owned_arrays_and_views() {
    let vector: Array1<i32> = Array::new(
        Layout::c_contiguous([3]).unwrap(),
        VecStorage::new(vec![1, 2, 3]),
    )
    .unwrap();
    assert_eq!(*vector.view().get([2]).unwrap(), 3);

    let matrix: Array2<i32> = Array::new(
        Layout::c_contiguous([2, 2]).unwrap(),
        VecStorage::new(vec![1, 2, 3, 4]),
    )
    .unwrap();
    assert_eq!(*matrix.view().get([1, 0]).unwrap(), 3);
}

#[test]
fn test_ndarray_parity_constructors_and_into_vec() {
    let zeros = Array::<i32, VecStorage<i32>, 2>::zeros([2, 2]);
    assert_eq!(zeros.storage().as_slice(), &[0, 0, 0, 0]);

    let filled = Array::<i32, VecStorage<i32>, 2>::from_elem([2, 2], 7);
    assert_eq!(filled.storage().as_slice(), &[7, 7, 7, 7]);

    let generated = Array::<i32, VecStorage<i32>, 2>::from_shape_fn([2, 3], |[row, col]| {
        (row as i32) * 10 + col as i32
    });
    assert_eq!(generated.storage().as_slice(), &[0, 1, 2, 10, 11, 12]);

    let from_vec =
        Array::<i32, VecStorage<i32>, 2>::from_shape_vec([2, 2], vec![1, 2, 3, 4]).unwrap();
    assert_eq!(from_vec.into_vec(), vec![1, 2, 3, 4]);

    assert!(matches!(
        Array::<i32, VecStorage<i32>, 2>::from_vec([2, 2], vec![1, 2, 3]),
        Err(LetoError::StorageError { .. })
    ));
}

#[test]
fn test_axis_iter_yields_read_only_subviews() {
    let array =
        Array::<i32, VecStorage<i32>, 2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let rows: Vec<Vec<i32>> = AxisIter::new(&array.view(), 0, RankMarker::<2>)
        .unwrap()
        .map(|row| {
            (0..row.shape()[0])
                .map(|col| *row.get([col]).unwrap())
                .collect()
        })
        .collect();

    assert_eq!(rows, vec![vec![1, 2, 3], vec![4, 5, 6]]);
}

#[test]
fn test_view_axis_iter_methods_yield_subviews() {
    let mut array =
        Array::<i32, VecStorage<i32>, 2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();

    let columns: Vec<Vec<i32>> = array
        .view()
        .axis_iter::<1>(1)
        .unwrap()
        .map(|column| {
            (0..column.shape()[0])
                .map(|row| *column.get([row]).unwrap())
                .collect()
        })
        .collect();
    assert_eq!(columns, vec![vec![1, 4], vec![2, 5], vec![3, 6]]);

    for mut row in array.view_mut().axis_iter_mut::<1>(0).unwrap() {
        *row.get_mut([0]).unwrap() *= 10;
    }
    assert_eq!(array.storage().as_slice(), &[10, 2, 3, 40, 5, 6]);
}

#[test]
fn test_slicing() {
    let layout = Layout::c_contiguous([4, 4]).unwrap();
    let storage = VecStorage::new((0..16).collect());
    let array = Array::new(layout, storage).unwrap();

    // Slice axis 0: from index 1 to 3, step 1. Slice axis 1: from index 1 to 4, step 2.
    let sliced_view = array.slice(&[(1, 3, 1), (1, 4, 2)]).unwrap();
    assert_eq!(sliced_view.shape(), [2, 2]);
    assert_eq!(sliced_view.strides(), [4, 2]);
    assert_eq!(sliced_view.offset(), 5);

    // Check physical elements:
    // Original array:
    //  0  1  2  3
    //  4  5  6  7
    //  8  9 10 11
    // 12 13 14 15
    // Sliced view shape [2, 2] starting at offset 5 (element 5).
    // Row 0: index [0, 0] -> physical offset 5 (val 5), index [0, 1] -> physical offset 7 (val 7)
    // Row 1: index [1, 0] -> physical offset 9 (val 9), index [1, 1] -> physical offset 11 (val 11)
    assert_eq!(*sliced_view.get([0, 0]).unwrap(), 5);
    assert_eq!(*sliced_view.get([0, 1]).unwrap(), 7);
    assert_eq!(*sliced_view.get([1, 0]).unwrap(), 9);
    assert_eq!(*sliced_view.get([1, 1]).unwrap(), 11);
}

#[test]
fn test_ndarray_style_slice_with_negative_bounds_and_reverse_stride() {
    let layout = Layout::c_contiguous([5, 4]).unwrap();
    let storage = VecStorage::new((0..20).collect());
    let array = Array::new(layout, storage).unwrap();

    let view = array
        .slice_with::<2>(&[
            SliceArg::range(Some(-1), None, -2),
            SliceArg::range(Some(1), None, 2),
        ])
        .unwrap();

    assert_eq!(view.shape(), [3, 2]);
    assert_eq!(view.strides(), [-8, 2]);
    assert_eq!(view.offset(), 17);
    assert_eq!(*view.get([0, 0]).unwrap(), 17);
    assert_eq!(*view.get([0, 1]).unwrap(), 19);
    assert_eq!(*view.get([1, 0]).unwrap(), 9);
    assert_eq!(*view.get([2, 1]).unwrap(), 3);
}

#[test]
fn test_ndarray_style_slice_drops_indexed_axis_and_adds_new_axis() {
    let layout = Layout::c_contiguous([2, 3, 4]).unwrap();
    let storage = VecStorage::new((0..24).collect());
    let array = Array::new(layout, storage).unwrap();

    let view = array
        .slice_with::<2>(&[
            SliceArg::Index(-1),
            SliceArg::NewAxis,
            SliceArg::range(Some(1), None, 1),
            SliceArg::Index(2),
        ])
        .unwrap();

    assert_eq!(view.shape(), [1, 2]);
    assert_eq!(view.strides(), [0, 4]);
    assert_eq!(view.offset(), 18);
    assert_eq!(*view.get([0, 0]).unwrap(), 18);
    assert_eq!(*view.get([0, 1]).unwrap(), 22);
}

#[test]
fn test_ndarray_style_slice_ellipsis_and_implicit_trailing_axes() {
    let layout = Layout::c_contiguous([2, 3, 4]).unwrap();
    let storage = VecStorage::new((0..24).collect());
    let array = Array::new(layout, storage).unwrap();

    let ellipsis = array
        .slice_with::<3>(&[SliceArg::Ellipsis, SliceArg::Index(-1), SliceArg::NewAxis])
        .unwrap();
    assert_eq!(ellipsis.shape(), [2, 3, 1]);
    assert_eq!(ellipsis.strides(), [12, 4, 0]);
    assert_eq!(*ellipsis.get([1, 2, 0]).unwrap(), 23);

    let implicit = array
        .slice_with::<3>(&[SliceArg::range(Some(1), None, 1)])
        .unwrap();
    assert_eq!(implicit.shape(), [1, 3, 4]);
    assert_eq!(implicit.strides(), [12, 4, 1]);
    assert_eq!(*implicit.get([0, 2, 3]).unwrap(), 23);
}

#[test]
fn test_transpose() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let storage = VecStorage::new(vec![1, 2, 3, 4, 5, 6]);
    let array = Array::new(layout, storage).unwrap();

    let transposed = array.transpose([1, 0]).unwrap();
    assert_eq!(transposed.shape(), [3, 2]);
    assert_eq!(transposed.strides(), [1, 3]);

    // Transposed layout:
    // [0, 0] -> physical offset 0 (val 1)
    // [0, 1] -> physical offset 3 (val 4)
    // [1, 0] -> physical offset 1 (val 2)
    // [2, 1] -> physical offset 5 (val 6)
    assert_eq!(*transposed.get([0, 0]).unwrap(), 1);
    assert_eq!(*transposed.get([0, 1]).unwrap(), 4);
    assert_eq!(*transposed.get([1, 0]).unwrap(), 2);
    assert_eq!(*transposed.get([2, 1]).unwrap(), 6);
}

#[test]
fn test_broadcasting() {
    let layout = Layout::c_contiguous([1, 3]).unwrap();
    let storage = VecStorage::new(vec![100, 200, 300]);
    let array = Array::new(layout, storage).unwrap();

    let broadcasted = array.broadcast([2, 3]).unwrap();
    assert_eq!(broadcasted.shape(), [2, 3]);
    assert_eq!(broadcasted.strides(), [0, 1]); // Stride of axis 0 is 0

    assert_eq!(*broadcasted.get([0, 0]).unwrap(), 100);
    assert_eq!(*broadcasted.get([1, 0]).unwrap(), 100);
    assert_eq!(*broadcasted.get([0, 2]).unwrap(), 300);
    assert_eq!(*broadcasted.get([1, 2]).unwrap(), 300);
}

#[test]
fn test_mutable_broadcast_rejects_zero_stride_write_aliasing() {
    let layout = Layout::c_contiguous([1, 3]).unwrap();
    let storage = VecStorage::new(vec![100, 200, 300]);
    let mut array = Array::new(layout, storage).unwrap();

    let result = array.view_mut().broadcast_mut([2, 3]);
    assert!(matches!(
        result,
        Err(LetoError::IncompatibleBroadcast { .. })
    ));
}

#[test]
fn test_mutable_broadcast_permits_non_aliasing_same_shape() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let storage = VecStorage::new(vec![1, 2, 3, 4, 5, 6]);
    let mut array = Array::new(layout, storage).unwrap();

    let mut view = array.view_mut().broadcast_mut([2, 3]).unwrap();
    *view.get_mut([1, 2]).unwrap() = 60;

    assert_eq!(*view.get([1, 2]).unwrap(), 60);
}
