#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array, Layout, LetoError, SliceArg, Storage, VecStorage};

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

#[test]
fn test_as_slice_exposes_offset_contiguous_subview() {
    // A row sliced out of a C-contiguous matrix is dense at a non-zero offset.
    let layout = Layout::c_contiguous([3, 4]).unwrap();
    let storage = VecStorage::new((0i32..12).collect::<Vec<_>>());
    let array = Array::new(layout, storage).unwrap();

    // Select row index 1 -> shape [4] starting at physical offset 4.
    let row = array
        .view()
        .slice_with::<1>(&[SliceArg::Index(1), SliceArg::All])
        .unwrap();
    assert_eq!(row.offset(), 4);
    assert!(!row.is_c_contiguous());
    assert!(row.is_contiguous());
    // Canonical as_slice now exposes the dense block independent of offset.
    assert_eq!(row.as_slice().unwrap(), &[4, 5, 6, 7]);
    assert_eq!(row.as_slice_memory_order().unwrap(), &[4, 5, 6, 7]);
}

#[test]
fn test_as_slice_memory_order_accepts_fortran_block() {
    // A pure transpose of a contiguous matrix is F-dense but not C-dense.
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let storage = VecStorage::new(vec![1, 2, 3, 4, 5, 6]);
    let array = Array::new(layout, storage).unwrap();
    let transposed = array.transpose([1, 0]).unwrap();

    assert!(transposed.as_slice().is_none());
    assert!(transposed.is_f_contiguous());
    assert_eq!(
        transposed.as_slice_memory_order().unwrap(),
        &[1, 2, 3, 4, 5, 6]
    );
}

#[test]
fn test_as_slice_rejects_strided_gap() {
    // Every other column is not a dense block in any memory order.
    let layout = Layout::c_contiguous([2, 4]).unwrap();
    let storage = VecStorage::new((0i32..8).collect::<Vec<_>>());
    let array = Array::new(layout, storage).unwrap();
    let strided = array
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(Some(0), Some(4), 2)])
        .unwrap();

    assert!(!strided.is_contiguous());
    assert!(strided.as_slice().is_none());
    assert!(strided.as_slice_memory_order().is_none());
}

#[test]
fn test_as_mut_slice_memory_order_writes_offset_block() {
    let layout = Layout::c_contiguous([3, 2]).unwrap();
    let storage = VecStorage::new((0i32..6).collect::<Vec<_>>());
    let mut array = Array::new(layout, storage).unwrap();

    {
        let mut row = array
            .view_mut()
            .slice_with_mut::<1>(&[SliceArg::Index(2), SliceArg::All])
            .unwrap();
        let block = row.as_mut_slice_memory_order().unwrap();
        assert_eq!(block, &[4, 5]);
        block[0] = 40;
        block[1] = 50;
    }

    assert_eq!(array.storage().as_slice(), &[0, 1, 2, 3, 40, 50]);
}

#[test]
fn test_reshape_reinterprets_dense_row_major_layout_without_copying() {
    let array = Array::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();

    let reshaped = array.reshape([3, 2]).unwrap();

    assert_eq!(reshaped.shape(), [3, 2]);
    assert_eq!(reshaped.strides(), [2, 1]);
    assert_eq!(*reshaped.get([0, 0]).unwrap(), 1);
    assert_eq!(*reshaped.get([1, 1]).unwrap(), 4);
    assert_eq!(*reshaped.get([2, 1]).unwrap(), 6);
    assert_eq!(reshaped.as_slice().unwrap(), &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_into_shape_preserves_owned_storage_and_new_rank() {
    let array = Array::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();

    let reshaped = array.into_shape([6]).unwrap();

    assert_eq!(reshaped.shape(), [6]);
    assert_eq!(reshaped.strides(), [1]);
    assert_eq!(reshaped.storage().as_slice(), &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_reshape_rejects_shape_size_mismatch_and_strided_layouts() {
    let array = Array::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let wrong_size = array.reshape([4, 2]);
    assert!(matches!(wrong_size, Err(LetoError::ShapeMismatch { .. })));

    let transposed = array.transpose([1, 0]).unwrap();
    let strided_reshape = transposed.reshape([6]);
    assert!(matches!(
        strided_reshape,
        Err(LetoError::StorageError { .. })
    ));
}

#[test]
fn test_reshape_mut_writes_through_dense_row_major_view() {
    let mut array = Array::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();

    {
        let mut reshaped = array.reshape_mut([3, 2]).unwrap();
        *reshaped.get_mut([2, 0]).unwrap() = 50;
    }

    assert_eq!(array.storage().as_slice(), &[1, 2, 3, 4, 50, 6]);
}

#[test]
fn test_permute_is_named_transpose_alias() {
    let array = Array::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();

    let permuted = array.permute([1, 0]).unwrap();
    let transposed = array.transpose([1, 0]).unwrap();

    assert_eq!(permuted.shape(), transposed.shape());
    assert_eq!(permuted.strides(), transposed.strides());
    assert_eq!(*permuted.get([2, 1]).unwrap(), 6);
}

#[test]
fn test_to_contiguous_materializes_logical_row_major_order() {
    let array = Array::from_shape_vec([3, 4], (0i32..12).collect::<Vec<_>>()).unwrap();
    let strided = array
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(Some(0), Some(4), 2)])
        .unwrap();

    let contiguous = strided.to_contiguous();

    assert_eq!(contiguous.shape(), [3, 2]);
    assert_eq!(contiguous.strides(), [2, 1]);
    assert_eq!(contiguous.storage().as_slice(), &[0, 2, 4, 6, 8, 10]);
}

#[test]
fn test_to_contiguous_materializes_broadcasted_views() {
    let row = Array::from_shape_vec([1, 3], vec![10, 20, 30]).unwrap();
    let broadcast = row.broadcast([2, 3]).unwrap();

    let contiguous = broadcast.to_contiguous();

    assert_eq!(contiguous.shape(), [2, 3]);
    assert_eq!(contiguous.storage().as_slice(), &[10, 20, 30, 10, 20, 30]);
}
