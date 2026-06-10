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
