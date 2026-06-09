use leto::{Array, Layout, LetoError, VecStorage};

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
