use leto::{Array, Layout, VecStorage, LetoError};

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
fn test_array_creation_and_indexing() {
    let layout = Layout::c_contiguous([2, 3]).unwrap();
    let storage = VecStorage::new(vec![10, 11, 12, 20, 21, 22]);
    let array = Array::new(layout, storage).unwrap();

    assert_eq!(*array.get([0, 0]).unwrap(), 10);
    assert_eq!(*array.get([0, 2]).unwrap(), 12);
    assert_eq!(*array.get([1, 1]).unwrap(), 21);
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
