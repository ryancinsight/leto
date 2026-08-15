#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Layout, LetoError};

#[test]
fn test_c_contiguous_layout() {
    let layout = Layout::c_contiguous([2, 3, 4]).unwrap();
    assert_eq!(layout.shape(), [2, 3, 4]);
    assert_eq!(layout.strides(), [12, 4, 1]);
    assert_eq!(layout.offset(), 0);
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
    assert_eq!(layout.shape(), [2, 3, 4]);
    assert_eq!(layout.strides(), [1, 2, 6]);
    assert_eq!(layout.offset(), 0);
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
fn test_negative_stride_offsets_are_checked_before_unsigned_conversion() {
    let layout = Layout::try_new([3], [-1], 2).unwrap();

    assert_eq!(layout.checked_min_max_offsets().unwrap(), (0, 2));
    assert_eq!(layout.offset_of([0]).unwrap(), 2);
    assert_eq!(layout.offset_of([2]).unwrap(), 0);

    // A layout walking [0..3) at stride -1 from base 0 addresses physical
    // offset -2. `Layout::new` used to build it and leave detection to the
    // downstream `checked_min_max_offsets`/`offset_of` calls; `try_new` now
    // refuses it outright, so the failure moves from the accessors to the
    // constructor. The stronger contract is asserted here.
    assert!(matches!(
        Layout::try_new([3], [-1], 0),
        Err(LetoError::StorageError { .. })
    ));
}

#[test]
fn test_array_rejects_layout_that_reaches_one_past_storage() {
    use leto::{Array, VecStorage};
    let layout = Layout::try_new([2], [1], 1).unwrap();
    let storage = VecStorage::new(vec![10, 20]);

    assert!(matches!(
        Array::new(layout, storage),
        Err(LetoError::StorageError { .. })
    ));
}
