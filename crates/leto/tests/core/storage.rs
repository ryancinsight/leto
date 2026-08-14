#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{
    domain::{RankMarker, RemoveAxis},
    Array, CowStorage, Layout, LetoError, Storage, VecStorage,
};

#[cfg(feature = "mnemosyne-alloc")]
use leto::MnemosyneStorage;

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
fn test_cow_storage_borrows_without_copying_on_read() {
    let backing = [10, 11, 12, 13];
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let array = Array::new(layout, CowStorage::borrowed(&backing)).unwrap();

    assert!(array.storage().is_borrowed());
    assert!(std::ptr::eq(
        array.storage().as_borrowed().unwrap().as_ptr(),
        backing.as_ptr()
    ));
    assert!(array.storage().as_owned().is_none());
    assert!(std::ptr::eq(
        array.storage().as_slice().as_ptr(),
        backing.as_ptr()
    ));
    assert_eq!(*array.get([1, 1]).unwrap(), 13);
}

#[test]
fn test_cow_storage_detaches_on_mutation() {
    let backing = [1, 2, 3, 4];
    let layout = Layout::c_contiguous([2, 2]).unwrap();
    let mut array = Array::new(layout, CowStorage::borrowed(&backing)).unwrap();

    *array.get_mut([0, 1]).unwrap() = 20;

    assert!(array.storage().is_owned());
    assert!(array.storage().as_borrowed().is_none());
    assert_eq!(array.storage().as_owned().unwrap(), &[1, 20, 3, 4]);
    assert_ne!(array.storage().as_slice().as_ptr(), backing.as_ptr());
    assert_eq!(backing, [1, 2, 3, 4]);
    assert_eq!(array.storage().as_slice(), &[1, 20, 3, 4]);
}

#[test]
fn test_leto_parity_constructors_and_into_vec() {
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

#[cfg(feature = "mnemosyne-alloc")]
#[test]
fn test_mnemosyne_array_constructor_parity_and_into_vec() {
    let zeros = Array::<i32, MnemosyneStorage<i32>, 2>::zeros_mnemosyne([2, 2]);
    assert_eq!(zeros.storage().as_slice(), &[0, 0, 0, 0]);

    let generated =
        Array::<i32, MnemosyneStorage<i32>, 2>::from_mnemosyne_shape_fn([2, 3], |[row, col]| {
            (row as i32) * 10 + col as i32
        });
    assert_eq!(generated.storage().as_slice(), &[0, 1, 2, 10, 11, 12]);

    let from_vec =
        Array::<i32, MnemosyneStorage<i32>, 2>::from_mnemosyne_shape_vec([2, 2], vec![1, 2, 3, 4])
            .unwrap();
    assert_eq!(from_vec.into_vec(), vec![1, 2, 3, 4]);

    assert!(matches!(
        Array::<i32, MnemosyneStorage<i32>, 2>::from_mnemosyne_vec([2, 2], vec![1, 2, 3]),
        Err(LetoError::StorageError { .. })
    ));
}

#[cfg(feature = "mnemosyne-alloc")]
#[test]
fn test_mnemosyne_storage_moves_non_clone_values() {
    #[derive(Debug, PartialEq, Eq)]
    struct Payload(usize);

    let storage = MnemosyneStorage::from_vec(vec![Payload(1), Payload(2), Payload(3)]);
    assert_eq!(storage.as_slice(), &[Payload(1), Payload(2), Payload(3)]);

    let values = storage.into_vec();
    assert_eq!(values, vec![Payload(1), Payload(2), Payload(3)]);
}

#[cfg(feature = "mnemosyne-alloc")]
#[test]
fn test_mnemosyne_storage_zst_drop() {
    static DROP_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    struct ZstDrop;
    impl Drop for ZstDrop {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    DROP_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);
    {
        let vec = vec![ZstDrop, ZstDrop, ZstDrop];
        let _storage = MnemosyneStorage::from_vec(vec);
    }
    assert_eq!(DROP_COUNT.load(std::sync::atomic::Ordering::SeqCst), 3);
}

#[test]
fn test_slice_step_isize_min() {
    let layout = Layout::c_contiguous([5]).unwrap();
    let ranges = [(0, 5, isize::MIN)];
    let res = layout.slice(&ranges);
    assert!(res.is_err());
}
