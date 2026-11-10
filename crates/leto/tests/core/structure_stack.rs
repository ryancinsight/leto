#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{stack, stack_into, Array, Layout, Storage, VecStorage};

fn array2(rows: usize, cols: usize, data: Vec<i32>) -> Array<i32, VecStorage<i32>, 2> {
    Array::new(
        Layout::c_contiguous([rows, cols]).unwrap(),
        VecStorage::new(data),
    )
    .unwrap()
}

fn array1(data: Vec<i32>) -> Array<i32, VecStorage<i32>, 1> {
    Array::new(
        Layout::c_contiguous([data.len()]).unwrap(),
        VecStorage::new(data),
    )
    .unwrap()
}

#[test]
fn test_stack_new_leading_axis() {
    let a = array1(vec![1, 2, 3]);
    let b = array1(vec![4, 5, 6]);
    let out = stack::<i32, 1, 2>(&[a.view(), b.view()], 0).unwrap();
    assert_eq!(out.shape(), [2, 3]);
    assert_eq!(out.storage().as_slice(), &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_stack_new_trailing_axis() {
    let a = array1(vec![1, 2, 3]);
    let b = array1(vec![4, 5, 6]);
    // Inserting the new axis last interleaves the inputs column-wise.
    let out = stack::<i32, 1, 2>(&[a.view(), b.view()], 1).unwrap();
    assert_eq!(out.shape(), [3, 2]);
    assert_eq!(out.storage().as_slice(), &[1, 4, 2, 5, 3, 6]);
}

#[test]
fn test_stack_rank2_into_rank3() {
    let a = array2(2, 2, vec![1, 2, 3, 4]);
    let b = array2(2, 2, vec![5, 6, 7, 8]);
    let out = stack::<i32, 2, 3>(&[a.view(), b.view()], 0).unwrap();
    assert_eq!(out.shape(), [2, 2, 2]);
    assert_eq!(out.storage().as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn test_stack_rejects_shape_mismatch() {
    let a = array1(vec![1, 2, 3]);
    let b = array1(vec![4, 5]);
    assert!(stack::<i32, 1, 2>(&[a.view(), b.view()], 0).is_err());
}

#[test]
fn test_stack_of_transposed_views_uses_logical_order() {
    let src = array2(2, 3, vec![1, 2, 3, 4, 5, 6]);
    let t = src.transpose([1, 0]).unwrap(); // logical [[1,4],[2,5],[3,6]] shape [3,2]
    let z = array2(3, 2, vec![0, 0, 0, 0, 0, 0]);
    let out = stack::<i32, 2, 3>(&[t, z.view()], 0).unwrap();
    assert_eq!(out.shape(), [2, 3, 2]);
    assert_eq!(
        out.storage().as_slice(),
        &[1, 4, 2, 5, 3, 6, 0, 0, 0, 0, 0, 0]
    );
}

#[test]
fn test_stack_panic_cleanup() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Debug)]
    struct PanicClone {
        _id: usize,
        drop_count: Arc<AtomicUsize>,
        panic_on_clone: bool,
    }

    impl Clone for PanicClone {
        fn clone(&self) -> Self {
            if self.panic_on_clone {
                panic!("clone panic!");
            }
            Self {
                _id: self._id,
                drop_count: self.drop_count.clone(),
                panic_on_clone: self.panic_on_clone,
            }
        }
    }

    impl Drop for PanicClone {
        fn drop(&mut self) {
            self.drop_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    let drop_count = Arc::new(AtomicUsize::new(0));

    let item1 = PanicClone {
        _id: 1,
        drop_count: drop_count.clone(),
        panic_on_clone: false,
    };
    let item2 = PanicClone {
        _id: 2,
        drop_count: drop_count.clone(),
        panic_on_clone: false,
    };
    let item3 = PanicClone {
        _id: 3,
        drop_count: drop_count.clone(),
        panic_on_clone: true,
    };

    let shape = [3];
    let a_layout = Layout::c_contiguous(shape).unwrap();
    let a_storage = VecStorage::new(vec![item1, item2, item3]);
    let a = Array::new(a_layout, a_storage).unwrap();

    let inputs = &[a.view(), a.view()];
    let res = std::panic::catch_unwind(|| {
        let _ = stack::<PanicClone, 1, 2>(inputs, 1);
    });
    let payload = res.expect_err("a panicking clone must unwind to the caller");
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"clone panic!"));

    std::mem::drop(a);

    assert_eq!(drop_count.load(Ordering::SeqCst), 7);
}

#[test]
fn test_stack_into_leading_axis_matches_allocating_stack() {
    let a = array1(vec![1, 2, 3]);
    let b = array1(vec![4, 5, 6]);
    let expected = stack::<i32, 1, 2>(&[a.view(), b.view()], 0).unwrap();
    let mut out = Array::new(
        Layout::c_contiguous([2, 3]).unwrap(),
        VecStorage::fill(6, -1),
    )
    .unwrap();
    stack_into::<i32, 1, 2>(&[a.view(), b.view()], 0, &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice(), expected.storage().as_slice());
}

#[test]
fn test_stack_into_trailing_axis_interleaves_inputs() {
    let a = array1(vec![1, 2, 3]);
    let b = array1(vec![4, 5, 6]);
    let mut out = Array::new(
        Layout::c_contiguous([3, 2]).unwrap(),
        VecStorage::fill(6, -1),
    )
    .unwrap();
    stack_into::<i32, 1, 2>(&[a.view(), b.view()], 1, &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice(), &[1, 4, 2, 5, 3, 6]);
}

#[test]
fn test_stack_into_rank2_into_rank3() {
    let a = array2(2, 2, vec![1, 2, 3, 4]);
    let b = array2(2, 2, vec![5, 6, 7, 8]);
    let mut out = Array::new(
        Layout::c_contiguous([2, 2, 2]).unwrap(),
        VecStorage::fill(8, -1),
    )
    .unwrap();
    stack_into::<i32, 2, 3>(&[a.view(), b.view()], 0, &mut out.view_mut()).unwrap();
    assert_eq!(out.storage().as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn test_stack_into_transposed_views_uses_logical_order() {
    let src = array2(2, 3, vec![1, 2, 3, 4, 5, 6]);
    let t = src.transpose([1, 0]).unwrap(); // logical [[1,4],[2,5],[3,6]] shape [3,2]
    let z = array2(3, 2, vec![0; 6]);
    let mut out = Array::new(
        Layout::c_contiguous([2, 3, 2]).unwrap(),
        VecStorage::fill(12, -1),
    )
    .unwrap();
    stack_into::<i32, 2, 3>(&[t, z.view()], 0, &mut out.view_mut()).unwrap();
    assert_eq!(
        out.storage().as_slice(),
        &[1, 4, 2, 5, 3, 6, 0, 0, 0, 0, 0, 0]
    );
}

#[test]
fn test_stack_into_rejects_shape_mismatch_output_shape_and_empty_inputs() {
    let a = array1(vec![1, 2, 3]);
    let short = array1(vec![4, 5]);
    let mut out = Array::new(
        Layout::c_contiguous([2, 3]).unwrap(),
        VecStorage::fill(6, 0),
    )
    .unwrap();
    let err = stack_into::<i32, 1, 2>(&[a.view(), short.view()], 0, &mut out.view_mut())
        .expect_err("inputs disagree");
    assert!(
        matches!(err, leto::LetoError::ShapeMismatch { .. }),
        "unexpected error: {err:?}"
    );

    let mut narrow = Array::new(
        Layout::c_contiguous([1, 3]).unwrap(),
        VecStorage::fill(3, 0),
    )
    .unwrap();
    let err = stack_into::<i32, 1, 2>(&[a.view(), a.view()], 0, &mut narrow.view_mut())
        .expect_err("output misses the stacked shape");
    assert!(
        matches!(err, leto::LetoError::ShapeMismatch { .. }),
        "unexpected error: {err:?}"
    );

    let err = stack_into::<i32, 1, 2>(&[], 0, &mut out.view_mut()).expect_err("no inputs");
    assert!(
        matches!(err, leto::LetoError::StorageError { .. }),
        "unexpected error: {err:?}"
    );
}
