#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{concat, pad, split, stack, Array, Layout, Storage, VecStorage};

fn array2(rows: usize, cols: usize, data: Vec<i32>) -> Array<i32, VecStorage<i32>, 2> {
    Array::new(
        Layout::c_contiguous([rows, cols]).unwrap(),
        VecStorage::new(data),
    )
    .unwrap()
}

#[test]
fn test_concat_axis0() {
    let a = array2(1, 3, vec![1, 2, 3]);
    let b = array2(2, 3, vec![4, 5, 6, 7, 8, 9]);
    let out = concat(&[a.view(), b.view()], 0).unwrap();
    assert_eq!(out.shape(), [3, 3]);
    assert_eq!(out.storage().as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn test_concat_axis1() {
    let a = array2(2, 1, vec![1, 4]);
    let b = array2(2, 2, vec![2, 3, 5, 6]);
    let out = concat(&[a.view(), b.view()], 1).unwrap();
    assert_eq!(out.shape(), [2, 3]);
    assert_eq!(out.storage().as_slice(), &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_concat_rejects_mismatched_offaxis() {
    let a = array2(2, 3, vec![0; 6]);
    let b = array2(2, 2, vec![0; 4]);
    assert!(concat(&[a.view(), b.view()], 0).is_err());
}

#[test]
fn test_concat_of_transposed_view_uses_logical_order() {
    // Transposed input: concat must read logical (row-major) order, not memory.
    let a = array2(2, 3, vec![1, 2, 3, 4, 5, 6]);
    let at = a.transpose([1, 0]).unwrap(); // shape [3,2] logical [[1,4],[2,5],[3,6]]
    let b = array2(3, 2, vec![10, 11, 12, 13, 14, 15]);
    let out = concat(&[at, b.view()], 1).unwrap();
    assert_eq!(out.shape(), [3, 4]);
    assert_eq!(
        out.storage().as_slice(),
        &[1, 4, 10, 11, 2, 5, 12, 13, 3, 6, 14, 15]
    );
}

#[test]
fn test_pad_constant() {
    let a = array2(1, 2, vec![1, 2]);
    let out = pad(&a.view(), [(1, 1), (2, 0)], 0).unwrap();
    // out shape: rows 1+1+1=3, cols 2+2+0=4
    assert_eq!(out.shape(), [3, 4]);
    assert_eq!(
        out.storage().as_slice(),
        &[0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0]
    );
}

#[test]
fn test_split_axis0() {
    let a = array2(4, 2, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    let parts = split(&a.view(), 0, &[1, 3]).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].shape(), [1, 2]);
    assert_eq!(parts[1].shape(), [3, 2]);
    assert_eq!(parts[0].as_slice().unwrap(), &[0, 1]);
    assert_eq!(parts[1].as_slice().unwrap(), &[2, 3, 4, 5, 6, 7]);
}

#[test]
fn test_split_rejects_bad_sizes() {
    let a = array2(4, 2, vec![0; 8]);
    assert!(split(&a.view(), 0, &[1, 2]).is_err());
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
    assert!(res.is_err());

    std::mem::drop(a);

    assert_eq!(drop_count.load(Ordering::SeqCst), 7);
}

#[test]
fn test_concat_panic_cleanup() {
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

    let layout = Layout::c_contiguous([1, 3]).unwrap();
    let storage = VecStorage::new(vec![item1, item2, item3]);
    let a = Array::new(layout, storage).unwrap();

    let inputs = &[a.view(), a.view()];
    let res = std::panic::catch_unwind(|| {
        let _ = concat(inputs, 1);
    });
    assert!(res.is_err());

    std::mem::drop(a);

    assert_eq!(drop_count.load(Ordering::SeqCst), 5);
}
