//! Sliding-window iteration: count theorem, content, strided, double-ended.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array2, LetoError};

fn leto(shape: [usize; 2], data: Vec<i32>) -> Array2<i32> {
    Array2::from_shape_vec(shape, data).unwrap()
}

/// Collect a window's elements in logical row-major order.
fn window_elems(w: &leto::ArrayView2<'_, i32>) -> Vec<i32> {
    w.iter().copied().collect()
}

#[test]
fn windows_count_matches_theorem() {
    // 3×4 array, 2×2 windows: (3−2+1)·(4−2+1) = 2·3 = 6 windows.
    let a = leto([3, 4], (1..=12).collect());
    let count = a.windows([2, 2]).unwrap().count();
    assert_eq!(count, 6);
    assert_eq!(a.windows([2, 2]).unwrap().len(), 6);
    // 1×N window over each axis.
    assert_eq!(a.windows([1, 4]).unwrap().len(), 3); // (3)·(1)
    assert_eq!(a.windows([3, 1]).unwrap().len(), 4); // (1)·(4)
                                                     // Full-size window: exactly one.
    assert_eq!(a.windows([3, 4]).unwrap().len(), 1);
}

#[test]
fn windows_content_row_major_order() {
    // [[1,2,3],[4,5,6]] with 2×2 windows → two windows:
    //   start (0,0): [[1,2],[4,5]]; start (0,1): [[2,3],[5,6]].
    let a = leto([2, 3], vec![1, 2, 3, 4, 5, 6]);
    let mut it = a.windows([2, 2]).unwrap();
    assert_eq!(window_elems(&it.next().unwrap()), vec![1, 2, 4, 5]);
    assert_eq!(window_elems(&it.next().unwrap()), vec![2, 3, 5, 6]);
    assert!(it.next().is_none());
}

#[test]
fn windows_full_window_equals_original() {
    let a = leto([2, 2], vec![1, 2, 3, 4]);
    let w = a.windows([2, 2]).unwrap().next().unwrap();
    assert_eq!(w.shape(), [2, 2]);
    assert_eq!(window_elems(&w), vec![1, 2, 3, 4]);
}

#[test]
fn windows_over_transposed_view_are_zero_copy_and_correct() {
    // Transposed view of [[1,2,3],[4,5,6]] is logical 3×2 [[1,4],[2,5],[3,6]].
    // 2×2 windows over it: (3−2+1)·(2−2+1) = 2 windows.
    let a = leto([2, 3], vec![1, 2, 3, 4, 5, 6]);
    let t = a.transpose([1, 0]).unwrap();
    let mut it = t.windows([2, 2]).unwrap();
    // start (0,0): rows 0,1 of transposed → [[1,4],[2,5]] = 1,4,2,5
    assert_eq!(window_elems(&it.next().unwrap()), vec![1, 4, 2, 5]);
    // start (1,0): rows 1,2 → [[2,5],[3,6]] = 2,5,3,6
    assert_eq!(window_elems(&it.next().unwrap()), vec![2, 5, 3, 6]);
    assert!(it.next().is_none());
}

#[test]
fn windows_double_ended() {
    let a = leto([1, 5], vec![10, 20, 30, 40, 50]);
    // 1×2 windows: starts 0..4 → 4 windows.
    let mut it = a.windows([1, 2]).unwrap();
    assert_eq!(window_elems(&it.next().unwrap()), vec![10, 20]); // front: start 0
    assert_eq!(window_elems(&it.next_back().unwrap()), vec![40, 50]); // back: start 3
    assert_eq!(window_elems(&it.next().unwrap()), vec![20, 30]); // start 1
    assert_eq!(window_elems(&it.next_back().unwrap()), vec![30, 40]); // start 2
    assert!(it.next().is_none());
    assert!(it.next_back().is_none());
}

#[test]
fn windows_rejects_zero_and_oversize_extents() {
    let a = leto([2, 3], vec![1, 2, 3, 4, 5, 6]);
    match a.windows([0, 2]) {
        Err(LetoError::StorageError { reason }) => {
            assert_eq!(reason, "window extent on axis 0 must be non-zero");
        }
        Err(other) => panic!("expected zero-extent error, got {other:?}"),
        Ok(_) => panic!("expected zero-extent error, got Ok"),
    }
    match a.windows([2, 4]) {
        Err(LetoError::StorageError { reason }) => {
            assert_eq!(reason, "window extent 4 on axis 1 exceeds array extent 3");
        }
        Err(other) => panic!("expected oversize error, got {other:?}"),
        Ok(_) => panic!("expected oversize error, got Ok"),
    }
}
