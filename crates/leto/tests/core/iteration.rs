//! Element / indexed iteration: logical-order, strided, and double-ended.

use leto::{Array2, ElementIter};

fn leto(shape: [usize; 2], data: Vec<i32>) -> Array2<i32> {
    Array2::from_shape_vec(shape, data).unwrap()
}

#[test]
fn iter_yields_row_major_order() {
    let a = leto([2, 3], vec![1, 2, 3, 4, 5, 6]);
    let collected: Vec<i32> = a.iter().copied().collect();
    assert_eq!(collected, vec![1, 2, 3, 4, 5, 6]);
    // ExactSizeIterator length matches logical size.
    assert_eq!(a.iter().len(), 6);
}

#[test]
fn indexed_iter_pairs_index_and_value() {
    let a = leto([2, 2], vec![10, 20, 30, 40]);
    let pairs: Vec<([usize; 2], i32)> = a.indexed_iter().map(|(i, &v)| (i, v)).collect();
    assert_eq!(
        pairs,
        vec![([0, 0], 10), ([0, 1], 20), ([1, 0], 30), ([1, 1], 40),]
    );
}

#[test]
fn iter_respects_transposed_logical_order() {
    // Row-major [[1,2,3],[4,5,6]]; transposed view is 3×2 logical order
    // [[1,4],[2,5],[3,6]] → flat 1,4,2,5,3,6 even though storage is unchanged.
    let a = leto([2, 3], vec![1, 2, 3, 4, 5, 6]);
    let t = a.transpose([1, 0]).unwrap();
    let collected: Vec<i32> = t.iter().copied().collect();
    assert_eq!(collected, vec![1, 4, 2, 5, 3, 6]);

    // indexed_iter over the transposed view reports transposed indices.
    let indexed: Vec<([usize; 2], i32)> = t.indexed_iter().map(|(i, &v)| (i, v)).collect();
    assert_eq!(indexed[0], ([0, 0], 1));
    assert_eq!(indexed[1], ([0, 1], 4));
    assert_eq!(indexed[2], ([1, 0], 2));
}

#[test]
fn iter_double_ended_meets_once() {
    let a = leto([1, 5], vec![1, 2, 3, 4, 5]);
    let mut it: ElementIter<'_, i32, 2> = a.iter();
    assert_eq!(it.next().copied(), Some(1));
    assert_eq!(it.next_back().copied(), Some(5));
    assert_eq!(it.next().copied(), Some(2));
    assert_eq!(it.next_back().copied(), Some(4));
    assert_eq!(it.next().copied(), Some(3));
    // Front and back cursors have met: both ends exhausted.
    assert_eq!(it.next(), None);
    assert_eq!(it.next_back(), None);
}

#[test]
fn iter_double_ended_rev_equals_reverse() {
    let a = leto([2, 3], vec![1, 2, 3, 4, 5, 6]);
    let forward: Vec<i32> = a.iter().copied().collect();
    let mut reversed: Vec<i32> = a.iter().rev().copied().collect();
    reversed.reverse();
    assert_eq!(forward, reversed);
}

#[test]
fn into_iter_for_reference_view() {
    let a = leto([2, 2], vec![7, 8, 9, 10]);
    let view = a.view();
    let mut sum = 0;
    for &x in &view {
        sum += x;
    }
    assert_eq!(sum, 34);
}

#[test]
fn iter_empty_array_yields_nothing() {
    let a: Array2<i32> = Array2::from_shape_vec([0, 0], vec![]).unwrap();
    assert_eq!(a.iter().count(), 0);
    assert_eq!(a.indexed_iter().count(), 0);
}
