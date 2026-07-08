//! Element / indexed iteration: logical-order, strided, and double-ended.

use leto::{Array, Array2, ElementIter, Layout, LetoError, VecStorage};

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
fn indexed_iter_mut_updates_contiguous_values_by_index() {
    let mut a = leto([2, 3], vec![1, 2, 3, 4, 5, 6]);

    for (index, value) in a.indexed_iter_mut().unwrap() {
        *value += index[0] as i32 * 10 + index[1] as i32;
    }

    assert_eq!(
        a.iter().copied().collect::<Vec<_>>(),
        vec![1, 3, 5, 14, 16, 18]
    );
}

#[test]
fn indexed_iter_mut_respects_transposed_logical_indices() {
    let mut a = leto([2, 3], vec![1, 2, 3, 4, 5, 6]);

    for (index, value) in a
        .view_mut()
        .transpose_mut([1, 0])
        .unwrap()
        .indexed_iter_mut()
        .unwrap()
    {
        *value = index[0] as i32 * 10 + index[1] as i32;
    }

    assert_eq!(
        a.iter().copied().collect::<Vec<_>>(),
        vec![0, 10, 20, 1, 11, 21]
    );
}

#[test]
fn indexed_iter_mut_double_ended_meets_once() {
    let mut a = leto([2, 3], vec![1, 2, 3, 4, 5, 6]);

    {
        let mut iter = a.indexed_iter_mut().unwrap();
        let (front_index, front) = iter.next().unwrap();
        assert_eq!(front_index, [0, 0]);
        *front = 10;

        let (back_index, back) = iter.next_back().unwrap();
        assert_eq!(back_index, [1, 2]);
        *back = 60;

        assert_eq!(iter.len(), 4);
    }

    assert_eq!(
        a.iter().copied().collect::<Vec<_>>(),
        vec![10, 2, 3, 4, 5, 60]
    );
}

#[test]
fn indexed_iter_mut_rejects_aliasing_layout() {
    let layout = Layout::new([2, 2], [1, 1], 0);
    let storage = VecStorage::new(vec![0, 1, 2]);
    let mut a = Array::<i32, VecStorage<i32>, 2>::new(layout, storage).unwrap();

    let err = match a.indexed_iter_mut() {
        Ok(_) => panic!("aliasing layout must not yield mutable references"),
        Err(err) => err,
    };
    assert_eq!(
        err,
        LetoError::StorageError {
            reason: "indexed_iter_mut requires provably disjoint logical offsets".to_string()
        }
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
