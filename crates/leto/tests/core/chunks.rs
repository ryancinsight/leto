//! Chunk iteration: non-overlapping zero-copy streaming blocks.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array2, LetoError};

fn leto(shape: [usize; 2], data: Vec<i32>) -> Array2<i32> {
    Array2::from_shape_vec(shape, data).unwrap()
}

fn view_elems(view: &leto::ArrayView2<'_, i32>) -> Vec<i32> {
    view.iter().copied().collect()
}

#[test]
fn exact_chunks_count_matches_floor_product_and_skips_remainders() {
    let a = leto([5, 6], (1..=30).collect());
    let chunks: Vec<_> = a.exact_chunks([2, 4]).unwrap().collect();

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].shape(), [2, 4]);
    assert_eq!(view_elems(&chunks[0]), vec![1, 2, 3, 4, 7, 8, 9, 10]);
    assert_eq!(view_elems(&chunks[1]), vec![13, 14, 15, 16, 19, 20, 21, 22]);
}

#[test]
fn exact_chunks_over_transposed_view_preserve_strides_and_values() {
    let a = leto([2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let transposed = a.transpose([1, 0]).unwrap();
    let mut chunks = transposed.exact_chunks([2, 2]).unwrap();

    let first = chunks.next().unwrap();
    assert_eq!(first.shape(), [2, 2]);
    assert_eq!(first.strides(), transposed.strides());
    assert_eq!(first.offset(), transposed.offset());
    assert_eq!(view_elems(&first), vec![1, 5, 2, 6]);

    let second = chunks.next().unwrap();
    assert_eq!(second.shape(), [2, 2]);
    assert_eq!(second.strides(), transposed.strides());
    assert_eq!(second.offset(), 2);
    assert_eq!(view_elems(&second), vec![3, 7, 4, 8]);

    assert!(chunks.next().is_none());
}

#[test]
fn exact_chunks_double_ended_meets_once() {
    let a = leto([4, 4], (1..=16).collect());
    let mut chunks = a.exact_chunks([2, 2]).unwrap();

    assert_eq!(view_elems(&chunks.next().unwrap()), vec![1, 2, 5, 6]);
    assert_eq!(
        view_elems(&chunks.next_back().unwrap()),
        vec![11, 12, 15, 16]
    );
    assert_eq!(view_elems(&chunks.next().unwrap()), vec![3, 4, 7, 8]);
    assert_eq!(
        view_elems(&chunks.next_back().unwrap()),
        vec![9, 10, 13, 14]
    );
    assert!(chunks.next().is_none());
    assert!(chunks.next_back().is_none());
}

#[test]
fn exact_chunks_oversize_extent_yields_empty_stream() {
    let a = leto([2, 3], vec![1, 2, 3, 4, 5, 6]);

    let chunks = a.exact_chunks([3, 1]).unwrap();

    assert_eq!(chunks.len(), 0);
}

#[test]
fn exact_chunks_rejects_zero_extent() {
    let a = leto([2, 3], vec![1, 2, 3, 4, 5, 6]);

    match a.exact_chunks([1, 0]) {
        Err(LetoError::StorageError { reason }) => {
            assert_eq!(reason, "exact chunk extent on axis 1 must be non-zero");
        }
        Err(other) => panic!("expected zero-extent error, got {other:?}"),
        Ok(_) => panic!("expected zero-extent error, got Ok"),
    }
}

#[test]
fn axis_chunks_iter_includes_remainder() {
    let a = leto([2, 5], (1..=10).collect());
    let chunks: Vec<_> = a.axis_chunks_iter(1, 2).unwrap().collect();

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].shape(), [2, 2]);
    assert_eq!(view_elems(&chunks[0]), vec![1, 2, 6, 7]);
    assert_eq!(chunks[1].shape(), [2, 2]);
    assert_eq!(view_elems(&chunks[1]), vec![3, 4, 8, 9]);
    assert_eq!(chunks[2].shape(), [2, 1]);
    assert_eq!(view_elems(&chunks[2]), vec![5, 10]);
}

#[test]
fn axis_chunks_iter_over_transposed_view_preserves_strides_and_values() {
    let a = leto([2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let transposed = a.transpose([1, 0]).unwrap();
    let chunks: Vec<_> = transposed.axis_chunks_iter(0, 3).unwrap().collect();

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].shape(), [3, 2]);
    assert_eq!(chunks[0].strides(), transposed.strides());
    assert_eq!(chunks[0].offset(), transposed.offset());
    assert_eq!(view_elems(&chunks[0]), vec![1, 5, 2, 6, 3, 7]);
    assert_eq!(chunks[1].shape(), [1, 2]);
    assert_eq!(chunks[1].strides(), transposed.strides());
    assert_eq!(chunks[1].offset(), 3);
    assert_eq!(view_elems(&chunks[1]), vec![4, 8]);
}

#[test]
fn axis_chunks_iter_double_ended_meets_once() {
    let a = leto([1, 7], (1..=7).collect());
    let mut chunks = a.axis_chunks_iter(1, 2).unwrap();

    assert_eq!(view_elems(&chunks.next().unwrap()), vec![1, 2]);
    assert_eq!(view_elems(&chunks.next_back().unwrap()), vec![7]);
    assert_eq!(view_elems(&chunks.next().unwrap()), vec![3, 4]);
    assert_eq!(view_elems(&chunks.next_back().unwrap()), vec![5, 6]);
    assert!(chunks.next().is_none());
    assert!(chunks.next_back().is_none());
}

#[test]
fn axis_chunks_iter_rejects_invalid_axis_and_zero_length() {
    let a = leto([2, 3], vec![1, 2, 3, 4, 5, 6]);

    match a.axis_chunks_iter(2, 1) {
        Err(LetoError::ShapeMismatch { lhs, rhs }) => {
            assert_eq!(lhs, vec![2]);
            assert_eq!(rhs, vec![2]);
        }
        Err(other) => panic!("expected axis error, got {other:?}"),
        Ok(_) => panic!("expected axis error, got Ok"),
    }

    match a.axis_chunks_iter(1, 0) {
        Err(LetoError::StorageError { reason }) => {
            assert_eq!(reason, "axis chunk length on axis 1 must be non-zero");
        }
        Err(other) => panic!("expected zero-length error, got {other:?}"),
        Ok(_) => panic!("expected zero-length error, got Ok"),
    }
}
