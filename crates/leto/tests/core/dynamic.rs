//! Runtime-rank `ArrayD` boundary carrier + zero-copy rank bridge (ADR 0007).

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::application::reduction::sum_all;
use leto::{Array2, ArrayD, LayoutDyn, LetoError, VecStorage};

fn dyn_vec(shape: &[usize], data: Vec<i32>) -> ArrayD<i32, VecStorage<i32>> {
    ArrayD::from_shape_vec(shape, data).unwrap()
}

#[track_caller]
fn assert_storage_reason<T: std::fmt::Debug>(result: leto::Result<T>, expected: &str) {
    match result {
        Err(LetoError::StorageError { reason }) => assert_eq!(reason, expected),
        other => panic!("expected storage error {expected:?}, got {other:?}"),
    }
}

#[track_caller]
fn assert_out_of_bounds<T: std::fmt::Debug>(
    result: leto::Result<T>,
    index: Vec<usize>,
    shape: Vec<usize>,
) {
    match result {
        Err(LetoError::OutOfBounds {
            index: actual_index,
            shape: actual_shape,
        }) => {
            assert_eq!(actual_index, index);
            assert_eq!(actual_shape, shape);
        }
        other => panic!("expected OutOfBounds({index:?}, {shape:?}), got {other:?}"),
    }
}

#[test]
fn construct_inspect_and_index() {
    let a = dyn_vec(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(a.ndim(), 2);
    assert_eq!(a.shape(), &[2, 3]);
    assert_eq!(a.size(), 6);
    assert!(!a.is_empty());
    assert_eq!(*a.get(&[0, 0]).unwrap(), 1);
    assert_eq!(*a.get(&[1, 2]).unwrap(), 6);
    assert_eq!(*a.get(&[0, 2]).unwrap(), 3);
}

#[test]
fn rank_is_a_runtime_value() {
    // Same type, three different ranks — impossible for the const-rank Array.
    let r1 = dyn_vec(&[4], vec![1, 2, 3, 4]);
    let r2 = dyn_vec(&[2, 2], vec![1, 2, 3, 4]);
    let r3 = dyn_vec(&[2, 1, 2], vec![1, 2, 3, 4]);
    assert_eq!(r1.ndim(), 1);
    assert_eq!(r2.ndim(), 2);
    assert_eq!(r3.ndim(), 3);
}

#[test]
fn get_rejects_wrong_arity_and_out_of_range() {
    let a = dyn_vec(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    assert_out_of_bounds(a.get(&[0]), vec![0], vec![2, 3]);
    assert_out_of_bounds(a.get(&[2, 0]), vec![2, 0], vec![2, 3]);
    assert_out_of_bounds(a.get(&[0, 3]), vec![0, 3], vec![2, 3]);
}

#[test]
fn from_shape_vec_rejects_length_mismatch() {
    assert_storage_reason(
        ArrayD::<i32, _>::from_shape_vec(&[2, 3], vec![1, 2, 3]),
        "vector length 3 does not match layout size 6",
    );
}

#[test]
fn zeros_and_to_vec_row_major() {
    let z: ArrayD<f64, _> = ArrayD::zeros(&[2, 2]).unwrap();
    assert_eq!(z.to_vec().unwrap(), vec![0.0; 4]);
    let a = dyn_vec(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(a.to_vec().unwrap(), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn into_shape_zero_copy_reshape() {
    let a = dyn_vec(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let b = a.into_shape(&[3, 2]).unwrap();
    assert_eq!(b.shape(), &[3, 2]);
    assert_eq!(b.to_vec().unwrap(), vec![1, 2, 3, 4, 5, 6]);
    // Mismatched element count is rejected.
    let c = dyn_vec(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    match c.into_shape(&[2, 2]) {
        Err(LetoError::ShapeMismatch { lhs, rhs }) => {
            assert_eq!(lhs, vec![2, 3]);
            assert_eq!(rhs, vec![2, 2]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn strided_layout_offset_and_indexing() {
    // Column-major-style strides over a 2×2: offset(i,j) = i*1 + j*2.
    let layout = LayoutDyn::new(
        vec![2usize, 2].into_boxed_slice(),
        vec![1isize, 2].into_boxed_slice(),
        0,
    )
    .unwrap();
    assert_eq!(layout.offset_of(&[0, 0]).unwrap(), 0);
    assert_eq!(layout.offset_of(&[1, 0]).unwrap(), 1);
    assert_eq!(layout.offset_of(&[0, 1]).unwrap(), 2);
    assert_eq!(layout.offset_of(&[1, 1]).unwrap(), 3);
    // Backed by data [10,20,30,40]: logical [[10,30],[20,40]].
    let a = ArrayD::new(layout, VecStorage::new(vec![10, 20, 30, 40])).unwrap();
    assert_eq!(*a.get(&[0, 1]).unwrap(), 30);
    assert_eq!(*a.get(&[1, 0]).unwrap(), 20);
    assert_eq!(a.to_vec().unwrap(), vec![10, 30, 20, 40]); // logical row-major
}

#[test]
fn layout_dyn_rejects_rank_mismatch() {
    assert_storage_reason(
        LayoutDyn::new(
            vec![2usize, 2].into_boxed_slice(),
            vec![1isize].into_boxed_slice(),
            0,
        ),
        "dynamic layout shape rank 2 does not match stride rank 1",
    );
}

#[test]
fn bridge_round_trip_preserves_value_and_shape() {
    let a = Array2::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let original: Vec<i32> = a.iter().copied().collect();
    let d = a.into_dyn();
    assert_eq!(d.ndim(), 2);
    assert_eq!(d.shape(), &[2, 3]);
    let back = d.into_dimensionality::<2>().unwrap();
    assert_eq!(back.shape(), [2, 3]);
    let recovered: Vec<i32> = back.iter().copied().collect();
    assert_eq!(recovered, original);
}

#[test]
fn bridge_rank_mismatch_is_rejected() {
    let a = Array2::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let d = a.into_dyn();
    // Requesting the wrong const rank fails (data is rank 2).
    assert_storage_reason(
        d.clone().into_dimensionality::<1>(),
        "array rank 2 does not match requested const rank 1",
    );
    assert_storage_reason(
        d.clone().into_dimensionality::<3>(),
        "array rank 2 does not match requested const rank 3",
    );
    let typed = d.into_dimensionality::<2>().unwrap();
    assert_eq!(typed.shape(), [2, 3]);
}

#[test]
fn bridge_recovers_for_const_rank_compute() {
    // The intended workflow: carry runtime-rank data, recover a typed rank, then
    // use an existing const-rank kernel.
    let a = Array2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let d = a.into_dyn();
    let typed = d.into_dimensionality::<2>().unwrap();
    assert_eq!(sum_all(&typed).unwrap(), 21.0);
}

#[test]
fn dynamic_rank_dispatch_pattern() {
    // Runtime-rank dispatch: sum an ArrayD of unknown rank via the bridge.
    fn dyn_sum(a: ArrayD<f64, VecStorage<f64>>) -> leto::Result<f64> {
        match a.ndim() {
            1 => sum_all(&a.into_dimensionality::<1>()?),
            2 => sum_all(&a.into_dimensionality::<2>()?),
            3 => sum_all(&a.into_dimensionality::<3>()?),
            n => Err(leto::LetoError::StorageError {
                reason: format!("rank {n} exceeds supported dispatch range"),
            }),
        }
    }
    let r1 = ArrayD::from_shape_vec(&[3], vec![1.0, 2.0, 3.0]).unwrap();
    let r2 = ArrayD::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(dyn_sum(r1).unwrap(), 6.0);
    assert_eq!(dyn_sum(r2).unwrap(), 10.0);
}

#[test]
fn layout_dyn_validates_spans_and_injectivity_without_materializing_offsets() {
    let dense = LayoutDyn::new(
        vec![2usize, 3].into_boxed_slice(),
        vec![3isize, 1].into_boxed_slice(),
        0,
    )
    .unwrap();
    assert_eq!(dense.checked_min_max_offsets().unwrap(), (0, 5));
    assert!(dense.is_injective().unwrap());

    let transposed = LayoutDyn::new(
        vec![2usize, 3].into_boxed_slice(),
        vec![1isize, 2].into_boxed_slice(),
        0,
    )
    .unwrap();
    assert_eq!(transposed.checked_min_max_offsets().unwrap(), (0, 5));
    assert!(transposed.is_injective().unwrap());

    let aliased = LayoutDyn::new(
        vec![2usize, 2].into_boxed_slice(),
        vec![1isize, 1].into_boxed_slice(),
        0,
    )
    .unwrap();
    assert!(!aliased.is_injective().unwrap());

    let reverse = LayoutDyn::new(
        vec![3usize].into_boxed_slice(),
        vec![-1isize].into_boxed_slice(),
        2,
    )
    .unwrap();
    assert_eq!(reverse.checked_min_max_offsets().unwrap(), (0, 2));

    let negative = LayoutDyn::new(
        vec![3usize].into_boxed_slice(),
        vec![-1isize].into_boxed_slice(),
        1,
    )
    .unwrap();
    assert_storage_reason(
        negative.checked_min_max_offsets(),
        "layout accesses negative physical offset -1",
    );
}

#[test]
fn layout_dyn_broadcast_preserves_storage_and_marks_expanded_axes() {
    let source = LayoutDyn::new(
        vec![1usize, 3].into_boxed_slice(),
        vec![0isize, 1].into_boxed_slice(),
        4,
    )
    .unwrap();
    let expanded = source.broadcast(&[2, 3]).unwrap();
    assert_eq!(expanded.shape.as_ref(), &[2, 3]);
    assert_eq!(expanded.strides.as_ref(), &[0, 1]);
    assert_eq!(expanded.offset, 4);
    assert_eq!(expanded.checked_min_max_offsets().unwrap(), (4, 6));
    assert!(!expanded.is_injective().unwrap());

    let incompatible = source.broadcast(&[2, 4]);
    match incompatible {
        Err(LetoError::IncompatibleBroadcast { from, to }) => {
            assert_eq!(from, vec![1, 3]);
            assert_eq!(to, vec![2, 4]);
        }
        other => panic!("expected incompatible broadcast, got {other:?}"),
    }
}

#[test]
fn layout_dyn_empty_and_overflow_contracts_are_explicit() {
    let empty = LayoutDyn::new(
        vec![0usize, 3].into_boxed_slice(),
        vec![3isize, 1].into_boxed_slice(),
        7,
    )
    .unwrap();
    assert_eq!(empty.checked_min_max_offsets().unwrap(), (7, 7));
    assert!(empty.is_injective().unwrap());

    let overflowing = LayoutDyn::new(
        vec![usize::MAX, 2].into_boxed_slice(),
        vec![1isize, 1].into_boxed_slice(),
        0,
    )
    .unwrap();
    match overflowing.checked_min_max_offsets() {
        Err(LetoError::Overflow { reason }) => {
            assert_eq!(reason, "layout dimension bound conversion")
        }
        other => panic!("expected layout overflow, got {other:?}"),
    }
}
