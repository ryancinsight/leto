//! Stack-allocated `StackStorage` backing: allocation-free arrays that reuse
//! the full operation surface through the `Storage` trait (SSOT).

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::application::reduction::{mean_all, sum_all, var_all};
use leto::{Array, StackStorage, Storage};

/// A stack-backed 2×2 `f64` array type — no heap allocation.
type StackMat2 = Array<f64, StackStorage<f64, 4>, 2>;

fn stack_2x2(data: [f64; 4]) -> StackMat2 {
    Array::from_stack([2, 2], data).unwrap()
}

#[test]
fn construct_inspect_and_index() {
    let a = stack_2x2([1.0, 2.0, 3.0, 4.0]);
    assert_eq!(a.shape(), [2, 2]);
    assert_eq!(a.size(), 4);
    assert_eq!(*a.get([0, 0]).unwrap(), 1.0);
    assert_eq!(*a.get([1, 1]).unwrap(), 4.0);
    // Backing is the inline array, exposed as a slice.
    assert_eq!(a.storage().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn from_stack_validates_capacity() {
    // CAP (4) must equal the shape's element count.
    let ok: Result<Array<f64, StackStorage<f64, 4>, 2>, _> = Array::from_stack([2, 2], [0.0; 4]);
    assert!(ok.is_ok());
    let bad: Result<Array<f64, StackStorage<f64, 4>, 2>, _> = Array::from_stack([3, 3], [0.0; 4]);
    assert!(bad.is_err());
}

#[test]
fn from_stack_elem_fills() {
    let a: Array<f64, StackStorage<f64, 6>, 2> = Array::from_stack_elem([2, 3], 7.0).unwrap();
    assert_eq!(a.storage().as_slice(), &[7.0; 6]);
}

#[test]
fn reductions_work_on_stack_backed_arrays() {
    // The whole reduction surface is generic over `Storage`, so it runs on a
    // stack-backed array with no per-backend code.
    let a = stack_2x2([1.0, 2.0, 3.0, 4.0]);
    assert_eq!(sum_all(&a).unwrap(), 10.0);
    assert!((mean_all(&a).unwrap() - 2.5).abs() < 1e-12);
    // population variance of [1,2,3,4] = 1.25.
    assert!((var_all(&a, 0.0).unwrap() - 1.25).abs() < 1e-12);
}

#[test]
fn iteration_and_transpose_work_on_stack_backed_arrays() {
    let a = stack_2x2([1.0, 2.0, 3.0, 4.0]);
    let elems: Vec<f64> = a.iter().copied().collect();
    assert_eq!(elems, vec![1.0, 2.0, 3.0, 4.0]);
    // Transpose produces a (borrowing) view; logical order is column-major.
    let t = a.transpose([1, 0]).unwrap();
    let tv: Vec<f64> = t.iter().copied().collect();
    assert_eq!(tv, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn stack_storage_is_copy_and_clone_is_heap_free() {
    // `StackStorage` is `Copy` (pure inline data) — a zero-cost duplicate.
    let s = StackStorage::<f64, 4>::new([1.0, 2.0, 3.0, 4.0]);
    let s_copy = s; // Copy, not move
    assert_eq!(s.as_slice(), s_copy.as_slice());
    // Cloning a stack-backed array copies inline data — no allocation.
    let a = stack_2x2([5.0, 6.0, 7.0, 8.0]);
    let b = a.clone();
    assert_eq!(a.storage().as_slice(), b.storage().as_slice());
}
