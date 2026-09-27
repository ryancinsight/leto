//! Logical-order element iteration over array views (leto `iter` /
//! `indexed_iter` / `indexed_iter_mut` parity).
//!
//! Both iterators walk every logical element of a view in row-major order,
//! resolving each element's physical offset through the view's strides — so an
//! arbitrarily strided, transposed, or broadcast view iterates in the same
//! logical order as its contiguous materialization. They are
//! `ExactSizeIterator` and `DoubleEndedIterator`; the two ends share one
//! `[front, back)` cursor so forward and backward consumption meet exactly once.
//!
//! `odometer` holds the shared row-major offset-stepping primitives and the
//! mutable-aliasing proof; `element_iter` and `indexed_iter` are the plain and
//! `(index, element)` iterators (shared and mutable); `task_partition` splits
//! a mutable view into disjoint logical ranges for parallel workers; `into_iter`
//! wires `&ArrayView`/`&Array`/`&ArrayViewMut` into `for elem in &view`.

mod element_iter;
mod indexed_iter;
mod into_iter;
mod odometer;
mod task_partition;

pub use element_iter::{ElementIter, ElementIterMut};
pub use indexed_iter::{IndexedIter, IndexedIterMut};
pub use task_partition::{TaskPartitionMut, TaskPartitionsMut};
