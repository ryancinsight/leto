//! Array iteration.
//!
//! # Submodules
//! - axis — [`AxisIter`], [`AxisIterMut`]: subview iteration along one axis.
//! - chunks — [`AxisChunks`], [`ExactChunks`]: non-overlapping zero-copy block
//!   streaming.
//! - element — [`ElementIter`], [`ElementIterMut`], [`IndexedIter`], [`IndexedIterMut`],
//!   [`TaskPartitionMut`], [`TaskPartitionsMut`]:
//!   logical-order element and `(index, element)` iteration (leto `iter` /
//!   `indexed_iter` / `indexed_iter_mut` parity).
//! - lanes — [`Lanes`], [`LanesMut`]: 1-D views along one axis (leto
//!   `lanes` / `lanes_mut` parity).
//! - windows — [`Windows`]: zero-copy sliding-window subviews (leto
//!   `windows` parity).

pub mod axis;
pub mod chunks;
pub mod element;
pub mod lanes;
pub mod lending;
pub mod windows;

pub use axis::{AxisIter, AxisIterMut};
pub use chunks::{AxisChunks, ExactChunks};
pub use element::{
    ElementIter, ElementIterMut, IndexedIter, IndexedIterMut, TaskPartitionMut, TaskPartitionsMut,
};
pub use lanes::{Lanes, LanesMut};
pub use lending::{LendingIterator, Tiles};
pub use windows::Windows;
