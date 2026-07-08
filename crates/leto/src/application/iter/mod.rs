//! Array iteration.
//!
//! # Submodules
//! - axis — [`AxisIter`], [`AxisIterMut`]: subview iteration along one axis.
//! - chunks — [`AxisChunks`], [`ExactChunks`]: non-overlapping zero-copy block
//!   streaming.
//! - element — [`ElementIter`], [`IndexedIter`], [`IndexedIterMut`]:
//!   logical-order element and `(index, element)` iteration (ndarray `iter` /
//!   `indexed_iter` / `indexed_iter_mut` parity).
//! - lanes — [`Lanes`], [`LanesMut`]: 1-D views along one axis (ndarray
//!   `lanes` / `lanes_mut` parity).
//! - windows — [`Windows`]: zero-copy sliding-window subviews (ndarray
//!   `windows` parity).

pub mod axis;
pub mod chunks;
pub mod element;
pub mod lanes;
pub mod windows;

pub use axis::{AxisIter, AxisIterMut};
pub use chunks::{AxisChunks, ExactChunks};
pub use element::{ElementIter, IndexedIter, IndexedIterMut};
pub use lanes::{Lanes, LanesMut};
pub use windows::Windows;
