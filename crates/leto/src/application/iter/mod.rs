//! Array iteration.
//!
//! # Submodules
//! - [`axis`] — [`AxisIter`], [`AxisIterMut`]: subview iteration along one axis.
//! - [`element`] — [`ElementIter`], [`IndexedIter`]: logical-order element and
//!   `(index, element)` iteration (ndarray `iter` / `indexed_iter` parity).
//! - [`windows`] — [`Windows`]: zero-copy sliding-window subviews (ndarray
//!   `windows` parity).

pub mod axis;
pub mod element;
pub mod windows;

pub use axis::{AxisIter, AxisIterMut};
pub use element::{ElementIter, IndexedIter};
pub use windows::Windows;
