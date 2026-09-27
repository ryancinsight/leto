//! Zero-copy read-only and mutable views over an N-dimensional strided array.
//!
//! `raw` holds the unsafe raw-pointer helpers shared by both view types,
//! `immutable` the read-only `ArrayView`, `mutable_core` the mutable
//! `ArrayViewMut` construction/accessor surface, `mutable_transform` its
//! slicing/transform/contiguity/iteration surface, and `indexing` the
//! `Index`/`IndexMut` operator impls for both types.

mod immutable;
mod indexing;
mod mutable_core;
mod mutable_transform;
mod raw;

#[cfg(test)]
mod as_array_tests;

pub use immutable::ArrayView;
pub use mutable_core::ArrayViewMut;
