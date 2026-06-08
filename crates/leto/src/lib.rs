#![warn(missing_docs)]
//! Leto is a systems-optimized N-dimensional strided array library.

pub mod domain;
pub mod infrastructure;
pub mod application;

// Re-exports
pub use domain::error::{LetoError, Result};
pub use domain::layout::Layout;

pub use infrastructure::storage::{
    Storage, StorageMut, SliceStorage, SliceStorageMut, VecStorage,
};

#[cfg(feature = "mnemosyne-alloc")]
pub use infrastructure::storage::MnemosyneStorage;

pub use application::array::Array;
pub use application::view::{ArrayView, ArrayViewMut};
