#![warn(missing_docs)]
//! Leto is a systems-optimized N-dimensional strided array library.

/// Application-level array and view types.
pub mod application;
/// Domain-level layout, slicing, and error contracts.
pub mod domain;
/// Infrastructure storage backends.
pub mod infrastructure;

// Re-exports
pub use domain::error::{LetoError, Result};
pub use domain::layout::Layout;
pub use domain::slice::SliceArg;

pub use infrastructure::storage::{SliceStorage, SliceStorageMut, Storage, StorageMut, VecStorage};

#[cfg(feature = "mnemosyne-alloc")]
pub use infrastructure::storage::MnemosyneStorage;

pub use application::array::Array;
pub use application::view::{ArrayView, ArrayViewMut};
