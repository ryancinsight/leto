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
pub use domain::remove_axis::{RankMarker, RemoveAxis};
pub use domain::slice::SliceArg;

pub use infrastructure::storage::{
    CowStorage, SliceStorage, SliceStorageMut, Storage, StorageMut, VecStorage,
};

#[cfg(feature = "mnemosyne-alloc")]
pub use infrastructure::storage::MnemosyneStorage;

pub use application::array::Array;
pub use application::view::{ArrayView, ArrayViewMut};
pub use application::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMut3, AxisIter, AxisIterMut,
};
