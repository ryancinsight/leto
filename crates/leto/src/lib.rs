#![warn(missing_docs)]
//! Leto is a systems-optimized N-dimensional strided array library.

/// Application-level array and view types.
pub mod application;
/// Domain-level layout, slicing, and error contracts.
pub mod domain;
/// Infrastructure storage backends.
pub mod infrastructure;

// Re-exports
pub use domain::dynamic::LayoutDyn;
pub use domain::error::{LetoError, Result};
pub use domain::insert_axis::InsertAxis;
pub use domain::layout::Layout;
pub use domain::remove_axis::{RankMarker, RemoveAxis};
pub use domain::slice::SliceArg;

pub use infrastructure::storage::{
    CowStorage, SliceStorage, SliceStorageMut, StackStorage, Storage, StorageMut, VecStorage,
};

#[cfg(feature = "mnemosyne-alloc")]
pub use infrastructure::storage::MnemosyneStorage;

pub use application::array::Array;
pub use application::view::{ArrayView, ArrayViewMut};
pub use application::{
    concat, covariance, median_all, median_axis, pad, pearson_correlation, quantile_all,
    quantile_axis, split, stack, Array1, Array2, Array3, ArrayD, ArrayView1, ArrayView2,
    ArrayView3, ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, AxisIter, AxisIterMut, ElementIter,
    IndexedIter, Interpolation, Lanes, LanesMut, PadWidth, ScalarOperand, Windows,
};

#[cfg(feature = "ndarray-compat")]
/// ndarray compatibility conversions.
pub use infrastructure::ndarray_compat;
