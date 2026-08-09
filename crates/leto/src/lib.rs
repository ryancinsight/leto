#![warn(missing_docs)]
//! Leto is a systems-optimized N-dimensional strided array library.

/// Application-level array and view types.
pub mod application;
/// Domain-level layout, slicing, and error contracts.
pub mod domain;
/// Fixed-size geometry primitives.
pub mod geometry;
/// Infrastructure storage backends.
pub mod infrastructure;

// Re-exports
pub use domain::convolution::{ConvolutionParameters, TransposedConvolutionParameters};
pub use domain::dynamic::LayoutDyn;
pub use domain::error::{LetoError, Result};
pub use domain::insert_axis::InsertAxis;
pub use domain::layout::Layout;
pub use domain::remove_axis::{RankMarker, RemoveAxis};
pub use domain::slice::SliceArg;
/// Complex number — re-exported from `eunomia`, the Atlas datatype SSOT
/// ([ADR 0011](../docs/adr/0011-num-complex-removal.md ; see also eunomia ADR 0001)).
pub use eunomia::Complex;
pub use geometry::{
    Isometry3, Point, Point2, Point3, Quaternion, RotationBasisError, Translation3, Unit,
    UnitQuaternion, Vector, Vector2, Vector3,
};

pub use infrastructure::storage::{
    CowStorage, SliceStorage, SliceStorageMut, StackStorage, Storage, StorageMut, VecStorage,
};

#[cfg(feature = "mnemosyne-alloc")]
pub use infrastructure::storage::MnemosyneStorage;

/// Sparse array storage formats (CSR, CSC, COO) for efficient sparse matrix operations.
pub use infrastructure::sparse::{
    CooArray, CscArray, CsrArray, SparseFormat, SparseStorage, SparseStorageMut,
};

pub use application::array::Array;
pub use application::stencil::{BoundaryCondition, Laplacian2D, LaplacianError, LaplacianPolarity};
pub use application::view::{ArrayView, ArrayViewMut};
pub use application::{
    concat, covariance, mean_all, mean_axis, median_all, median_axis, pad, pearson_correlation,
    quantile_all, quantile_axis, split, stack, sum_all, sum_axis, Array1, Array2, Array3, Array4,
    ArrayD, ArrayView1, ArrayView2, ArrayView3, ArrayView4, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMut3, ArrayViewMut4, AxisChunks, AxisIter, AxisIterMut, ElementIter, ElementIterMut,
    ExactChunks, FixedMatrix, FixedVector, IndexedIter, IndexedIterMut, Interpolation, Lanes,
    LanesMut, LendingIterator, PadWidth, ScalarOperand, Tiles, Windows,
};
