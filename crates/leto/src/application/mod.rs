/// Rank-specific type aliases.
pub mod aliases;
/// Elementwise arithmetic operators (`Add`/`Sub`/`Mul`/`Div`/`Neg`).
pub mod arithmetic;
/// Owned array type.
pub mod array;
/// Named row and column view helpers.
mod axis;
mod constructors;
/// Runtime-rank array boundary layer (ADR 0007).
pub mod dynamic;
/// Stack-backed fixed-size vector and matrix primitives.
pub mod fixed;
mod index;
/// Subview iteration.
pub mod iter;
/// Element-wise reduction operations (sum, mean, min, max, argmin, argmax).
pub mod reduction;
/// Multivariate summary statistics (covariance, correlation).
pub mod statistics;
/// Typed finite-difference stencil contracts.
pub mod stencil;
/// Structural array operations (concat, pad, split).
pub mod structure;
/// Borrowed array view types.
pub mod view;

pub use aliases::{
    Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, ArrayViewMut1,
    ArrayViewMut2, ArrayViewMut3, ArrayViewMut4,
};
pub use arithmetic::ScalarOperand;
pub use array::Array;
pub use dynamic::ArrayD;
pub use fixed::{FixedMatrix, FixedVector};
pub use iter::{
    AxisChunks, AxisIter, AxisIterMut, ElementIter, ElementIterMut, ExactChunks, IndexedIter,
    IndexedIterMut, Lanes, LanesMut, TaskPartitionMut, TaskPartitionsMut, Tiles, Windows,
};
pub use reduction::{
    argmax, argmax_all, argmin, argmin_all, max_all, max_axis, mean_all, mean_axis, median_all,
    median_axis, min_all, min_axis, quantile_all, quantile_axis, std_all, std_axis, sum_all,
    sum_axis, var_all, var_axis, Interpolation,
};
pub use statistics::{covariance, pearson_correlation};
pub use stencil::{BoundaryCondition, Laplacian2D, LaplacianError, LaplacianPolarity};
pub use structure::{concat, pad, split, stack, PadWidth};
pub use view::{ArrayView, ArrayViewMut};
