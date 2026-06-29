/// Runtime-rank (dynamic) layout primitives (ADR 0007).
pub mod dynamic;
/// Atlas-native complex number type (ADR: num-complex removal).
pub mod complex;
/// Error types and result alias.
pub mod error;
/// ZST-based compile-time rank expansion helper.
pub mod insert_axis;
/// Const-rank strided layout arithmetic.
pub mod layout;
/// ZST-based compile-time rank reduction helper.
pub mod remove_axis;
/// ndarray-style slicing arguments and normalization.
pub mod slice;

pub use dynamic::LayoutDyn;
pub use error::{LetoError, Result};
pub use insert_axis::InsertAxis;
pub use layout::Layout;
pub use remove_axis::{RankMarker, RemoveAxis};
pub use slice::SliceArg;
