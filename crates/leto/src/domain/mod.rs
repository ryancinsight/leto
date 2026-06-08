/// Error types and result alias.
pub mod error;
/// Const-rank strided layout arithmetic.
pub mod layout;
/// ZST-based compile-time rank reduction helper.
pub mod remove_axis;
/// ndarray-style slicing arguments and normalization.
pub mod slice;

pub use error::{LetoError, Result};
pub use layout::Layout;
pub use remove_axis::{RankMarker, RemoveAxis};
pub use slice::SliceArg;
