/// Validated regular and transposed convolution parameters.
pub mod convolution;
/// Runtime-rank (dynamic) layout primitives (ADR 0007).
pub mod dynamic;
/// Error types and result alias.
pub mod error;
/// ZST-based compile-time rank expansion helper.
pub mod insert_axis;
/// Const-rank strided layout arithmetic.
pub mod layout;
/// ZST-based compile-time rank reduction helper.
pub mod remove_axis;
/// leto-style slicing arguments and normalization.
pub mod slice;
/// Validated spatial sliding-window parameters.
pub mod window;

pub use convolution::{ConvolutionParameters, TransposedConvolutionParameters};
pub use dynamic::LayoutDyn;
pub use error::{LetoError, Result};
pub use insert_axis::InsertAxis;
pub use layout::Layout;
pub use remove_axis::{RankMarker, RemoveAxis};
pub use slice::SliceArg;
pub use window::WindowParameters;
