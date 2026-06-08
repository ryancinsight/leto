/// Error types and result alias.
pub mod error;
/// Const-rank strided layout arithmetic.
pub mod layout;
/// ndarray-style slicing arguments and normalization.
pub mod slice;

pub use error::{LetoError, Result};
pub use layout::Layout;
pub use slice::SliceArg;
