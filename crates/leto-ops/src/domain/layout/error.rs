//! Dimension failures for in-place complex square movement.

use thiserror::Error;

/// A square transpose extent that cannot describe the supplied storage.
///
/// Dimensions are retained directly; constructing either error allocates no
/// storage and leaves the caller's matrix unchanged.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SquareTransposeError {
    /// The number of complex samples exceeds the representable element count.
    #[error("complex square matrix element count overflows for side {side}")]
    Overflow {
        /// Requested row and column count.
        side: usize,
    },
    /// Storage does not contain exactly one square of the requested side.
    #[error(
        "complex square transpose side {side} requires {expected} samples, but storage contains {actual}"
    )]
    Length {
        /// Requested row and column count.
        side: usize,
        /// Required complex sample count.
        expected: usize,
        /// Supplied complex sample count.
        actual: usize,
    },
}
