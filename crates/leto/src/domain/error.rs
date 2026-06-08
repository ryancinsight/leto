use thiserror::Error;

/// Core error types for the Leto N-dimensional strided array library.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LetoError {
    /// The index was out of bounds for the layout shape.
    #[error("Index out of bounds: index {index:?}, shape {shape:?}")]
    OutOfBounds {
        /// The requested index.
        index: Vec<usize>,
        /// The shape of the layout.
        shape: Vec<usize>,
    },

    /// The requested slice is incompatible with the layout.
    #[error("Incompatible slice: range {range:?}, shape {shape:?}")]
    IncompatibleSlice {
        /// The requested slice range.
        range: (usize, usize),
        /// The shape of the layout.
        shape: Vec<usize>,
    },

    /// The shapes are incompatible for the requested operation.
    #[error("Shape mismatch: lhs {lhs:?}, rhs {rhs:?}")]
    ShapeMismatch {
        /// The shape of the left operand.
        lhs: Vec<usize>,
        /// The shape of the right operand.
        rhs: Vec<usize>,
    },

    /// Shapes are incompatible for broadcasting.
    #[error("Incompatible broadcast: from {from:?} to {to:?}")]
    IncompatibleBroadcast {
        /// The source shape.
        from: Vec<usize>,
        /// The target shape.
        to: Vec<usize>,
    },

    /// Mathematical calculation overflowed integer bounds.
    #[error("Arithmetic overflow in layout calculations: {reason}")]
    Overflow {
        /// The cause/location of the overflow.
        reason: &'static str,
    },

    /// Storage allocation or size validation failed.
    #[error("Storage error: {reason}")]
    StorageError {
        /// The detail message.
        reason: String,
    },
}

/// A specialized Result alias for Leto operations.
pub type Result<T> = std::result::Result<T, LetoError>;
