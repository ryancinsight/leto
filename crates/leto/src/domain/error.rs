use thiserror::Error;

/// Core error types for the Leto N-dimensional strided array library.
#[derive(Debug, Error, Clone, PartialEq)]
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

    /// An iterative solver did not converge within the iteration limit.
    #[error(
        "Solver did not converge after {max_iters} iterations (residual {residual:e}, \
         tolerance {tol:e})"
    )]
    ConvergenceError {
        /// The maximum iteration count that was exceeded.
        max_iters: usize,
        /// The final relative residual norm.
        residual: f64,
        /// The convergence tolerance that was not met.
        tol: f64,
    },

    /// An invalid input or configuration was supplied to an operation or solver.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// A numerical breakdown condition was detected (e.g., near-zero denominator, NaN).
    #[error("Numerical breakdown: {0}")]
    NumericalBreakdown(String),

    /// The matrix is not positive definite (required for Cholesky / CG).
    #[error("Matrix is not positive definite: {detail}")]
    NotPositiveDefinite {
        /// Explanation of the condition detected.
        detail: String,
    },
}

/// A specialized Result alias for Leto operations.
pub type Result<T> = std::result::Result<T, LetoError>;
