use leto::LetoError;
use thiserror::Error;

/// Operand names used by scaled dot-product attention diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionOperand {
    /// Query input.
    Query,
    /// Key input.
    Key,
    /// Value input.
    Value,
    /// Optional keep-mask input.
    Mask,
    /// Forward output.
    Output,
    /// Post-softmax attention weights.
    Weights,
    /// Backward output gradient.
    OutputGradient,
    /// Query-gradient destination.
    QueryGradient,
    /// Key-gradient destination.
    KeyGradient,
    /// Value-gradient destination.
    ValueGradient,
    /// Score scaling factor.
    Scale,
}

/// Typed scaled dot-product attention contract failures.
#[derive(Debug, Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum AttentionError {
    /// A rank-3 operand has the wrong semantic shape.
    #[error("attention {operand:?} shape mismatch: expected {expected:?}, actual {actual:?}")]
    Shape {
        /// Operand whose shape is invalid.
        operand: AttentionOperand,
        /// Required semantic shape.
        expected: [usize; 3],
        /// Supplied shape.
        actual: [usize; 3],
    },
    /// The keep mask cannot broadcast to the score matrix.
    #[error("attention mask shape {actual:?} cannot broadcast to {target:?}")]
    MaskShape {
        /// Supplied mask shape.
        actual: [usize; 3],
        /// Score-matrix shape.
        target: [usize; 3],
    },
    /// The key sequence is empty, so softmax has no defined support.
    #[error("attention key sequence must contain at least one element")]
    EmptyKeySequence,
    /// Backward was called without any requested gradient destination.
    #[error("attention backward requires at least one gradient target")]
    NoGradientTargets,
    /// A numeric input contains NaN or infinity.
    #[error("attention {operand:?} contains a non-finite value")]
    NonFinite {
        /// Operand containing the invalid value.
        operand: AttentionOperand,
    },
    /// Finite inputs would overflow or produce NaN in the selected precision.
    #[error("attention arithmetic produces a non-finite {operand:?} value")]
    ArithmeticNonFinite {
        /// Result operand whose arithmetic is not representable.
        operand: AttentionOperand,
    },
    /// Backward weights do not represent a forward softmax row.
    #[error("attention weights row [{batch}, {query}] is not a probability row")]
    InvalidWeights {
        /// Batch containing the invalid row.
        batch: usize,
        /// Query position containing the invalid row.
        query: usize,
    },
    /// A Leto view layout or storage invariant is invalid.
    #[error("attention {operand:?} layout is invalid: {source}")]
    Layout {
        /// Operand whose view is invalid.
        operand: AttentionOperand,
        /// Underlying Leto layout failure.
        #[source]
        source: LetoError,
    },
    /// Checked workspace-size arithmetic overflowed.
    #[error("attention workspace size overflow")]
    WorkspaceOverflow,
}

/// Result type for scaled dot-product attention operations.
pub type AttentionResult<T> = core::result::Result<T, AttentionError>;
