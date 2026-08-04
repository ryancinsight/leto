use leto::LetoError;
use thiserror::Error;

/// Operand names used by cross-entropy diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossEntropyOperand {
    /// Rank-two input logits.
    Logits,
    /// Rank-two saved softmax probabilities.
    Probabilities,
    /// Rank-one scalar loss destination.
    Loss,
    /// Rank-two additive logit-gradient destination.
    LogitGradient,
    /// Rank-one upstream scalar gradient.
    OutputGradient,
}

/// Typed failures from the mean cross-entropy contract.
#[derive(Debug, Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum CrossEntropyError {
    /// An operand has the wrong semantic shape.
    #[error("cross-entropy {operand:?} shape mismatch: expected {expected:?}, actual {actual:?}")]
    Shape {
        /// Operand whose shape is invalid.
        operand: CrossEntropyOperand,
        /// Required semantic shape.
        expected: Vec<usize>,
        /// Supplied shape.
        actual: Vec<usize>,
    },
    /// The batch dimension is empty, so mean reduction is undefined.
    #[error("cross-entropy batch must contain at least one sample")]
    EmptyBatch,
    /// The class dimension is empty, so softmax has no support.
    #[error("cross-entropy class dimension must contain at least one class")]
    EmptyClasses,
    /// The target count differs from the batch size.
    #[error("cross-entropy target count mismatch: expected {expected}, actual {actual}")]
    TargetCount {
        /// Required target count.
        expected: usize,
        /// Supplied target count.
        actual: usize,
    },
    /// A target does not name a class in the logits row.
    #[error("cross-entropy target {target} at batch {batch} is outside class range 0..{classes}")]
    TargetOutOfRange {
        /// Batch row containing the invalid target.
        batch: usize,
        /// Invalid target index.
        target: usize,
        /// Number of classes.
        classes: usize,
    },
    /// A layout or storage invariant is invalid.
    #[error("cross-entropy {operand:?} layout is invalid: {source}")]
    Layout {
        /// Operand whose layout is invalid.
        operand: CrossEntropyOperand,
        /// Underlying Leto layout failure.
        #[source]
        source: LetoError,
    },
    /// A dimension cannot be represented in the selected scalar precision.
    #[error("cross-entropy {dimension} extent {extent} is not finite in the selected scalar")]
    ScalarExtent {
        /// Dimension whose scalar conversion failed.
        dimension: &'static str,
        /// Unrepresentable extent.
        extent: usize,
    },
    /// Probability-row validation cannot derive a meaningful error bound.
    #[error(
        "cross-entropy class extent {classes} exceeds the selected scalar's probability-resolution bound"
    )]
    ProbabilityResolution {
        /// Class count whose accumulated roundoff bound is at least one half.
        classes: usize,
    },
    /// A numeric operand contains NaN or infinity.
    #[error("cross-entropy {operand:?} contains a non-finite value")]
    NonFinite {
        /// Operand containing the invalid value.
        operand: CrossEntropyOperand,
    },
    /// Finite logits exceed the selected precision's stable arithmetic range.
    #[error("cross-entropy arithmetic is not finite for batch row {batch}")]
    ArithmeticNonFinite {
        /// Batch row whose finite range is not representable.
        batch: usize,
    },
    /// Saved probabilities do not form a valid row.
    #[error("cross-entropy probabilities at batch row {batch} do not form a valid row")]
    InvalidProbabilities {
        /// Batch row containing invalid probabilities.
        batch: usize,
    },
}

/// Result type for cross-entropy operations.
pub type CrossEntropyResult<T> = core::result::Result<T, CrossEntropyError>;
