use thiserror::Error;

/// Failures in temporal alignment inputs, storage, or arithmetic.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CtcError {
    /// Mean reduction requires a nonempty batch and class alphabet.
    #[error("CTC requires nonzero batch and classes, got shape {shape:?}")]
    EmptyDimension {
        /// Logical `[frames, batch, classes]` dimensions.
        shape: [usize; 3],
    },
    /// The length arrays must each have one entry per sample.
    #[error("CTC length counts must equal batch {batch}, got inputs {inputs}, targets {targets}")]
    LengthCount {
        /// Batch extent.
        batch: usize,
        /// Input-length entries.
        inputs: usize,
        /// Target-length entries.
        targets: usize,
    },
    /// A sample requests more frames than the input contains.
    #[error("CTC sample {sample} input length {actual} exceeds {maximum}")]
    InputLength {
        /// Sample index.
        sample: usize,
        /// Requested valid frames.
        actual: usize,
        /// Available frames.
        maximum: usize,
    },
    /// Concatenated targets must match the sum of their lengths.
    #[error("CTC target count must be {expected}, got {actual}")]
    TargetCount {
        /// Sum of target lengths.
        expected: usize,
        /// Supplied labels.
        actual: usize,
    },
    /// A blank or target label is outside the alphabet, or a target is blank.
    #[error("CTC invalid label {label} with blank {blank} and {classes} classes")]
    Label {
        /// Invalid class index.
        label: usize,
        /// Blank class index.
        blank: usize,
        /// Class extent.
        classes: usize,
    },
    /// Shape arithmetic cannot represent the requested state.
    #[error("CTC state size overflows for {frames} frames and {targets} targets")]
    SizeOverflow {
        /// Input frame extent.
        frames: usize,
        /// Target count or accumulated extent.
        targets: usize,
    },
    /// Allocation failed without modifying caller storage.
    #[error("CTC state allocation failed")]
    Allocation(#[source] std::collections::TryReserveError),
    /// A borrowed layout cannot reach its storage.
    #[error("CTC operand layout is invalid")]
    Layout(#[source] leto::LetoError),
    /// The additive destination must match the forward input shape.
    #[error("CTC gradient shape must be {expected:?}, got {actual:?}")]
    GradientShape {
        /// Forward input dimensions.
        expected: [usize; 3],
        /// Destination dimensions.
        actual: [usize; 3],
    },
    /// A mutable layout aliases distinct logical elements.
    #[error("CTC gradient layout contains overlapping elements")]
    AliasedGradient,
    /// Active log-probabilities must be nonpositive and not NaN.
    #[error(
        "CTC invalid log-probability at {index:?}; expected nonpositive finite value or -infinity"
    )]
    LogProbability {
        /// Logical input coordinate.
        index: [usize; 3],
    },
    /// A normalization count is outside the scalar's finite range.
    #[error("CTC normalization extent {extent} is not finite in the selected scalar")]
    ScalarExtent {
        /// Unrepresentable count.
        extent: usize,
    },
    /// A finite computation or seed cannot be represented in the scalar.
    #[error("CTC arithmetic is not finite for sample {sample}")]
    Arithmetic {
        /// Affected sample index.
        sample: usize,
    },
    /// Zero total alignment probability has no finite loss derivative.
    #[error("CTC sample {sample} has no positive-probability alignment")]
    ImpossibleAlignment {
        /// Affected sample index.
        sample: usize,
    },
}
