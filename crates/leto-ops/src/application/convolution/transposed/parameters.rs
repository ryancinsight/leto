use leto::{LetoError, Result};

/// Spatial transposed-convolution parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransposedConvolutionParameters<const D: usize> {
    stride: [usize; D],
    padding: [usize; D],
    output_padding: [usize; D],
    dilation: [usize; D],
}

impl<const D: usize> TransposedConvolutionParameters<D> {
    /// Construct validated spatial transposed-convolution parameters.
    ///
    /// `output_padding` changes only the derived output shape. It does not add
    /// values or padding during the scatter operation.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::InvalidInput`] when a stride or dilation is zero.
    pub fn new(
        stride: [usize; D],
        padding: [usize; D],
        output_padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<Self> {
        if stride.contains(&0) {
            return Err(LetoError::InvalidInput(
                "transposed convolution stride must be nonzero".to_string(),
            ));
        }
        if dilation.contains(&0) {
            return Err(LetoError::InvalidInput(
                "transposed convolution dilation must be nonzero".to_string(),
            ));
        }
        Ok(Self {
            stride,
            padding,
            output_padding,
            dilation,
        })
    }

    /// Return the per-axis scatter stride.
    #[must_use]
    pub const fn stride(&self) -> &[usize; D] {
        &self.stride
    }

    /// Return the symmetric per-axis padding.
    #[must_use]
    pub const fn padding(&self) -> &[usize; D] {
        &self.padding
    }

    /// Return the per-axis shape-only output padding.
    #[must_use]
    pub const fn output_padding(&self) -> &[usize; D] {
        &self.output_padding
    }

    /// Return the per-axis kernel dilation.
    #[must_use]
    pub const fn dilation(&self) -> &[usize; D] {
        &self.dilation
    }
}
