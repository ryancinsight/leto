//! Validated convolution parameter vocabulary.

use super::error::{LetoError, Result};

/// Spatial regular-convolution parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvolutionParameters<const D: usize> {
    stride: [usize; D],
    padding: [usize; D],
    dilation: [usize; D],
}

impl<const D: usize> ConvolutionParameters<D> {
    /// Construct validated spatial convolution parameters.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::InvalidInput`] when a stride or dilation is zero.
    pub fn new(stride: [usize; D], padding: [usize; D], dilation: [usize; D]) -> Result<Self> {
        if stride.contains(&0) {
            return Err(LetoError::InvalidInput(
                "convolution stride must be nonzero".to_string(),
            ));
        }
        if dilation.contains(&0) {
            return Err(LetoError::InvalidInput(
                "convolution dilation must be nonzero".to_string(),
            ));
        }
        Ok(Self {
            stride,
            padding,
            dilation,
        })
    }

    /// Return the per-axis traversal stride.
    #[must_use]
    pub const fn stride(&self) -> &[usize; D] {
        &self.stride
    }

    /// Return the symmetric per-axis padding.
    #[must_use]
    pub const fn padding(&self) -> &[usize; D] {
        &self.padding
    }

    /// Return the per-axis kernel dilation.
    #[must_use]
    pub const fn dilation(&self) -> &[usize; D] {
        &self.dilation
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_regular_stride_and_dilation() {
        assert_eq!(
            ConvolutionParameters::new([0], [0], [1]),
            Err(LetoError::InvalidInput(
                "convolution stride must be nonzero".to_string()
            ))
        );
        assert_eq!(
            ConvolutionParameters::new([1], [0], [0]),
            Err(LetoError::InvalidInput(
                "convolution dilation must be nonzero".to_string()
            ))
        );
    }

    #[test]
    fn rejects_zero_transposed_stride_and_dilation() {
        assert_eq!(
            TransposedConvolutionParameters::new([0], [0], [0], [1]),
            Err(LetoError::InvalidInput(
                "transposed convolution stride must be nonzero".to_string()
            ))
        );
        assert_eq!(
            TransposedConvolutionParameters::new([1], [0], [0], [0]),
            Err(LetoError::InvalidInput(
                "transposed convolution dilation must be nonzero".to_string()
            ))
        );
    }

    #[test]
    fn preserves_validated_parameter_values() {
        let regular = ConvolutionParameters::new([2, 3], [4, 5], [6, 7])
            .expect("valid regular convolution parameters");
        assert_eq!(regular.stride(), &[2, 3]);
        assert_eq!(regular.padding(), &[4, 5]);
        assert_eq!(regular.dilation(), &[6, 7]);

        let transposed = TransposedConvolutionParameters::new([2, 3], [4, 5], [1, 2], [6, 7])
            .expect("valid transposed convolution parameters");
        assert_eq!(transposed.stride(), &[2, 3]);
        assert_eq!(transposed.padding(), &[4, 5]);
        assert_eq!(transposed.output_padding(), &[1, 2]);
        assert_eq!(transposed.dilation(), &[6, 7]);
    }
}
