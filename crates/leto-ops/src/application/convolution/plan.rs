use leto::{ArrayView, ArrayViewMut, LetoError, Result};

/// Spatial convolution parameters.
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

    pub(super) const fn stride(&self) -> &[usize; D] {
        &self.stride
    }

    pub(super) const fn padding(&self) -> &[usize; D] {
        &self.padding
    }

    pub(super) const fn dilation(&self) -> &[usize; D] {
        &self.dilation
    }
}

pub(super) struct ConvolutionPlan<const D: usize> {
    pub(super) input_channels: usize,
    pub(super) input_spatial: [usize; D],
    pub(super) kernel_spatial: [usize; D],
    pub(super) parameters: ConvolutionParameters<D>,
    pub(super) output_elements: usize,
    pub(super) kernel_elements: usize,
}

impl<const D: usize> ConvolutionPlan<D> {
    pub(super) fn validate<T, const R: usize>(
        input: &ArrayView<'_, T, R>,
        weight: &ArrayView<'_, T, R>,
        bias: Option<&ArrayView<'_, T, 1>>,
        parameters: ConvolutionParameters<D>,
        output: &ArrayViewMut<'_, T, R>,
    ) -> Result<Self> {
        let expected_rank = D.checked_add(2).ok_or(LetoError::Overflow {
            reason: "convolution tensor rank",
        })?;
        if D == 0 || R != expected_rank {
            return Err(LetoError::InvalidInput(format!(
                "convolution rank {R} must equal spatial rank {D} plus batch/channel axes"
            )));
        }

        input.layout().validate_storage_len(input.data().len())?;
        weight.layout().validate_storage_len(weight.data().len())?;
        output.layout().validate_storage_len(output.data().len())?;
        if output.layout().has_zero_stride_aliasing() {
            return Err(LetoError::StorageError {
                reason: "convolution output layout must not contain zero-stride aliasing"
                    .to_string(),
            });
        }
        if let Some(bias) = bias {
            bias.layout().validate_storage_len(bias.data().len())?;
        }

        let input_shape = input.shape();
        let weight_shape = weight.shape();
        let output_shape = output.shape();
        let batch = input_shape[0];
        let input_channels = input_shape[1];
        let output_channels = weight_shape[0];

        if weight_shape[1] != input_channels
            || output_shape[0] != batch
            || output_shape[1] != output_channels
        {
            return Err(LetoError::ShapeMismatch {
                lhs: input_shape.to_vec(),
                rhs: output_shape.to_vec(),
            });
        }
        if let Some(bias) = bias {
            if bias.shape() != [output_channels] {
                return Err(LetoError::ShapeMismatch {
                    lhs: bias.shape().to_vec(),
                    rhs: vec![output_channels],
                });
            }
        }

        let mut input_spatial = [0; D];
        let mut kernel_spatial = [0; D];
        let mut output_spatial = [0; D];
        for axis in 0..D {
            let input_extent = input_shape[axis + 2];
            let kernel_extent = weight_shape[axis + 2];
            if kernel_extent == 0 {
                return Err(LetoError::InvalidInput(
                    "convolution kernel extents must be nonzero".to_string(),
                ));
            }
            let effective_kernel = parameters.dilation()[axis]
                .checked_mul(kernel_extent - 1)
                .and_then(|extent| extent.checked_add(1))
                .ok_or(LetoError::Overflow {
                    reason: "convolution effective kernel extent",
                })?;
            let padded_input = parameters.padding()[axis]
                .checked_mul(2)
                .and_then(|padding| input_extent.checked_add(padding))
                .ok_or(LetoError::Overflow {
                    reason: "convolution padded input extent",
                })?;
            let output_extent = padded_input
                .checked_sub(effective_kernel)
                .map(|extent| extent / parameters.stride()[axis] + 1)
                .unwrap_or(0);
            input_spatial[axis] = input_extent;
            kernel_spatial[axis] = kernel_extent;
            output_spatial[axis] = output_extent;
        }
        if output_shape[2..] != output_spatial {
            let mut expected_output = Vec::with_capacity(R);
            expected_output.push(batch);
            expected_output.push(output_channels);
            expected_output.extend_from_slice(&output_spatial);
            return Err(LetoError::ShapeMismatch {
                lhs: output_shape.to_vec(),
                rhs: expected_output,
            });
        }

        let output_elements = output_shape.iter().try_fold(1usize, |count, &extent| {
            count.checked_mul(extent).ok_or(LetoError::Overflow {
                reason: "convolution output element count",
            })
        })?;
        let kernel_elements = kernel_spatial.iter().try_fold(1usize, |count, &extent| {
            count.checked_mul(extent).ok_or(LetoError::Overflow {
                reason: "convolution kernel element count",
            })
        })?;

        Ok(Self {
            input_channels,
            input_spatial,
            kernel_spatial,
            parameters,
            output_elements,
            kernel_elements,
        })
    }
}
