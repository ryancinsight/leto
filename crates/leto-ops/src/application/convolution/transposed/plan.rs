use leto::{ArrayView, ArrayViewMut, LetoError, Result, TransposedConvolutionParameters};

pub(super) struct TransposedConvolutionPlan<const D: usize> {
    pub(super) batch: usize,
    pub(super) output_channels: usize,
    pub(super) input_spatial: [usize; D],
    pub(super) kernel_spatial: [usize; D],
    pub(super) output_spatial: [usize; D],
    pub(super) parameters: TransposedConvolutionParameters<D>,
    pub(super) input_elements: usize,
    pub(super) input_spatial_elements: usize,
    pub(super) weight_elements: usize,
    pub(super) output_elements: usize,
    pub(super) output_spatial_elements: usize,
    pub(super) kernel_elements: usize,
}

impl<const D: usize> TransposedConvolutionPlan<D> {
    pub(super) fn validate<T, const R: usize>(
        input: &ArrayView<'_, T, R>,
        weight: &ArrayView<'_, T, R>,
        bias: Option<&ArrayView<'_, T, 1>>,
        parameters: TransposedConvolutionParameters<D>,
        output: &ArrayViewMut<'_, T, R>,
    ) -> Result<Self> {
        output.layout().validate_storage_len(output.data().len())?;
        if output.layout().has_zero_stride_aliasing() {
            return Err(LetoError::StorageError {
                reason:
                    "transposed convolution output layout must not contain zero-stride aliasing"
                        .to_string(),
            });
        }
        Self::validate_readonly(input, weight, bias, parameters, &output.as_view())
    }

    pub(super) fn validate_backward<T, const R: usize>(
        input: &ArrayView<'_, T, R>,
        weight: &ArrayView<'_, T, R>,
        grad_output: &ArrayView<'_, T, R>,
        parameters: TransposedConvolutionParameters<D>,
        grad_input: Option<&ArrayViewMut<'_, T, R>>,
        grad_weight: Option<&ArrayViewMut<'_, T, R>>,
        grad_bias: Option<&ArrayViewMut<'_, T, 1>>,
    ) -> Result<Self> {
        if grad_input.is_none() && grad_weight.is_none() && grad_bias.is_none() {
            return Err(LetoError::InvalidInput(
                "transposed convolution backward requires at least one gradient target".to_string(),
            ));
        }

        let plan = Self::validate_readonly(input, weight, None, parameters, grad_output)?;
        if let Some(target) = grad_input {
            validate_gradient_target(target, input.shape(), "input")?;
        }
        if let Some(target) = grad_weight {
            validate_gradient_target(target, weight.shape(), "weight")?;
        }
        if let Some(target) = grad_bias {
            validate_gradient_target(target, [plan.output_channels], "bias")?;
        }
        Ok(plan)
    }

    fn validate_readonly<T, const R: usize>(
        input: &ArrayView<'_, T, R>,
        weight: &ArrayView<'_, T, R>,
        bias: Option<&ArrayView<'_, T, 1>>,
        parameters: TransposedConvolutionParameters<D>,
        output: &ArrayView<'_, T, R>,
    ) -> Result<Self> {
        input.layout().validate_storage_len(input.data().len())?;
        weight.layout().validate_storage_len(weight.data().len())?;
        output.layout().validate_storage_len(output.data().len())?;
        if let Some(bias) = bias {
            bias.layout().validate_storage_len(bias.data().len())?;
        }
        Self::validate_shapes(input, weight, bias, parameters, output.shape())
    }

    fn validate_shapes<T, const R: usize>(
        input: &ArrayView<'_, T, R>,
        weight: &ArrayView<'_, T, R>,
        bias: Option<&ArrayView<'_, T, 1>>,
        parameters: TransposedConvolutionParameters<D>,
        output_shape: [usize; R],
    ) -> Result<Self> {
        let expected_rank = D.checked_add(2).ok_or(LetoError::Overflow {
            reason: "transposed convolution tensor rank",
        })?;
        if D == 0 || R != expected_rank {
            return Err(LetoError::InvalidInput(format!(
                "transposed convolution rank {R} must equal spatial rank {D} plus batch/channel axes"
            )));
        }

        let input_shape = input.shape();
        let weight_shape = weight.shape();
        let batch = input_shape[0];
        let input_channels = input_shape[1];
        let output_channels = weight_shape[1];
        if weight_shape[0] != input_channels
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
            if input_extent == 0 || kernel_extent == 0 {
                return Err(LetoError::InvalidInput(
                    "transposed convolution spatial and kernel extents must be nonzero".to_string(),
                ));
            }
            let expanded_input = (input_extent - 1)
                .checked_mul(parameters.stride()[axis])
                .ok_or(LetoError::Overflow {
                    reason: "transposed convolution expanded input extent",
                })?;
            let effective_kernel = (kernel_extent - 1)
                .checked_mul(parameters.dilation()[axis])
                .and_then(|extent| extent.checked_add(1))
                .ok_or(LetoError::Overflow {
                    reason: "transposed convolution effective kernel extent",
                })?;
            let unpadded_output = expanded_input
                .checked_add(effective_kernel)
                .and_then(|extent| extent.checked_add(parameters.output_padding()[axis]))
                .ok_or(LetoError::Overflow {
                    reason: "transposed convolution unpadded output extent",
                })?;
            let total_padding =
                parameters.padding()[axis]
                    .checked_mul(2)
                    .ok_or(LetoError::Overflow {
                        reason: "transposed convolution total padding",
                    })?;
            let output_extent = unpadded_output.checked_sub(total_padding).ok_or_else(|| {
                LetoError::InvalidInput(
                    "transposed convolution padding exceeds the generated extent".to_string(),
                )
            })?;
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

        Ok(Self {
            batch,
            output_channels,
            input_spatial,
            kernel_spatial,
            output_spatial,
            parameters,
            input_elements: checked_elements(
                &input_shape,
                "transposed convolution input element count",
            )?,
            input_spatial_elements: checked_elements(
                &input_spatial,
                "transposed convolution input spatial element count",
            )?,
            weight_elements: checked_elements(
                &weight_shape,
                "transposed convolution weight element count",
            )?,
            output_elements: checked_elements(
                &output_shape,
                "transposed convolution output element count",
            )?,
            output_spatial_elements: checked_elements(
                &output_spatial,
                "transposed convolution output spatial element count",
            )?,
            kernel_elements: checked_elements(
                &kernel_spatial,
                "transposed convolution kernel element count",
            )?,
        })
    }
}

fn validate_gradient_target<T, const R: usize>(
    target: &ArrayViewMut<'_, T, R>,
    expected_shape: [usize; R],
    target_name: &str,
) -> Result<()> {
    target.layout().validate_storage_len(target.data().len())?;
    if target.layout().has_zero_stride_aliasing() {
        return Err(LetoError::StorageError {
            reason: format!(
                "transposed convolution {target_name} gradient layout must not contain zero-stride aliasing"
            ),
        });
    }
    if target.shape() != expected_shape {
        return Err(LetoError::ShapeMismatch {
            lhs: target.shape().to_vec(),
            rhs: expected_shape.to_vec(),
        });
    }
    Ok(())
}

fn checked_elements(shape: &[usize], reason: &'static str) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, &extent| {
        count
            .checked_mul(extent)
            .ok_or(LetoError::Overflow { reason })
    })
}
