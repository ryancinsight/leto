use crate::application::index::validate_mutable_output;
use leto::{ArrayView, ArrayViewMut, ConvolutionParameters, LetoError, Result};

pub(super) struct ConvolutionPlan<const D: usize> {
    pub(super) batch: usize,
    pub(super) input_channels: usize,
    pub(super) output_channels: usize,
    pub(super) input_spatial: [usize; D],
    pub(super) kernel_spatial: [usize; D],
    pub(super) output_spatial: [usize; D],
    pub(super) parameters: ConvolutionParameters<D>,
    pub(super) input_elements: usize,
    pub(super) weight_elements: usize,
    pub(super) output_elements: usize,
    pub(super) output_spatial_elements: usize,
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
        validate_mutable_output(output, "convolution")?;
        Self::validate_readonly(input, weight, bias, parameters, &output.as_view())
    }

    pub(super) fn validate_backward<T, const R: usize>(
        input: &ArrayView<'_, T, R>,
        weight: &ArrayView<'_, T, R>,
        grad_output: &ArrayView<'_, T, R>,
        parameters: ConvolutionParameters<D>,
        grad_input: Option<&ArrayViewMut<'_, T, R>>,
        grad_weight: Option<&ArrayViewMut<'_, T, R>>,
        grad_bias: Option<&ArrayViewMut<'_, T, 1>>,
    ) -> Result<Self> {
        if grad_input.is_none() && grad_weight.is_none() && grad_bias.is_none() {
            return Err(LetoError::InvalidInput(
                "convolution backward requires at least one gradient target".to_string(),
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
        parameters: ConvolutionParameters<D>,
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
        parameters: ConvolutionParameters<D>,
        output_shape: [usize; R],
    ) -> Result<Self> {
        let expected_rank = D.checked_add(2).ok_or(LetoError::Overflow {
            reason: "convolution tensor rank",
        })?;
        if D == 0 || R != expected_rank {
            return Err(LetoError::InvalidInput(format!(
                "convolution rank {R} must equal spatial rank {D} plus batch/channel axes"
            )));
        }

        let input_shape = input.shape();
        let weight_shape = weight.shape();
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

        let input_elements = checked_elements(&input_shape, "convolution input element count")?;
        let weight_elements = checked_elements(&weight_shape, "convolution weight element count")?;
        let output_elements = checked_elements(&output_shape, "convolution output element count")?;
        let output_spatial_elements =
            checked_elements(&output_spatial, "convolution output spatial element count")?;
        let kernel_elements =
            checked_elements(&kernel_spatial, "convolution kernel element count")?;

        Ok(Self {
            batch,
            input_channels,
            output_channels,
            input_spatial,
            kernel_spatial,
            output_spatial,
            parameters,
            input_elements,
            weight_elements,
            output_elements,
            output_spatial_elements,
            kernel_elements,
        })
    }
}

fn validate_gradient_target<T, const R: usize>(
    target: &ArrayViewMut<'_, T, R>,
    expected_shape: [usize; R],
    target_name: &str,
) -> Result<()> {
    validate_mutable_output(target, &format!("convolution {target_name} gradient"))?;
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
