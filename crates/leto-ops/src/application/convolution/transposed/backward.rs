use super::parameters::TransposedConvolutionParameters;
use super::plan::TransposedConvolutionPlan;
use crate::application::convolution::coordinates::decode_index;
use crate::domain::scalar::Scalar;
use leto::{ArrayView, ArrayViewMut, Result};

/// Mutable targets selected for a transposed-convolution backward pass.
///
/// Existing target values are preserved and the computed gradients are added
/// to them. At least one target must be present.
pub struct TransposedConvolutionGradients<'target, 'view, T, const R: usize> {
    grad_input: Option<&'target mut ArrayViewMut<'view, T, R>>,
    grad_weight: Option<&'target mut ArrayViewMut<'view, T, R>>,
    grad_bias: Option<&'target mut ArrayViewMut<'view, T, 1>>,
}

impl<'target, 'view, T, const R: usize> TransposedConvolutionGradients<'target, 'view, T, R> {
    /// Construct a selected set of additive gradient targets.
    #[must_use]
    pub const fn new(
        grad_input: Option<&'target mut ArrayViewMut<'view, T, R>>,
        grad_weight: Option<&'target mut ArrayViewMut<'view, T, R>>,
        grad_bias: Option<&'target mut ArrayViewMut<'view, T, 1>>,
    ) -> Self {
        Self {
            grad_input,
            grad_weight,
            grad_bias,
        }
    }
}

/// Accumulate an N-dimensional transposed-convolution backward pass.
///
/// `input`, `weight`, and `grad_output` follow
/// [`super::convolution_transposed_forward_into`]. Each selected target must
/// exactly match its corresponding source shape. The kernel monomorphizes over
/// scalar, tensor rank, and spatial rank and uses fixed-size coordinate arrays,
/// so successful execution performs no heap allocation.
///
/// Validation of all selected targets completes before the first write.
///
/// # Errors
///
/// Returns [`leto::LetoError`] when no target is selected, a shape or storage
/// contract is invalid, a mutable target aliases through a zero stride, or
/// dimension arithmetic overflows.
pub fn convolution_transposed_backward_accumulate<T: Scalar, const R: usize, const D: usize>(
    input: &ArrayView<'_, T, R>,
    weight: &ArrayView<'_, T, R>,
    grad_output: &ArrayView<'_, T, R>,
    parameters: TransposedConvolutionParameters<D>,
    mut gradients: TransposedConvolutionGradients<'_, '_, T, R>,
) -> Result<()> {
    let plan = TransposedConvolutionPlan::validate_backward(
        input,
        weight,
        grad_output,
        parameters,
        gradients.grad_input.as_deref(),
        gradients.grad_weight.as_deref(),
        gradients.grad_bias.as_deref(),
    )?;

    if let Some(target) = gradients.grad_input.as_mut() {
        accumulate_input_gradient(weight, grad_output, &plan, target);
    }
    if let Some(target) = gradients.grad_weight.as_mut() {
        accumulate_weight_gradient(input, grad_output, &plan, target);
    }
    if let Some(target) = gradients.grad_bias.as_mut() {
        accumulate_bias_gradient(grad_output, &plan, target);
    }
    Ok(())
}

fn accumulate_input_gradient<T: Scalar, const R: usize, const D: usize>(
    weight: &ArrayView<'_, T, R>,
    grad_output: &ArrayView<'_, T, R>,
    plan: &TransposedConvolutionPlan<D>,
    grad_input: &mut ArrayViewMut<'_, T, R>,
) {
    let mut input_index = [0; R];
    let mut weight_index = [0; R];
    let mut output_index = [0; R];
    let mut kernel_spatial = [0; D];

    for flat_input in 0..plan.input_elements {
        decode_index(flat_input, &grad_input.shape(), &mut input_index);
        let mut sum = T::ZERO;
        for output_channel in 0..plan.output_channels {
            weight_index[0] = input_index[1];
            weight_index[1] = output_channel;
            for flat_kernel in 0..plan.kernel_elements {
                decode_index(flat_kernel, &plan.kernel_spatial, &mut kernel_spatial);
                if map_output_position(
                    &input_index,
                    &kernel_spatial,
                    plan,
                    &mut output_index,
                    &mut weight_index,
                ) {
                    output_index[0] = input_index[0];
                    output_index[1] = output_channel;
                    sum += grad_output[output_index] * weight[weight_index];
                }
            }
        }
        grad_input[input_index] += sum;
    }
}

fn accumulate_weight_gradient<T: Scalar, const R: usize, const D: usize>(
    input: &ArrayView<'_, T, R>,
    grad_output: &ArrayView<'_, T, R>,
    plan: &TransposedConvolutionPlan<D>,
    grad_weight: &mut ArrayViewMut<'_, T, R>,
) {
    let mut weight_index = [0; R];
    let mut input_index = [0; R];
    let mut output_index = [0; R];
    let mut kernel_spatial = [0; D];
    let mut input_spatial = [0; D];

    for flat_weight in 0..plan.weight_elements {
        decode_index(flat_weight, &grad_weight.shape(), &mut weight_index);
        kernel_spatial.copy_from_slice(&weight_index[2..]);
        let mut sum = T::ZERO;
        for batch in 0..plan.batch {
            input_index[0] = batch;
            input_index[1] = weight_index[0];
            for flat_input in 0..plan.input_spatial_elements {
                decode_index(flat_input, &plan.input_spatial, &mut input_spatial);
                input_index[2..].copy_from_slice(&input_spatial);
                if map_output_position(
                    &input_index,
                    &kernel_spatial,
                    plan,
                    &mut output_index,
                    &mut weight_index,
                ) {
                    output_index[0] = batch;
                    output_index[1] = weight_index[1];
                    sum += input[input_index] * grad_output[output_index];
                }
            }
        }
        grad_weight[weight_index] += sum;
    }
}

fn accumulate_bias_gradient<T: Scalar, const R: usize, const D: usize>(
    grad_output: &ArrayView<'_, T, R>,
    plan: &TransposedConvolutionPlan<D>,
    grad_bias: &mut ArrayViewMut<'_, T, 1>,
) {
    let mut output_index = [0; R];
    let mut output_spatial = [0; D];
    for output_channel in 0..plan.output_channels {
        let mut sum = T::ZERO;
        for batch in 0..plan.batch {
            for flat_output in 0..plan.output_spatial_elements {
                decode_index(flat_output, &plan.output_spatial, &mut output_spatial);
                output_index[0] = batch;
                output_index[1] = output_channel;
                output_index[2..].copy_from_slice(&output_spatial);
                sum += grad_output[output_index];
            }
        }
        grad_bias[output_channel] += sum;
    }
}

fn map_output_position<const R: usize, const D: usize>(
    input_index: &[usize; R],
    kernel_spatial: &[usize; D],
    plan: &TransposedConvolutionPlan<D>,
    output_index: &mut [usize; R],
    weight_index: &mut [usize; R],
) -> bool {
    for axis in 0..D {
        // Plan validation proves the largest expanded-input and dilated-kernel
        // coordinates, their sum, and output padding fit `usize`.
        let padded_position = input_index[axis + 2] * plan.parameters.stride()[axis]
            + kernel_spatial[axis] * plan.parameters.dilation()[axis];
        if padded_position < plan.parameters.padding()[axis] {
            return false;
        }
        let output_position = padded_position - plan.parameters.padding()[axis];
        if output_position >= plan.output_spatial[axis] {
            return false;
        }
        output_index[axis + 2] = output_position;
        weight_index[axis + 2] = kernel_spatial[axis];
    }
    true
}
