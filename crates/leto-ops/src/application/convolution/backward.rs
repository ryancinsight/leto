use super::coordinates::decode_index;
use super::plan::{ConvolutionParameters, ConvolutionPlan};
use crate::domain::scalar::Scalar;
use leto::{ArrayView, ArrayViewMut, Result};

/// Accumulate an N-dimensional convolution backward pass into selected gradients.
///
/// `input`, `weight`, and `grad_output` follow the same layouts as
/// [`super::convolution_forward_into`]. Each requested gradient target must
/// exactly match its corresponding source shape. Existing target values are
/// preserved and the computed gradient is added to them.
///
/// Validation of every requested target completes before the first write.
/// The implementation uses fixed-size coordinate arrays and performs no heap
/// allocation after successful validation.
///
/// # Errors
///
/// Returns [`leto::LetoError`] when no target is requested, a shape or storage
/// contract is invalid, a mutable target aliases through a zero stride, or
/// dimension arithmetic overflows.
#[allow(clippy::too_many_arguments)]
pub fn convolution_backward_accumulate<T: Scalar, const R: usize, const D: usize>(
    input: &ArrayView<'_, T, R>,
    weight: &ArrayView<'_, T, R>,
    grad_output: &ArrayView<'_, T, R>,
    parameters: ConvolutionParameters<D>,
    mut grad_input: Option<&mut ArrayViewMut<'_, T, R>>,
    mut grad_weight: Option<&mut ArrayViewMut<'_, T, R>>,
    mut grad_bias: Option<&mut ArrayViewMut<'_, T, 1>>,
) -> Result<()> {
    let plan = ConvolutionPlan::validate_backward(
        input,
        weight,
        grad_output,
        parameters,
        grad_input.as_deref(),
        grad_weight.as_deref(),
        grad_bias.as_deref(),
    )?;

    if let Some(target) = grad_input.as_mut() {
        accumulate_input_gradient(input, weight, grad_output, &plan, target);
    }
    if let Some(target) = grad_weight.as_mut() {
        accumulate_weight_gradient(input, grad_output, &plan, target);
    }
    if let Some(target) = grad_bias.as_mut() {
        accumulate_bias_gradient(grad_output, &plan, target);
    }
    Ok(())
}

fn accumulate_input_gradient<T: Scalar, const R: usize, const D: usize>(
    input: &ArrayView<'_, T, R>,
    weight: &ArrayView<'_, T, R>,
    grad_output: &ArrayView<'_, T, R>,
    plan: &ConvolutionPlan<D>,
    grad_input: &mut ArrayViewMut<'_, T, R>,
) {
    let input_shape = input.shape();
    let mut input_index = [0; R];
    let mut output_index = [0; R];
    let mut weight_index = [0; R];
    let mut kernel_index = [0; D];

    for flat_input in 0..plan.input_elements {
        decode_index(flat_input, &input_shape, &mut input_index);
        let mut sum = T::ZERO;
        for output_channel in 0..plan.output_channels {
            for flat_kernel in 0..plan.kernel_elements {
                decode_index(flat_kernel, &plan.kernel_spatial, &mut kernel_index);
                let mut contributes = true;
                for axis in 0..D {
                    let padded_input = input_index[axis + 2] + plan.parameters.padding()[axis];
                    let kernel_position = kernel_index[axis] * plan.parameters.dilation()[axis];
                    if padded_input < kernel_position {
                        contributes = false;
                        break;
                    }
                    let output_numerator = padded_input - kernel_position;
                    if !output_numerator.is_multiple_of(plan.parameters.stride()[axis]) {
                        contributes = false;
                        break;
                    }
                    let output_position = output_numerator / plan.parameters.stride()[axis];
                    if output_position >= plan.output_spatial[axis] {
                        contributes = false;
                        break;
                    }
                    output_index[axis + 2] = output_position;
                    weight_index[axis + 2] = kernel_index[axis];
                }
                if contributes {
                    output_index[0] = input_index[0];
                    output_index[1] = output_channel;
                    weight_index[0] = output_channel;
                    weight_index[1] = input_index[1];
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
    plan: &ConvolutionPlan<D>,
    grad_weight: &mut ArrayViewMut<'_, T, R>,
) {
    let weight_shape = grad_weight.shape();
    let mut weight_index = [0; R];
    let mut input_index = [0; R];
    let mut output_index = [0; R];
    let mut output_spatial = [0; D];

    for flat_weight in 0..plan.weight_elements {
        decode_index(flat_weight, &weight_shape, &mut weight_index);
        let mut sum = T::ZERO;
        for batch in 0..plan.batch {
            for flat_output in 0..plan.output_spatial_elements {
                decode_index(flat_output, &plan.output_spatial, &mut output_spatial);
                let mut inside = true;
                for axis in 0..D {
                    let padded_position = output_spatial[axis] * plan.parameters.stride()[axis]
                        + weight_index[axis + 2] * plan.parameters.dilation()[axis];
                    if padded_position < plan.parameters.padding()[axis] {
                        inside = false;
                        break;
                    }
                    let input_position = padded_position - plan.parameters.padding()[axis];
                    if input_position >= plan.input_spatial[axis] {
                        inside = false;
                        break;
                    }
                    input_index[axis + 2] = input_position;
                    output_index[axis + 2] = output_spatial[axis];
                }
                if inside {
                    input_index[0] = batch;
                    input_index[1] = weight_index[1];
                    output_index[0] = batch;
                    output_index[1] = weight_index[0];
                    sum += input[input_index] * grad_output[output_index];
                }
            }
        }
        grad_weight[weight_index] += sum;
    }
}

fn accumulate_bias_gradient<T: Scalar, const R: usize, const D: usize>(
    grad_output: &ArrayView<'_, T, R>,
    plan: &ConvolutionPlan<D>,
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
