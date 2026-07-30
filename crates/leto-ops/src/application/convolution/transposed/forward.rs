use super::parameters::TransposedConvolutionParameters;
use super::plan::TransposedConvolutionPlan;
use crate::application::convolution::coordinates::decode_index;
use crate::domain::scalar::Scalar;
use leto::{ArrayView, ArrayViewMut, Result};

/// Compute an N-dimensional transposed convolution into caller-owned storage.
///
/// Inputs use `[batch, input_channels, spatial...]`, weights use
/// `[input_channels, output_channels, kernel...]`, and outputs use
/// `[batch, output_channels, spatial...]`. Each input value is scattered across
/// the output positions selected by stride, padding, and dilation.
/// `output_padding` changes only the validated output shape.
///
/// The generic kernel monomorphizes once per scalar, tensor rank, and spatial
/// rank. It borrows all inputs, writes caller-provided output storage, and uses
/// fixed-size index arrays, so the compute path performs no allocation.
///
/// Validation completes before the first output write. Invalid shape, layout,
/// storage, or dimension arithmetic therefore leaves `output` unchanged.
///
/// # Errors
///
/// Returns [`leto::LetoError`] when any transposed-convolution or storage
/// contract is invalid or dimension arithmetic overflows.
pub fn convolution_transposed_forward_into<T: Scalar, const R: usize, const D: usize>(
    input: &ArrayView<'_, T, R>,
    weight: &ArrayView<'_, T, R>,
    bias: Option<&ArrayView<'_, T, 1>>,
    parameters: TransposedConvolutionParameters<D>,
    output: &mut ArrayViewMut<'_, T, R>,
) -> Result<()> {
    let plan = TransposedConvolutionPlan::validate(input, weight, bias, parameters, output)?;

    let mut output_index = [0; R];
    let output_shape = output.shape();
    for flat_output in 0..plan.output_elements {
        decode_index(flat_output, &output_shape, &mut output_index);
        output[output_index] = bias.map_or(T::ZERO, |bias| bias[output_index[1]]);
    }

    let mut input_index = [0; R];
    let mut weight_index = [0; R];
    let mut kernel_spatial = [0; D];
    let input_shape = input.shape();
    for flat_input in 0..plan.input_elements {
        decode_index(flat_input, &input_shape, &mut input_index);
        let batch = input_index[0];
        let input_channel = input_index[1];
        for output_channel in 0..plan.output_channels {
            weight_index[0] = input_channel;
            weight_index[1] = output_channel;
            for flat_kernel in 0..plan.kernel_elements {
                decode_index(flat_kernel, &plan.kernel_spatial, &mut kernel_spatial);
                let mut inside = true;
                for axis in 0..D {
                    // Plan validation proves both products and their sum fit
                    // `usize` at the largest input and kernel coordinates.
                    let padded_position = input_index[axis + 2] * plan.parameters.stride()[axis]
                        + kernel_spatial[axis] * plan.parameters.dilation()[axis];
                    if padded_position < plan.parameters.padding()[axis] {
                        inside = false;
                        break;
                    }
                    let output_position = padded_position - plan.parameters.padding()[axis];
                    if output_position >= plan.output_spatial[axis] {
                        inside = false;
                        break;
                    }
                    output_index[axis + 2] = output_position;
                    weight_index[axis + 2] = kernel_spatial[axis];
                }
                if inside {
                    output_index[0] = batch;
                    output_index[1] = output_channel;
                    output[output_index] += input[input_index] * weight[weight_index];
                }
            }
        }
    }
    Ok(())
}
