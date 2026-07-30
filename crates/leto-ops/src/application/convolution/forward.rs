use super::plan::{ConvolutionParameters, ConvolutionPlan};
use crate::domain::scalar::Scalar;
use leto::{ArrayView, ArrayViewMut, Result};

/// Compute an N-dimensional cross-correlation into caller-owned storage.
///
/// Inputs use `[batch, input_channels, spatial...]`, weights use
/// `[output_channels, input_channels, kernel...]`, and outputs use
/// `[batch, output_channels, spatial...]`. The implementation is generic over
/// scalar type, tensor rank, and spatial rank; rank-specific call sites
/// monomorphize this single kernel.
///
/// Validation completes before the first output write. Invalid shape, layout,
/// storage, stride, padding, dilation, or element-count contracts therefore
/// leave `output` unchanged.
///
/// # Errors
///
/// Returns [`leto::LetoError`] when any convolution or storage contract is
/// invalid or dimension arithmetic overflows.
pub fn convolution_forward_into<T: Scalar, const R: usize, const D: usize>(
    input: &ArrayView<'_, T, R>,
    weight: &ArrayView<'_, T, R>,
    bias: Option<&ArrayView<'_, T, 1>>,
    parameters: ConvolutionParameters<D>,
    output: &mut ArrayViewMut<'_, T, R>,
) -> Result<()> {
    let plan = ConvolutionPlan::validate(input, weight, bias, parameters, output)?;
    if plan.output_elements == 0 {
        return Ok(());
    }

    let mut output_index = [0; R];
    let mut input_index = [0; R];
    let mut weight_index = [0; R];
    let mut output_spatial = [0; D];
    let mut kernel_spatial = [0; D];

    for flat_output in 0..plan.output_elements {
        decode_index(flat_output, output.shape(), &mut output_index);
        output_spatial.copy_from_slice(&output_index[2..]);
        let batch = output_index[0];
        let output_channel = output_index[1];
        let mut sum = bias.map_or(T::ZERO, |bias| bias[output_channel]);

        for input_channel in 0..plan.input_channels {
            for flat_kernel in 0..plan.kernel_elements {
                decode_index(flat_kernel, plan.kernel_spatial, &mut kernel_spatial);
                let mut inside = true;
                for axis in 0..D {
                    // Validation proves the largest padded position is at most
                    // the checked padded input extent minus one.
                    let padded_position = output_spatial[axis] * plan.parameters.stride()[axis]
                        + kernel_spatial[axis] * plan.parameters.dilation()[axis];
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
                    weight_index[axis + 2] = kernel_spatial[axis];
                }
                if inside {
                    input_index[0] = batch;
                    input_index[1] = input_channel;
                    weight_index[0] = output_channel;
                    weight_index[1] = input_channel;
                    sum += input[input_index] * weight[weight_index];
                }
            }
        }
        output[output_index] = sum;
    }
    Ok(())
}

#[inline]
fn decode_index<const N: usize>(mut flat: usize, shape: [usize; N], index: &mut [usize; N]) {
    for axis in (0..N).rev() {
        let extent = shape[axis];
        if extent == 0 {
            index[axis] = 0;
        } else {
            index[axis] = flat % extent;
            flat /= extent;
        }
    }
}
