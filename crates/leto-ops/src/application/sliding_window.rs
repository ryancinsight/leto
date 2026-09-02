//! Generic CPU unfold/fold kernels over Leto views.

use crate::application::index::index_from_flat;
use crate::domain::scalar::Scalar;
use leto::{ArrayView, ArrayViewMut, LetoError, Result, WindowParameters};

use super::window::{
    geometry_from_view, tensor_index, validate_fold_input, validate_mutable_layout,
    validate_output_layout, validate_unfold_output, window_input_coordinate,
};

/// Extract spatial windows into channel-major columns.
///
/// An input of shape `[N, C, spatial...]` becomes `[N, C*K, L]`, where `K` is
/// the kernel volume and `L` is the number of derived window locations. Values
/// outside the padded input are written as zero.
///
/// # Errors
///
/// Returns [`LetoError`] when rank, layout, storage, shape, or window geometry
/// validation fails.
pub fn unfold_into<T: Scalar, const R: usize, const D: usize>(
    input: &ArrayView<'_, T, R>,
    parameters: WindowParameters<D>,
    output: &mut ArrayViewMut<'_, T, 3>,
) -> Result<()> {
    let geometry = geometry_from_view(input, parameters)?;
    validate_unfold_output(output, geometry)?;

    for flat_output in 0..output.layout().checked_size()? {
        let output_index = index_from_flat(flat_output, &output.shape());
        let batch = output_index[0];
        let channel_kernel = output_index[1];
        let location = output_index[2];
        let channel = channel_kernel / geometry.kernel_volume;
        let kernel_flat = channel_kernel % geometry.kernel_volume;
        let output_spatial = index_from_flat(location, &geometry.output_spatial);
        let kernel_spatial = index_from_flat(kernel_flat, parameters.kernel());
        let value = window_input_coordinate(
            output_spatial,
            kernel_spatial,
            geometry.input_spatial,
            parameters,
        )
        .map(|input_spatial| {
            *input
                .get(tensor_index(batch, channel, input_spatial))
                .expect("invariant: validated unfold input index is in bounds")
        })
        .unwrap_or(T::ZERO);
        *output
            .get_mut(output_index)
            .expect("invariant: validated unfold output index is in bounds") = value;
    }
    Ok(())
}

/// Accumulate channel-major columns into a spatial output.
///
/// The output is zeroed before accumulation. Overlapping windows therefore
/// implement the adjoint of [`unfold_into`] with deterministic serial writes.
///
/// # Errors
///
/// Returns [`LetoError`] when rank, layout, storage, shape, or window geometry
/// validation fails.
pub fn fold_into<T: Scalar, const R: usize, const D: usize>(
    input: &ArrayView<'_, T, 3>,
    output_spatial_shape: [usize; D],
    parameters: WindowParameters<D>,
    output: &mut ArrayViewMut<'_, T, R>,
) -> Result<()> {
    validate_output_layout(input, "fold input")?;
    validate_mutable_layout(output, "fold output")?;
    let output_array = output.as_view();
    let geometry = geometry_from_view(&output_array, parameters)?;
    if geometry.input_spatial != output_spatial_shape {
        return Err(LetoError::ShapeMismatch {
            lhs: output_spatial_shape.to_vec(),
            rhs: geometry.input_spatial.to_vec(),
        });
    }
    validate_fold_input(
        input,
        geometry.input_shape[0],
        geometry.input_shape[1],
        geometry.kernel_volume,
        geometry.output_locations,
    )?;

    for flat_output in 0..output.layout().checked_size()? {
        let output_index = index_from_flat(flat_output, &output.shape());
        *output
            .get_mut(output_index)
            .expect("invariant: validated fold output index is in bounds") = T::ZERO;
    }

    for flat_input in 0..input.layout().checked_size()? {
        let input_index = index_from_flat(flat_input, &input.shape());
        let batch = input_index[0];
        let channel_kernel = input_index[1];
        let location = input_index[2];
        let channel = channel_kernel / geometry.kernel_volume;
        let kernel_flat = channel_kernel % geometry.kernel_volume;
        let output_spatial = index_from_flat(location, &geometry.output_spatial);
        let kernel_spatial = index_from_flat(kernel_flat, parameters.kernel());
        let Some(target_spatial) = window_input_coordinate(
            output_spatial,
            kernel_spatial,
            geometry.input_spatial,
            parameters,
        ) else {
            continue;
        };
        let target = tensor_index(batch, channel, target_spatial);
        let value = *input
            .get(input_index)
            .expect("invariant: validated fold input index is in bounds");
        *output
            .get_mut(target)
            .expect("invariant: validated fold output index is in bounds") += value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{fold_into, unfold_into};
    use leto::{Array, WindowParameters};

    #[test]
    fn unfold_then_fold_matches_the_adjoint_overlap_counts() {
        let input =
            Array::from_shape_vec([1, 1, 4], vec![1_i32, 2, 3, 4]).expect("valid input array");
        let parameters = WindowParameters::new([2], [1], [0], [1])
            .expect("valid one-dimensional window parameters");
        let mut unfolded = Array::from_elem([1, 2, 3], 0_i32);
        unfold_into(&input.view(), parameters, &mut unfolded.view_mut()).expect("unfold succeeds");
        assert_eq!(
            unfolded.view().iter().copied().collect::<Vec<_>>(),
            [1, 2, 3, 2, 3, 4]
        );

        let mut folded = Array::from_elem([1, 1, 4], 0_i32);
        fold_into(&unfolded.view(), [4], parameters, &mut folded.view_mut())
            .expect("fold succeeds");
        assert_eq!(
            folded.view().iter().copied().collect::<Vec<_>>(),
            [1, 4, 6, 4]
        );
    }
}
