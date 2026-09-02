//! Generic CPU pooling kernels over Leto views.

use crate::application::index::index_from_flat;
use crate::domain::scalar::Scalar;
use leto::{ArrayView, ArrayViewMut, LetoError, Result, WindowParameters};

use super::window::{
    geometry_from_shape, geometry_from_view, tensor_index, validate_mutable_layout,
    validate_pool_output, window_input_coordinate,
};

/// Pooling reduction applied to each spatial window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolingMode {
    /// Select the greatest valid value in each window.
    Maximum,
    /// Average the valid values in each window.
    Average,
}

/// Compute a spatial pooling operation into caller-owned storage.
///
/// Inputs use `[batch, channel, spatial...]`; the output preserves the batch
/// and channel axes and replaces the spatial axes with the extents derived by
/// [`WindowParameters`]. Padding contributes no value to the average and an
/// all-padding maximum window writes zero, matching the historical Coeus CPU
/// reference semantics.
///
/// # Errors
///
/// Returns [`LetoError`] when rank, layout, storage, shape, or window geometry
/// validation fails.
pub fn pooling_forward_into<T: Scalar, const R: usize, const D: usize>(
    input: &ArrayView<'_, T, R>,
    parameters: WindowParameters<D>,
    mode: PoolingMode,
    output: &mut ArrayViewMut<'_, T, R>,
) -> Result<()> {
    let geometry = geometry_from_view(input, parameters)?;
    validate_pool_output(output, geometry)?;
    let output_elements = output.layout().checked_size()?;

    for flat_output in 0..output_elements {
        let output_index = index_from_flat(flat_output, &output.shape());
        let batch = output_index[0];
        let channel = output_index[1];
        let mut output_spatial = [0_usize; D];
        output_spatial.copy_from_slice(&output_index[2..]);
        let mut value = None;
        let mut sum = T::ZERO;
        let mut count = 0usize;

        for flat_kernel in 0..geometry.kernel_volume {
            let kernel_spatial = index_from_flat(flat_kernel, parameters.kernel());
            let Some(input_spatial) = window_input_coordinate(
                output_spatial,
                kernel_spatial,
                geometry.input_spatial,
                parameters,
            ) else {
                continue;
            };
            let input_index = tensor_index(batch, channel, input_spatial);
            let input_value = *input
                .get(input_index)
                .expect("invariant: validated pooling input index is in bounds");
            match mode {
                PoolingMode::Maximum => {
                    value = Some(match value {
                        None => input_value,
                        Some(current) if input_value > current => input_value,
                        Some(current) => current,
                    });
                }
                PoolingMode::Average => {
                    sum += input_value;
                    count = count
                        .checked_add(1)
                        .expect("invariant: window count is bounded");
                }
            }
        }

        let pooled = match mode {
            PoolingMode::Maximum => value.unwrap_or(T::ZERO),
            PoolingMode::Average => {
                if count == 0 {
                    T::ZERO
                } else {
                    sum / T::from_usize(count)
                }
            }
        };
        *output
            .get_mut(output_index)
            .expect("invariant: validated pooling output index is in bounds") = pooled;
    }
    Ok(())
}

/// Accumulate the gradient of a spatial pooling operation.
///
/// The maximum path recomputes the first maximum selected by the forward
/// operation and therefore requires `input`. The average path distributes
/// each output gradient over valid input points and can derive its geometry
/// from `grad_input`, so it accepts `None` for `input`. Both paths are serial
/// gathers/scatters over validated injective output storage, so overlapping
/// windows accumulate deterministically.
///
/// # Errors
///
/// Returns [`LetoError`] when rank, layout, storage, shape, or window geometry
/// validation fails.
pub fn pooling_backward_accumulate<T: Scalar, const R: usize, const D: usize>(
    grad_output: &ArrayView<'_, T, R>,
    input: Option<&ArrayView<'_, T, R>>,
    parameters: WindowParameters<D>,
    mode: PoolingMode,
    grad_input: &mut ArrayViewMut<'_, T, R>,
) -> Result<()> {
    let geometry = match input {
        Some(input) => geometry_from_view(input, parameters)?,
        None if matches!(mode, PoolingMode::Average) => {
            validate_mutable_layout(grad_input, "pooling gradient")?;
            geometry_from_shape(grad_input.shape(), parameters)?
        }
        None => {
            return Err(LetoError::InvalidInput(
                "maximum pooling backward requires the forward input".to_owned(),
            ));
        }
    };
    super::window::validate_output_layout(grad_output, "pooling gradient")?;
    super::window::validate_mutable_layout(grad_input, "pooling gradient")?;
    let grad_output_shape = grad_output.shape();
    if grad_output.shape()[0] != geometry.input_shape[0]
        || grad_output.shape()[1] != geometry.input_shape[1]
        || !grad_output_shape[2..]
            .iter()
            .eq(geometry.output_spatial.iter())
    {
        let mut expected = [0; R];
        expected[0] = geometry.input_shape[0];
        expected[1] = geometry.input_shape[1];
        expected[2..].copy_from_slice(&geometry.output_spatial);
        return Err(LetoError::ShapeMismatch {
            lhs: grad_output.shape().to_vec(),
            rhs: expected.to_vec(),
        });
    }
    if grad_input.shape() != geometry.input_shape {
        return Err(LetoError::ShapeMismatch {
            lhs: grad_input.shape().to_vec(),
            rhs: geometry.input_shape.to_vec(),
        });
    }

    let output_elements = grad_output.layout().checked_size()?;
    for flat_output in 0..output_elements {
        let output_index = index_from_flat(flat_output, &grad_output.shape());
        let batch = output_index[0];
        let channel = output_index[1];
        let mut output_spatial = [0_usize; D];
        output_spatial.copy_from_slice(&output_index[2..]);
        let gradient = *grad_output
            .get(output_index)
            .expect("invariant: validated pooling gradient index is in bounds");
        let mut maximum = None;
        let mut average_count = 0usize;

        for flat_kernel in 0..geometry.kernel_volume {
            let kernel_spatial = index_from_flat(flat_kernel, parameters.kernel());
            let Some(input_spatial) = window_input_coordinate(
                output_spatial,
                kernel_spatial,
                geometry.input_spatial,
                parameters,
            ) else {
                continue;
            };
            match mode {
                PoolingMode::Maximum => {
                    let input = match input {
                        Some(input) => input,
                        None => {
                            return Err(LetoError::InvalidInput(
                                "maximum pooling backward requires the forward input".to_owned(),
                            ));
                        }
                    };
                    let input_index = tensor_index(batch, channel, input_spatial);
                    let input_value = *input
                        .get(input_index)
                        .expect("invariant: validated pooling input index is in bounds");
                    let replace = match maximum {
                        None => true,
                        Some((current, _)) => input_value > current,
                    };
                    if replace {
                        maximum = Some((input_value, input_spatial));
                    }
                }
                PoolingMode::Average => {
                    average_count = average_count
                        .checked_add(1)
                        .expect("invariant: window count is bounded");
                }
            }
        }

        match mode {
            PoolingMode::Maximum => {
                if let Some((_, input_spatial)) = maximum {
                    let target = tensor_index(batch, channel, input_spatial);
                    *grad_input
                        .get_mut(target)
                        .expect("invariant: validated pooling gradient target is in bounds") +=
                        gradient;
                }
            }
            PoolingMode::Average => {
                if average_count == 0 {
                    continue;
                }
                let share = gradient / T::from_usize(average_count);
                for flat_kernel in 0..geometry.kernel_volume {
                    let kernel_spatial = index_from_flat(flat_kernel, parameters.kernel());
                    let Some(input_spatial) = window_input_coordinate(
                        output_spatial,
                        kernel_spatial,
                        geometry.input_spatial,
                        parameters,
                    ) else {
                        continue;
                    };
                    let target = tensor_index(batch, channel, input_spatial);
                    *grad_input
                        .get_mut(target)
                        .expect("invariant: validated pooling gradient target is in bounds") +=
                        share;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{PoolingMode, pooling_backward_accumulate, pooling_forward_into};
    use leto::{Array, WindowParameters};

    fn parameters() -> WindowParameters<2> {
        WindowParameters::new([2, 2], [1, 1], [0, 0], [1, 1])
            .expect("valid two-dimensional pooling parameters")
    }

    #[test]
    fn maximum_forward_selects_each_window_maximum() {
        let input =
            Array::from_shape_vec([1, 1, 3, 3], (1..=9).collect()).expect("valid input array");
        let mut output = Array::from_elem([1, 1, 2, 2], 0);
        pooling_forward_into(
            &input.view(),
            parameters(),
            PoolingMode::Maximum,
            &mut output.view_mut(),
        )
        .expect("maximum pooling succeeds");
        assert_eq!(
            output.view().iter().copied().collect::<Vec<_>>(),
            [5, 6, 8, 9]
        );
    }

    #[test]
    fn average_backward_distributes_overlap_and_preserves_accumulation() {
        let input = Array::from_shape_vec([1, 1, 3, 3], (1..=9).map(f64::from).collect())
            .expect("valid input array");
        let grad_output = Array::from_elem([1, 1, 2, 2], 1.0_f64);
        let mut grad_input = Array::from_elem([1, 1, 3, 3], 0.0_f64);
        pooling_backward_accumulate(
            &grad_output.view(),
            Some(&input.view()),
            parameters(),
            PoolingMode::Average,
            &mut grad_input.view_mut(),
        )
        .expect("average pooling backward succeeds");
        assert_eq!(
            grad_input.view().iter().copied().collect::<Vec<_>>(),
            [0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25]
        );
    }
}
