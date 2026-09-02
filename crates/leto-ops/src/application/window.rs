//! Shared validation and indexing for spatial window operations.

use crate::application::index::validate_mutable_output;
use leto::{ArrayView, ArrayViewMut, LetoError, Result, WindowParameters};

/// Validated geometry shared by pooling and unfold/fold kernels.
#[derive(Clone, Copy, Debug)]
pub(super) struct WindowGeometry<const R: usize, const D: usize> {
    pub(super) input_shape: [usize; R],
    pub(super) input_spatial: [usize; D],
    pub(super) output_spatial: [usize; D],
    pub(super) kernel_volume: usize,
    pub(super) output_locations: usize,
}

pub(super) fn validate_readonly<T, const N: usize>(
    view: &ArrayView<'_, T, N>,
    role: &'static str,
) -> Result<()> {
    view.layout()
        .validate_storage_len(view.data().len())
        .map_err(|source| LetoError::StorageError {
            reason: format!("{role} layout is invalid: {source}"),
        })
}

pub(super) fn geometry_from_view<T, const R: usize, const D: usize>(
    input: &ArrayView<'_, T, R>,
    parameters: WindowParameters<D>,
) -> Result<WindowGeometry<R, D>> {
    validate_readonly(input, "window input")?;
    geometry_from_shape(input.shape(), parameters)
}

pub(super) fn geometry_from_shape<const R: usize, const D: usize>(
    input_shape: [usize; R],
    parameters: WindowParameters<D>,
) -> Result<WindowGeometry<R, D>> {
    let expected_rank = D.checked_add(2).ok_or(LetoError::Overflow {
        reason: "window tensor rank",
    })?;
    if D == 0 || R != expected_rank {
        return Err(LetoError::InvalidInput(format!(
            "window tensor rank {R} must equal spatial rank {D} plus batch/channel axes"
        )));
    }

    let mut input_spatial = [0_usize; D];
    input_spatial.copy_from_slice(&input_shape[2..]);
    let output_spatial = parameters.output_shape(input_spatial)?;
    let output_locations = output_spatial.iter().try_fold(1_usize, |count, &extent| {
        count.checked_mul(extent).ok_or(LetoError::Overflow {
            reason: "window output location count",
        })
    })?;
    Ok(WindowGeometry {
        input_shape,
        input_spatial,
        output_spatial,
        kernel_volume: parameters.kernel_volume()?,
        output_locations,
    })
}

pub(super) fn validate_pool_output<T, const R: usize, const D: usize>(
    output: &mut ArrayViewMut<'_, T, R>,
    geometry: WindowGeometry<R, D>,
) -> Result<()> {
    validate_mutable_output(output, "pooling")?;
    let mut expected = [0; R];
    expected[0] = geometry.input_shape[0];
    expected[1] = geometry.input_shape[1];
    expected[2..].copy_from_slice(&geometry.output_spatial);
    if output.shape() != expected {
        return Err(LetoError::ShapeMismatch {
            lhs: output.shape().to_vec(),
            rhs: expected.to_vec(),
        });
    }
    Ok(())
}

pub(super) fn validate_unfold_output<T, const R: usize, const D: usize>(
    output: &mut ArrayViewMut<'_, T, 3>,
    geometry: WindowGeometry<R, D>,
) -> Result<()> {
    validate_mutable_output(output, "unfold")?;
    let channels = geometry.input_shape[1]
        .checked_mul(geometry.kernel_volume)
        .ok_or(LetoError::Overflow {
            reason: "unfold channel-kernel count",
        })?;
    let expected = [geometry.input_shape[0], channels, geometry.output_locations];
    if output.shape() != expected {
        return Err(LetoError::ShapeMismatch {
            lhs: output.shape().to_vec(),
            rhs: expected.to_vec(),
        });
    }
    Ok(())
}

pub(super) fn validate_fold_input<T>(
    input: &ArrayView<'_, T, 3>,
    batch: usize,
    channels: usize,
    kernel_volume: usize,
    output_locations: usize,
) -> Result<()> {
    let expected_channels = channels
        .checked_mul(kernel_volume)
        .ok_or(LetoError::Overflow {
            reason: "fold channel-kernel count",
        })?;
    let expected = [batch, expected_channels, output_locations];
    if input.shape() != expected {
        return Err(LetoError::ShapeMismatch {
            lhs: input.shape().to_vec(),
            rhs: expected.to_vec(),
        });
    }
    Ok(())
}

pub(super) fn tensor_index<const R: usize, const D: usize>(
    batch: usize,
    channel: usize,
    spatial: [usize; D],
) -> [usize; R] {
    let mut index = [0; R];
    index[0] = batch;
    index[1] = channel;
    index[2..].copy_from_slice(&spatial);
    index
}

pub(super) fn window_input_coordinate<const D: usize>(
    output_spatial: [usize; D],
    kernel_spatial: [usize; D],
    input_spatial: [usize; D],
    parameters: WindowParameters<D>,
) -> Option<[usize; D]> {
    let mut input = [0_usize; D];
    for axis in 0..D {
        let traversed = output_spatial[axis]
            .checked_mul(parameters.stride()[axis])?
            .checked_add(kernel_spatial[axis].checked_mul(parameters.dilation()[axis])?)?;
        let padding = parameters.padding()[axis];
        let coordinate = traversed.checked_sub(padding)?;
        if coordinate >= input_spatial[axis] {
            return None;
        }
        input[axis] = coordinate;
    }
    Some(input)
}

pub(super) fn validate_output_layout<T, const N: usize>(
    view: &ArrayView<'_, T, N>,
    role: &'static str,
) -> Result<()> {
    validate_readonly(view, role)
}

pub(super) fn validate_mutable_layout<T, const N: usize>(
    view: &mut ArrayViewMut<'_, T, N>,
    role: &'static str,
) -> Result<()> {
    validate_mutable_output(view, role)
}
