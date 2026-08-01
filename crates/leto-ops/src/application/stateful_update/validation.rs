use leto::{ArrayView, ArrayViewMut, LetoError, Result};

fn validate_output<T, const N: usize>(
    output: &ArrayViewMut<'_, T, N>,
    expected_shape: [usize; N],
) -> Result<()> {
    output.layout().validate_storage_len(output.data().len())?;
    if output.shape() != expected_shape {
        return Err(LetoError::ShapeMismatch {
            lhs: expected_shape.to_vec(),
            rhs: output.shape().to_vec(),
        });
    }
    if !output.layout().is_injective()? {
        return Err(LetoError::StorageError {
            reason: "stateful-update output layout must be injective".to_string(),
        });
    }
    Ok(())
}

pub(super) fn validate_one<T, const N: usize>(
    parameter: &ArrayViewMut<'_, T, N>,
    gradient: &ArrayView<'_, T, N>,
    state: &ArrayViewMut<'_, T, N>,
) -> Result<()> {
    gradient
        .layout()
        .validate_storage_len(gradient.data().len())?;
    let shape = gradient.shape();
    validate_output(parameter, shape)?;
    validate_output(state, shape)
}

pub(super) fn validate_two<T, const N: usize>(
    parameter: &ArrayViewMut<'_, T, N>,
    gradient: &ArrayView<'_, T, N>,
    first: &ArrayViewMut<'_, T, N>,
    second: &ArrayViewMut<'_, T, N>,
) -> Result<()> {
    validate_one(parameter, gradient, first)?;
    validate_output(second, gradient.shape())
}
