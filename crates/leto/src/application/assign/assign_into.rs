//! Validated assignment kernels: per-element, logical-index, and dense
//! fast paths.

use super::transpose::transpose_copy;
use crate::application::array::{linear_to_index, AssignSource};
use crate::application::view::{ArrayView, ArrayViewMut};
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;

/// Checked per-element assignment for views whose physical window is shared
/// with sibling lane/axis views: every write goes through `get_mut`, never
/// through a whole-window slice that would alias sibling elements.
pub(crate) fn assign_into_elements<T, Rhs, const N: usize>(
    destination: &mut ArrayViewMut<'_, T, N>,
    source: &Rhs,
) -> Result<()>
where
    T: Copy,
    Rhs: AssignSource<T, N>,
{
    let destination_shape = destination.shape();
    let source_shape = source.assign_shape();
    if destination_shape != source_shape {
        return Err(LetoError::ShapeMismatch {
            lhs: destination_shape.to_vec(),
            rhs: source_shape.to_vec(),
        });
    }
    for linear in 0..destination.layout().checked_size()? {
        let index = linear_to_index(linear, destination_shape);
        let value = *source.assign_get(index)?;
        *destination.get_mut(index)? = value;
    }
    Ok(())
}

pub(crate) fn assign_into<T, Rhs, const N: usize>(
    destination_layout: Layout<N>,
    destination_data: &mut [T],
    source: &Rhs,
) -> Result<()>
where
    T: Copy,
    Rhs: AssignSource<T, N>,
{
    let destination_shape = destination_layout.shape();
    let source_shape = source.assign_shape();
    if destination_shape != source_shape {
        return Err(LetoError::ShapeMismatch {
            lhs: destination_shape.to_vec(),
            rhs: source_shape.to_vec(),
        });
    }

    destination_layout.validate_storage_len(destination_data.len())?;
    let source_view = source.assign_view();
    if let Some(source_view) = source_view {
        if source_view.shape() != source_shape {
            return Err(LetoError::ShapeMismatch {
                lhs: destination_shape.to_vec(),
                rhs: source_view.shape().to_vec(),
            });
        }
        source_view
            .layout()
            .validate_storage_len(source_view.data().len())?;
        if destination_layout.is_injective()? {
            return assign_view_into(destination_layout, destination_data, source_view);
        }
    }

    for linear in 0..destination_layout.checked_size()? {
        let index = linear_to_index(linear, destination_shape);
        let value = *source.assign_get(index)?;
        let destination_offset = destination_layout.offset_of(index)?;
        destination_data[destination_offset] = value;
    }
    Ok(())
}

fn assign_view_into<T: Copy, const N: usize>(
    destination_layout: Layout<N>,
    destination_data: &mut [T],
    source: ArrayView<'_, T, N>,
) -> Result<()> {
    let size = destination_layout.checked_size()?;
    if size == 0 {
        return Ok(());
    }

    if destination_layout.is_c_dense() && source.layout().is_c_dense() {
        let source_values = source.as_slice().ok_or_else(|| LetoError::StorageError {
            reason: "assignment source dense range exceeds its storage".to_string(),
        })?;
        let destination_values = dense_destination(&destination_layout, destination_data, size)?;
        destination_values.copy_from_slice(source_values);
        return Ok(());
    }

    if destination_layout.is_f_dense() && source.layout().is_f_dense() {
        // Same shape and both column-major dense: the physical element order
        // is identical, so the dense blocks copy directly at any rank.
        let source_values =
            source
                .as_slice_memory_order()
                .ok_or_else(|| LetoError::StorageError {
                    reason: "assignment source memory-order range exceeds its storage".to_string(),
                })?;
        let destination_values = dense_destination(&destination_layout, destination_data, size)?;
        destination_values.copy_from_slice(source_values);
        return Ok(());
    }

    if N == 2 && destination_layout.is_c_dense() && source.layout().is_f_dense() {
        let source_values =
            source
                .as_slice_memory_order()
                .ok_or_else(|| LetoError::StorageError {
                    reason: "assignment source memory-order range exceeds its storage".to_string(),
                })?;
        let destination_values = dense_destination(&destination_layout, destination_data, size)?;
        return transpose_copy(
            source_values,
            destination_values,
            destination_layout.shape()[1],
            destination_layout.shape()[0],
        );
    }

    if N == 2 && destination_layout.is_f_dense() && source.layout().is_c_dense() {
        // The destination's memory-order block is the row-major transpose of
        // the source, so the row/column arguments swap relative to the
        // C-destination case above.
        let source_values = source.as_slice().ok_or_else(|| LetoError::StorageError {
            reason: "assignment source dense range exceeds its storage".to_string(),
        })?;
        let destination_values = dense_destination(&destination_layout, destination_data, size)?;
        return transpose_copy(
            source_values,
            destination_values,
            destination_layout.shape()[0],
            destination_layout.shape()[1],
        );
    }

    let destination = ArrayViewMut::try_new(destination_layout, destination_data)?;
    for (target, value) in destination.try_iter_mut()?.zip(source.iter()) {
        *target = *value;
    }
    Ok(())
}

fn dense_destination<'a, T, const N: usize>(
    layout: &Layout<N>,
    data: &'a mut [T],
    size: usize,
) -> Result<&'a mut [T]> {
    let start = layout.offset();
    #[expect(
        clippy::unnecessary_lazy_evaluations,
        reason = "Avoid eager LetoError drop on successful arithmetic; ADR 0027"
    )]
    let end = start.checked_add(size).ok_or_else(|| LetoError::Overflow {
        reason: "assignment destination dense range",
    })?;
    data.get_mut(start..end)
        .ok_or_else(|| LetoError::StorageError {
            reason: "assignment destination dense range exceeds its storage".to_string(),
        })
}
