use crate::application::array::{linear_to_index, Array, AssignSource};
use crate::application::view::{ArrayView, ArrayViewMut};
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::infrastructure::storage::StorageMut;

/// Caps the combined source and destination element payload considered by the
/// tile calculation. Cache metadata and occupancy by neighboring data are not
/// part of this payload model.
const TRANSPOSE_PAYLOAD_BUDGET_BYTES: usize = 32 * 1024;

impl<T, S, const N: usize> Array<T, S, N>
where
    S: StorageMut<T>,
{
    /// Assign all elements from another array-like source with the same shape.
    ///
    /// Built-in array and view sources validate storage and destination
    /// storage before mutation. Dense and rank-2 transposed layouts use
    /// allocation-free bulk-copy kernels; other injective layouts use
    /// validated logical iterators. Aliased destinations and external
    /// [`AssignSource`] implementations retain the checked logical-index route.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError`] when shapes differ, either layout exceeds its
    /// storage, or an external source rejects an index.
    #[inline]
    pub fn try_assign<Rhs>(&mut self, rhs: &Rhs) -> Result<()>
    where
        T: Copy,
        Rhs: AssignSource<T, N>,
    {
        assign_into(self.layout, self.storage.as_mut_slice(), rhs)
    }

    /// Assign all elements from another array-like source with the same shape.
    ///
    /// # Panics
    ///
    /// Panics when [`Self::try_assign`] rejects the source or destination.
    #[inline]
    pub fn assign<Rhs>(&mut self, rhs: &Rhs)
    where
        T: Copy,
        Rhs: AssignSource<T, N>,
    {
        self.try_assign(rhs)
            .expect("invariant: assigned arrays have valid compatible layouts");
    }
}

impl<T, const N: usize> ArrayViewMut<'_, T, N> {
    /// Assign all elements from another array-like source with the same shape.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError`] when shapes differ, either layout exceeds its
    /// storage, or an external source rejects an index. Validation completes
    /// before built-in sources mutate the output.
    #[inline]
    pub fn try_assign<Rhs>(&mut self, rhs: &Rhs) -> Result<()>
    where
        T: Copy,
        Rhs: AssignSource<T, N>,
    {
        if self.window_shared {
            // Iterator-yielded interleaved sub-views must not materialize
            // their window as a slice (it contains sibling views' elements);
            // assign through checked per-element access instead.
            return assign_into_elements(self, rhs);
        }
        let layout = self.layout;
        assign_into(layout, self.data_mut(), rhs)
    }

    /// Assign all elements from another array-like source with the same shape.
    ///
    /// # Panics
    ///
    /// Panics when [`Self::try_assign`] rejects the source or destination.
    #[inline]
    pub fn assign<Rhs>(&mut self, rhs: &Rhs)
    where
        T: Copy,
        Rhs: AssignSource<T, N>,
    {
        self.try_assign(rhs)
            .expect("invariant: assigned views have valid compatible layouts");
    }
}

/// Checked per-element assignment for views whose physical window is shared
/// with sibling lane/axis views: every write goes through `get_mut`, never
/// through a whole-window slice that would alias sibling elements.
fn assign_into_elements<T, Rhs, const N: usize>(
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

fn assign_into<T, Rhs, const N: usize>(
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

fn transpose_tile<T>() -> usize {
    let element_bytes = core::mem::size_of::<T>().max(1);
    let tile_elements = (TRANSPOSE_PAYLOAD_BUDGET_BYTES / (2 * element_bytes)).max(1);
    let side = tile_elements.isqrt().max(1);
    1usize << side.ilog2()
}

/// Copies a row-major matrix into its row-major transpose.
///
/// `source` has shape `[rows, columns]` and `destination` has shape
/// `[columns, rows]`. Both slices must contain exactly `rows * columns`
/// elements. Each destination at `column * rows + row` receives a clone of
/// `source[row * columns + column]`. Offsets are supplied by borrowing the
/// desired subslices; elements outside those slices remain untouched.
///
/// The cache-blocked traversal creates no intermediate storage. It clones
/// each source element once, including zero-sized types; a user-defined
/// [`Clone`] implementation may itself allocate. Copy scalar payloads undergo
/// no arithmetic, preserving signed zeros and NaN representations.
///
/// # Errors
///
/// Returns [`LetoError::Overflow`] if the element count overflows `usize` or
/// a nonempty matrix exceeds the signed extent supported by dense layouts.
/// Returns [`LetoError::StorageError`] for an inexact slice length. Validation
/// proceeds in this order: product, source length, destination length, then
/// signed extent. A zero product with empty slices succeeds even when the
/// other dimension exceeds `isize::MAX`. Errors leave both slices unchanged.
///
/// # Panics
///
/// Element cloning or destruction may panic after earlier destination values
/// have been replaced. Validation errors do not execute either operation.
///
/// # Examples
///
/// ```
/// use leto::transpose_copy;
/// let source = [1, 2, 3, 4, 5, 6];
/// let mut destination = [0; 6];
/// transpose_copy(&source, &mut destination, 2, 3)?;
/// assert_eq!(destination, [1, 4, 2, 5, 3, 6]);
/// # Ok::<(), leto::LetoError>(())
/// ```
pub fn transpose_copy<T: Clone>(
    source: &[T],
    destination: &mut [T],
    rows: usize,
    columns: usize,
) -> Result<()> {
    if transpose_extent(source.len(), destination.len(), rows, columns)? == 0 {
        return Ok(());
    }
    // The traversal addresses destination rows and source columns. Nonzero
    // dimensions and their product fit isize, bounding every slice product
    // and tile endpoint; all exact chunks below have nonzero widths.
    let (height, width) = (columns, rows);
    let tile = transpose_tile::<T>();
    if width >= height {
        for row_start in (0..height).step_by(tile) {
            let row_end = (row_start + tile).min(height);
            for column_start in (0..width).step_by(tile) {
                let column_end = (column_start + tile).min(width);
                let source_columns = &source[column_start * height..column_end * height];
                let destination_rows = &mut destination[row_start * width..row_end * width];
                for (row_offset, destination_row) in
                    destination_rows.chunks_exact_mut(width).enumerate()
                {
                    let row = row_start + row_offset;
                    for (target, source_column) in destination_row[column_start..column_end]
                        .iter_mut()
                        .zip(source_columns.chunks_exact(height))
                    {
                        *target = source_column[row].clone();
                    }
                }
            }
        }
        return Ok(());
    }

    for column_start in (0..width).step_by(tile) {
        let column_end = (column_start + tile).min(width);
        for row_start in (0..height).step_by(tile) {
            let row_end = (row_start + tile).min(height);
            let destination_rows = &mut destination[row_start * width..row_end * width];
            for column in column_start..column_end {
                let source_column = &source[column * height + row_start..column * height + row_end];
                for (destination_row, value) in
                    destination_rows.chunks_exact_mut(width).zip(source_column)
                {
                    destination_row[column] = value.clone();
                }
            }
        }
    }
    Ok(())
}

fn transpose_extent(
    source_len: usize,
    destination_len: usize,
    rows: usize,
    columns: usize,
) -> Result<usize> {
    #[expect(
        clippy::unnecessary_lazy_evaluations,
        reason = "Avoid eager LetoError drop on successful arithmetic; ADR 0027"
    )]
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| LetoError::Overflow {
            reason: "dense transpose element count",
        })?;
    for (role, actual) in [("source", source_len), ("destination", destination_len)] {
        if actual != elements {
            return Err(LetoError::StorageError {
                reason: format!(
                    "dense transpose {role} length {actual} does not match expected {elements}"
                ),
            });
        }
    }
    isize::try_from(elements).map_err(|_| LetoError::Overflow {
        reason: "dense transpose signed layout extent",
    })?;
    Ok(elements)
}
