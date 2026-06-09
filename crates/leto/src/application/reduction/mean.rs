//! Mean reductions over N-dimensional strided arrays.

use num_traits::Float;

use crate::application::array::Array;
use crate::application::index::index_from_flat;
use crate::application::reduction::iter_elements;
use crate::application::reduction::sum::sum_axis;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::domain::remove_axis::{RankMarker, RemoveAxis};
use crate::infrastructure::storage::{Storage, VecStorage};

/// Mean of all elements of `arr`.
///
/// # Errors
/// Returns `Err` if `arr` is empty.
pub fn mean_all<T, S, const N: usize>(arr: &Array<T, S, N>) -> Result<T>
where
    T: Float + for<'a> std::ops::AddAssign<&'a T>,
    S: Storage<T>,
{
    if arr.size() == 0 {
        return Err(LetoError::StorageError {
            reason: "reduction over empty array is undefined".to_string(),
        });
    }
    let view = arr.view();
    let mut acc = T::zero();
    for elem in iter_elements(&view) {
        acc += elem;
    }
    let count = T::from(arr.size()).ok_or(LetoError::StorageError {
        reason: "element count exceeds float precision range".to_string(),
    })?;
    Ok(acc / count)
}

/// Mean along `axis`, reducing rank by one.
///
/// # Errors
/// Returns `Err` if `axis >= N` or the axis has length 0.
pub fn mean_axis<T, S, const N: usize, const M: usize>(
    arr: &Array<T, S, N>,
    axis: usize,
) -> Result<Array<T, VecStorage<T>, M>>
where
    T: Float + std::ops::Add<Output = T> + Copy,
    S: Storage<T>,
    RankMarker<N>: RemoveAxis<N, SmallerShape = [usize; M], SmallerStrides = [isize; M]>,
{
    if axis >= N {
        return Err(LetoError::StorageError {
            reason: format!("axis {axis} out of bounds for rank {N}"),
        });
    }
    let axis_len = arr.shape()[axis];
    if axis_len == 0 {
        return Err(LetoError::StorageError {
            reason: format!("axis {axis} has length 0; mean is undefined"),
        });
    }
    let count = T::from(axis_len).ok_or(LetoError::StorageError {
        reason: "axis length exceeds float precision range".to_string(),
    })?;

    let sum: Array<T, VecStorage<T>, M> = sum_axis::<T, S, N, M>(arr, axis)?;

    let sum_view = sum.view();
    let sum_size = sum_view.size();
    let sum_shape = sum_view.shape();
    let sum_layout = sum_view.layout();
    let sum_data = sum_view.data();

    let mut buf: Vec<T> = Vec::with_capacity(sum_size);
    for flat in 0..sum_size {
        let idx = index_from_flat(flat, &sum_shape);
        let off = sum_layout.offset_of(idx)?;
        buf.push(sum_data[off] / count);
    }

    let out_layout = Layout::c_contiguous(sum_shape)?;
    let storage = VecStorage::new(buf);
    Array::new(out_layout, storage)
}
