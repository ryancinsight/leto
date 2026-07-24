//! Mean reductions over N-dimensional strided arrays.

use eunomia::FloatElement;

use crate::application::array::Array;
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
    T: FloatElement + for<'a> std::ops::AddAssign<&'a T>,
    S: Storage<T>,
{
    if arr.size() == 0 {
        return Err(LetoError::StorageError {
            reason: "reduction over empty array is undefined".to_string(),
        });
    }
    let view = arr.view();
    let mut acc = T::ZERO;
    for elem in iter_elements(&view) {
        acc += elem;
    }
    let count = T::from_f64((arr.size()) as f64);
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
    T: FloatElement + std::ops::Add<Output = T> + Copy,
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
    let count = T::from_f64((axis_len) as f64);

    let sum: Array<T, VecStorage<T>, M> = sum_axis::<T, S, N, M>(arr, axis)?;

    let sum_view = sum.view();
    let sum_size = sum_view.size();
    let sum_shape = sum_view.shape();

    let mut buf: Vec<T> = Vec::with_capacity(sum_size);
    if let Some(slice) = sum_view.as_slice() {
        for &val in slice {
            buf.push(val / count);
        }
    } else {
        for val in sum_view.iter() {
            buf.push(*val / count);
        }
    }

    let out_layout = Layout::c_contiguous(sum_shape)?;
    let storage = VecStorage::new(buf);
    Array::new(out_layout, storage)
}
