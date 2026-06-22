//! Sum reductions over N-dimensional strided arrays.

use num_traits::Zero;

use crate::application::array::Array;
use crate::application::iter::AxisIter;
use crate::application::reduction::iter_elements;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::domain::remove_axis::{RankMarker, RemoveAxis};
use crate::infrastructure::storage::{Storage, VecStorage};

/// Sum all elements of `arr`.
///
/// Traverses the full logical index space; correct for any strided layout.
///
/// # Errors
/// Returns `Err` if `arr` is empty.
pub fn sum_all<T, S, const N: usize>(arr: &Array<T, S, N>) -> Result<T>
where
    T: Zero + for<'a> std::ops::AddAssign<&'a T> + Copy,
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
    Ok(acc)
}

/// Sum elements along `axis`, reducing rank by one.
///
/// Returns `Array<T, VecStorage<T>, M>` where `M = N - 1`.
///
/// # Errors
/// Returns `Err` if `axis >= N` or if the layout is invalid.
#[allow(clippy::needless_range_loop)]
pub fn sum_axis<T, S, const N: usize, const M: usize>(
    arr: &Array<T, S, N>,
    axis: usize,
) -> Result<Array<T, VecStorage<T>, M>>
where
    T: Zero + std::ops::Add<Output = T> + Copy,
    S: Storage<T>,
    RankMarker<N>: RemoveAxis<N, SmallerShape = [usize; M], SmallerStrides = [isize; M]>,
{
    let view = arr.view();
    let iter: AxisIter<'_, T, N, M> = AxisIter::new(&view, axis, RankMarker::<N>)?;
    let out_shape = RankMarker::<N>.remove_shape(arr.shape(), axis)?;
    let out_size: usize = out_shape.iter().product();

    let mut buf: Vec<T> = vec![T::zero(); out_size];
    for lane in iter {
        if let Some(slice) = lane.as_slice() {
            for (buf_val, &lane_val) in buf.iter_mut().zip(slice) {
                *buf_val = *buf_val + lane_val;
            }
        } else {
            for (buf_val, &lane_val) in buf.iter_mut().zip(lane.iter()) {
                *buf_val = *buf_val + lane_val;
            }
        }
    }

    let out_layout = Layout::c_contiguous(out_shape)?;
    let storage = VecStorage::new(buf);
    Array::new(out_layout, storage)
}
