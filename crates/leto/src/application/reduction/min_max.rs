//! Min, max, argmin, and argmax reductions over N-dimensional strided arrays.

use crate::application::array::Array;
use crate::application::index::index_from_flat;
use crate::application::iter::AxisIter;
use crate::application::reduction::iter_elements;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::domain::remove_axis::{RankMarker, RemoveAxis};
use crate::infrastructure::storage::{Storage, VecStorage};

// ── All-reduce ────────────────────────────────────────────────────────────────

/// Minimum element of `arr`.
///
/// Uses `PartialOrd`; the first element that compares less replaces the
/// running minimum. NaN behaviour depends on the concrete type's `PartialOrd`.
///
/// # Errors
/// Returns `Err` if `arr` is empty.
pub fn min_all<T, S, const N: usize>(arr: &Array<T, S, N>) -> Result<T>
where
    T: PartialOrd + Copy,
    S: Storage<T>,
{
    if arr.size() == 0 {
        return Err(LetoError::StorageError {
            reason: "reduction over empty array is undefined".to_string(),
        });
    }
    let view = arr.view();
    let mut it = iter_elements(&view);
    let mut acc = *it.next().expect("non-empty array has at least one element");
    for elem in it {
        if *elem < acc {
            acc = *elem;
        }
    }
    Ok(acc)
}

/// Maximum element of `arr`.
///
/// # Errors
/// Returns `Err` if `arr` is empty.
pub fn max_all<T, S, const N: usize>(arr: &Array<T, S, N>) -> Result<T>
where
    T: PartialOrd + Copy,
    S: Storage<T>,
{
    if arr.size() == 0 {
        return Err(LetoError::StorageError {
            reason: "reduction over empty array is undefined".to_string(),
        });
    }
    let view = arr.view();
    let mut it = iter_elements(&view);
    let mut acc = *it.next().expect("non-empty array has at least one element");
    for elem in it {
        if *elem > acc {
            acc = *elem;
        }
    }
    Ok(acc)
}

// ── Axis-reduce ───────────────────────────────────────────────────────────────

/// Minimum along `axis`, reducing rank by one.
///
/// # Errors
/// Returns `Err` if `axis >= N` or the axis has length 0.
pub fn min_axis<T, S, const N: usize, const M: usize>(
    arr: &Array<T, S, N>,
    axis: usize,
) -> Result<Array<T, VecStorage<T>, M>>
where
    T: PartialOrd + Copy,
    S: Storage<T>,
    RankMarker<N>: RemoveAxis<N, SmallerShape = [usize; M], SmallerStrides = [isize; M]>,
{
    axis_reduce_lanes::<T, S, N, M, _>(arr, axis, |acc, elem| {
        if elem < acc { elem } else { acc }
    })
}

/// Maximum along `axis`, reducing rank by one.
///
/// # Errors
/// Returns `Err` if `axis >= N` or the axis has length 0.
pub fn max_axis<T, S, const N: usize, const M: usize>(
    arr: &Array<T, S, N>,
    axis: usize,
) -> Result<Array<T, VecStorage<T>, M>>
where
    T: PartialOrd + Copy,
    S: Storage<T>,
    RankMarker<N>: RemoveAxis<N, SmallerShape = [usize; M], SmallerStrides = [isize; M]>,
{
    axis_reduce_lanes::<T, S, N, M, _>(arr, axis, |acc, elem| {
        if elem > acc { elem } else { acc }
    })
}

// ── argmin / argmax ───────────────────────────────────────────────────────────

/// Index of the minimum element along `axis`.
///
/// Returns `Array<usize, VecStorage<usize>, M>` where each element holds the
/// position along `axis` of the minimum value in that lane.
///
/// # Errors
/// Returns `Err` if `axis >= N` or the axis has length 0.
pub fn argmin<T, S, const N: usize, const M: usize>(
    arr: &Array<T, S, N>,
    axis: usize,
) -> Result<Array<usize, VecStorage<usize>, M>>
where
    T: PartialOrd + Copy,
    S: Storage<T>,
    RankMarker<N>: RemoveAxis<N, SmallerShape = [usize; M], SmallerStrides = [isize; M]>,
{
    axis_arg_reduce::<T, S, N, M>(arr, axis, |a, b| b < a)
}

/// Index of the maximum element along `axis`.
///
/// Returns `Array<usize, VecStorage<usize>, M>` where each element holds the
/// position along `axis` of the maximum value in that lane.
///
/// # Errors
/// Returns `Err` if `axis >= N` or the axis has length 0.
pub fn argmax<T, S, const N: usize, const M: usize>(
    arr: &Array<T, S, N>,
    axis: usize,
) -> Result<Array<usize, VecStorage<usize>, M>>
where
    T: PartialOrd + Copy,
    S: Storage<T>,
    RankMarker<N>: RemoveAxis<N, SmallerShape = [usize; M], SmallerStrides = [isize; M]>,
{
    axis_arg_reduce::<T, S, N, M>(arr, axis, |a, b| b > a)
}

// ── Internal kernels ──────────────────────────────────────────────────────────

/// Generic axis-reduce kernel applying a binary `fold` elementwise across lanes.
///
/// The first lane is used as the initial accumulator — no identity value needed.
#[allow(clippy::needless_range_loop)]
fn axis_reduce_lanes<T, S, const N: usize, const M: usize, F>(
    arr: &Array<T, S, N>,
    axis: usize,
    fold: F,
) -> Result<Array<T, VecStorage<T>, M>>
where
    T: Copy,
    S: Storage<T>,
    F: Fn(T, T) -> T,
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
            reason: format!("axis {axis} has length 0"),
        });
    }

    let out_shape = RankMarker::<N>.remove_shape(arr.shape(), axis)?;
    let out_size: usize = out_shape.iter().product();

    let view = arr.view();
    let mut iter: AxisIter<'_, T, N, M> = AxisIter::new(&view, axis, RankMarker::<N>)?;

    let first = iter.next().expect("axis_len > 0 implies at least one lane");
    let first_layout = first.layout();
    let first_data = first.data();
    let first_shape = first_layout.shape;
    let mut buf: Vec<T> = Vec::with_capacity(out_size);
    for flat in 0..out_size {
        let idx = index_from_flat(flat, &first_shape);
        let off = first_layout.offset_of(idx)?;
        buf.push(first_data[off]);
    }

    for lane in iter {
        let lane_layout = lane.layout();
        let lane_data = lane.data();
        let lane_shape = lane_layout.shape;
        for flat in 0..out_size {
            let idx = index_from_flat(flat, &lane_shape);
            let off = lane_layout.offset_of(idx)?;
            buf[flat] = fold(buf[flat], lane_data[off]);
        }
    }

    let out_layout = Layout::c_contiguous(out_shape)?;
    let storage = VecStorage::new(buf);
    Array::new(out_layout, storage)
}

/// Generic argreduce kernel.
///
/// `is_better(current_best, candidate) -> bool` returns `true` when `candidate`
/// should replace `current_best`.
#[allow(clippy::needless_range_loop)]
fn axis_arg_reduce<T, S, const N: usize, const M: usize>(
    arr: &Array<T, S, N>,
    axis: usize,
    is_better: impl Fn(T, T) -> bool,
) -> Result<Array<usize, VecStorage<usize>, M>>
where
    T: Copy,
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
            reason: format!("axis {axis} has length 0"),
        });
    }

    let out_shape = RankMarker::<N>.remove_shape(arr.shape(), axis)?;
    let out_size: usize = out_shape.iter().product();

    let view = arr.view();
    let mut iter: AxisIter<'_, T, N, M> = AxisIter::new(&view, axis, RankMarker::<N>)?;

    let first = iter.next().expect("axis_len > 0 implies at least one lane");
    let first_layout = first.layout();
    let first_data = first.data();
    let first_shape = first_layout.shape;

    let mut best_val: Vec<T> = Vec::with_capacity(out_size);
    let mut best_idx: Vec<usize> = vec![0usize; out_size];
    for flat in 0..out_size {
        let idx = index_from_flat(flat, &first_shape);
        let off = first_layout.offset_of(idx)?;
        best_val.push(first_data[off]);
    }

    for (lane_pos, lane) in iter.enumerate() {
        let lane_axis_pos = lane_pos + 1;
        let lane_layout = lane.layout();
        let lane_data = lane.data();
        let lane_shape = lane_layout.shape;
        for flat in 0..out_size {
            let idx = index_from_flat(flat, &lane_shape);
            let off = lane_layout.offset_of(idx)?;
            let candidate = lane_data[off];
            if is_better(best_val[flat], candidate) {
                best_val[flat] = candidate;
                best_idx[flat] = lane_axis_pos;
            }
        }
    }

    let out_layout = Layout::c_contiguous(out_shape)?;
    let storage = VecStorage::new(best_idx);
    Array::new(out_layout, storage)
}
