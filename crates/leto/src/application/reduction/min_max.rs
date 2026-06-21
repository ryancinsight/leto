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

/// Multi-index of the minimum element of the whole array (ndarray-stats
/// `argmin` parity).
///
/// Returns the N-dimensional index `[usize; N]` of the smallest element in
/// logical row-major order. On ties the **first** (lowest row-major index)
/// occurrence wins, matching ndarray-stats. The companion [`min_all`] returns
/// the value at this index.
///
/// # Errors
/// [`LetoError`] if `arr` is empty.
pub fn argmin_all<T, S, const N: usize>(arr: &Array<T, S, N>) -> Result<[usize; N]>
where
    T: PartialOrd + Copy,
    S: Storage<T>,
{
    arg_reduce_all(arr, |best, candidate| candidate < best)
}

/// Multi-index of the maximum element of the whole array (ndarray-stats
/// `argmax` parity).
///
/// Returns the N-dimensional index `[usize; N]` of the largest element in
/// logical row-major order. On ties the **first** occurrence wins. The
/// companion [`max_all`] returns the value at this index.
///
/// # Errors
/// [`LetoError`] if `arr` is empty.
pub fn argmax_all<T, S, const N: usize>(arr: &Array<T, S, N>) -> Result<[usize; N]>
where
    T: PartialOrd + Copy,
    S: Storage<T>,
{
    arg_reduce_all(arr, |best, candidate| candidate > best)
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
    axis_reduce_lanes::<T, S, N, M, _>(arr, axis, |acc, elem| if elem < acc { elem } else { acc })
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
    axis_reduce_lanes::<T, S, N, M, _>(arr, axis, |acc, elem| if elem > acc { elem } else { acc })
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

/// Whole-array argreduce: scan every logical element in row-major order,
/// tracking the running best value and its flat index, then convert the winning
/// flat index to a multi-index.
///
/// `is_better(current_best, candidate) -> bool` is strict, so the first
/// occurrence wins on ties (logical row-major order).
fn arg_reduce_all<T, S, const N: usize>(
    arr: &Array<T, S, N>,
    is_better: impl Fn(T, T) -> bool,
) -> Result<[usize; N]>
where
    T: Copy,
    S: Storage<T>,
{
    if arr.size() == 0 {
        return Err(LetoError::StorageError {
            reason: "argreduce over empty array is undefined".to_string(),
        });
    }
    let view = arr.view();
    let shape = view.layout().shape;
    let mut it = iter_elements(&view).enumerate();
    let (_, first) = it.next().expect("non-empty array has at least one element");
    let mut best_val = *first;
    let mut best_flat = 0usize;
    for (flat, elem) in it {
        if is_better(best_val, *elem) {
            best_val = *elem;
            best_flat = flat;
        }
    }
    Ok(index_from_flat(best_flat, &shape))
}

/// Generic axis-reduce kernel applying a binary `fold` elementwise across lanes.
///
/// The first lane is used as the initial accumulator — no identity value needed.
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
    let mut buf: Vec<T> = Vec::with_capacity(out_size);
    if let Some(slice) = first.as_slice() {
        buf.extend_from_slice(slice);
    } else {
        for val in first.iter() {
            buf.push(*val);
        }
    }

    for lane in iter {
        if let Some(slice) = lane.as_slice() {
            for (buf_val, &lane_val) in buf.iter_mut().zip(slice) {
                *buf_val = fold(*buf_val, lane_val);
            }
        } else {
            for (flat, val) in lane.iter().enumerate() {
                buf[flat] = fold(buf[flat], *val);
            }
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
    let mut best_val: Vec<T> = Vec::with_capacity(out_size);
    if let Some(slice) = first.as_slice() {
        best_val.extend_from_slice(slice);
    } else {
        for val in first.iter() {
            best_val.push(*val);
        }
    }
    let mut best_idx: Vec<usize> = vec![0usize; out_size];

    for (lane_pos, lane) in iter.enumerate() {
        let lane_axis_pos = lane_pos + 1;
        if let Some(slice) = lane.as_slice() {
            for ((val_ref, idx_ref), &candidate) in best_val.iter_mut().zip(&mut best_idx).zip(slice) {
                if is_better(*val_ref, candidate) {
                    *val_ref = candidate;
                    *idx_ref = lane_axis_pos;
                }
            }
        } else {
            for ((val_ref, idx_ref), val) in best_val.iter_mut().zip(&mut best_idx).zip(lane.iter()) {
                let candidate = *val;
                if is_better(*val_ref, candidate) {
                    *val_ref = candidate;
                    *idx_ref = lane_axis_pos;
                }
            }
        }
    }

    let out_layout = Layout::c_contiguous(out_shape)?;
    let storage = VecStorage::new(best_idx);
    Array::new(out_layout, storage)
}
