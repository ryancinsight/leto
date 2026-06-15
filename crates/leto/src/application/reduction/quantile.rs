//! Quantile and median reductions (ndarray-stats / numpy parity).
//!
//! A quantile generalizes the median: the `q`-quantile of a sample is the value
//! below which a fraction `q ∈ [0, 1]` of the (sorted) data lies. Because a
//! finite sample rarely has an element at the exact fractional rank `q·(n−1)`,
//! an *interpolation method* selects a value between the two bracketing order
//! statistics; [`Interpolation`] enumerates the five standard choices, matching
//! numpy's `numpy.quantile` and `ndarray-stats`' `Quantile1dExt`.

use num_traits::Float;

use crate::application::array::Array;
use crate::application::index::index_from_flat;
use crate::application::iter::AxisIter;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::domain::remove_axis::{RankMarker, RemoveAxis};
use crate::infrastructure::storage::{Storage, VecStorage};

/// Strategy for interpolating between the two order statistics that bracket a
/// fractional rank `h = q·(n−1)`.
///
/// With sorted data `v₀ ≤ … ≤ vₙ₋₁`, `lo = ⌊h⌋`, and `g = h − lo`:
/// - [`Linear`](Interpolation::Linear): `v[lo] + g·(v[lo+1] − v[lo])` — numpy's
///   default; the unique method that is continuous in `q`.
/// - [`Lower`](Interpolation::Lower): `v[lo]`.
/// - [`Higher`](Interpolation::Higher): `v[⌈h⌉]`.
/// - [`Nearest`](Interpolation::Nearest): `v[round(h)]`.
/// - [`Midpoint`](Interpolation::Midpoint): `(v[lo] + v[⌈h⌉]) / 2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Continuous linear interpolation (numpy default).
    #[default]
    Linear,
    /// Lower of the two bracketing order statistics.
    Lower,
    /// Higher of the two bracketing order statistics.
    Higher,
    /// Whichever bracketing order statistic is nearer the fractional rank.
    Nearest,
    /// Arithmetic mean of the two bracketing order statistics.
    Midpoint,
}

/// `q`-quantile of all elements under the chosen [`Interpolation`].
///
/// # Theorem (fractional-rank quantile)
/// For a sample of size `n`, the `q`-quantile sits at fractional rank
/// `h = q·(n−1)` within the sorted order statistics `v₀ ≤ … ≤ vₙ₋₁`
/// (`0`-based, so `q = 0 → v₀`, `q = 1 → vₙ₋₁`). [`Interpolation::Linear`]
/// returns the unique value making the empirical quantile function continuous
/// and piecewise-linear in `q`; the other methods select a representative order
/// statistic without interpolation. ∎
///
/// # Errors
/// [`LetoError`] if `arr` is empty, `q ∉ [0, 1]` or non-finite, or any element
/// is NaN (which would make the sort order ill-defined).
pub fn quantile_all<T, S, const N: usize>(
    arr: &Array<T, S, N>,
    q: T,
    method: Interpolation,
) -> Result<T>
where
    T: Float,
    S: Storage<T>,
{
    let n = arr.size();
    if n == 0 {
        return Err(LetoError::StorageError {
            reason: "quantile over empty array is undefined".to_string(),
        });
    }
    let view = arr.view();
    let layout = view.layout();
    let data = view.data();
    let shape = layout.shape;
    let mut values: Vec<T> = Vec::with_capacity(n);
    for flat in 0..n {
        let off = layout.offset_of(index_from_flat(flat, &shape))?;
        values.push(data[off]);
    }
    quantile_of_slice(&mut values, q, method)
}

/// Median of all elements: `quantile_all(arr, 0.5, Linear)`.
///
/// # Errors
/// As [`quantile_all`].
pub fn median_all<T, S, const N: usize>(arr: &Array<T, S, N>) -> Result<T>
where
    T: Float,
    S: Storage<T>,
{
    let half = T::from(0.5).ok_or(LetoError::StorageError {
        reason: "0.5 not representable in target float type".to_string(),
    })?;
    quantile_all(arr, half, Interpolation::Linear)
}

/// `q`-quantile along `axis` (reducing rank by one) under `method`.
///
/// Gathers each output position's axis lane, sorts it, and interpolates per
/// [`quantile_all`]. A single `out_size × axis_len` scratch buffer is reused
/// across all lanes — no per-lane allocation.
///
/// # Errors
/// [`LetoError`] if `axis ≥ N`, the axis is empty, `q ∉ [0, 1]` or non-finite,
/// or any element is NaN.
pub fn quantile_axis<T, S, const N: usize, const M: usize>(
    arr: &Array<T, S, N>,
    axis: usize,
    q: T,
    method: Interpolation,
) -> Result<Array<T, VecStorage<T>, M>>
where
    T: Float,
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
            reason: format!("axis {axis} has length 0; quantile is undefined"),
        });
    }
    let out_shape = RankMarker::<N>.remove_shape(arr.shape(), axis)?;
    let out_size: usize = out_shape.iter().product();

    // Scatter each axis lane (index k) into column k of a row-major
    // [out_size × axis_len] scratch buffer, then quantile each row in place.
    let view = arr.view();
    let iter: AxisIter<'_, T, N, M> = AxisIter::new(&view, axis, RankMarker::<N>)?;
    let mut scratch = vec![T::zero(); out_size * axis_len];
    for (k, lane) in iter.enumerate() {
        let lane_layout = lane.layout();
        let lane_data = lane.data();
        let lane_shape = lane_layout.shape;
        for flat in 0..out_size {
            let off = lane_layout.offset_of(index_from_flat(flat, &lane_shape))?;
            scratch[flat * axis_len + k] = lane_data[off];
        }
    }

    let mut buf = vec![T::zero(); out_size];
    for (flat, slot) in buf.iter_mut().enumerate() {
        let row = &mut scratch[flat * axis_len..(flat + 1) * axis_len];
        *slot = quantile_of_slice(row, q, method)?;
    }

    let out_layout = Layout::c_contiguous(out_shape)?;
    Array::new(out_layout, VecStorage::new(buf))
}

/// Median along `axis`: `quantile_axis(arr, axis, 0.5, Linear)`.
///
/// # Errors
/// As [`quantile_axis`].
pub fn median_axis<T, S, const N: usize, const M: usize>(
    arr: &Array<T, S, N>,
    axis: usize,
) -> Result<Array<T, VecStorage<T>, M>>
where
    T: Float,
    S: Storage<T>,
    RankMarker<N>: RemoveAxis<N, SmallerShape = [usize; M], SmallerStrides = [isize; M]>,
{
    let half = T::from(0.5).ok_or(LetoError::StorageError {
        reason: "0.5 not representable in target float type".to_string(),
    })?;
    quantile_axis(arr, axis, half, Interpolation::Linear)
}

/// Sort `values` in place and return its `q`-quantile under `method`.
///
/// Shared SSOT kernel for [`quantile_all`] and [`quantile_axis`]. `values` is
/// non-empty by construction at both call sites.
fn quantile_of_slice<T: Float>(values: &mut [T], q: T, method: Interpolation) -> Result<T> {
    if !q.is_finite() || q < T::zero() || q > T::one() {
        return Err(LetoError::StorageError {
            reason: "quantile q must be finite and within [0, 1]".to_string(),
        });
    }
    if values.iter().any(|v| v.is_nan()) {
        return Err(LetoError::StorageError {
            reason: "quantile over data containing NaN is undefined".to_string(),
        });
    }
    // NaN-free, so partial_cmp is a total order here.
    values.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("invariant: NaN rejected above, ordering is total")
    });

    let n = values.len();
    if n == 1 {
        return Ok(values[0]);
    }
    let len_minus_one = T::from(n - 1).ok_or(LetoError::StorageError {
        reason: "sample size exceeds float precision range".to_string(),
    })?;
    let h = q * len_minus_one;
    let lo = h.floor();
    let lo_idx = lo.to_usize().ok_or(LetoError::StorageError {
        reason: "fractional rank not representable as index".to_string(),
    })?;
    let g = h - lo;

    let value = match method {
        Interpolation::Lower => values[lo_idx],
        Interpolation::Higher => values[ceil_idx(lo_idx, g, n)],
        Interpolation::Nearest => {
            let half = T::from(0.5).expect("invariant: 0.5 representable in float type");
            if g > half || (g == half && lo_idx % 2 == 1) {
                values[ceil_idx(lo_idx, g, n)]
            } else {
                values[lo_idx]
            }
        }
        Interpolation::Midpoint => {
            let hi = values[ceil_idx(lo_idx, g, n)];
            let two = T::from(2).expect("invariant: 2 representable in float type");
            (values[lo_idx] + hi) / two
        }
        Interpolation::Linear => {
            if g.is_zero() {
                values[lo_idx]
            } else {
                let hi = values[lo_idx + 1];
                values[lo_idx] + g * (hi - values[lo_idx])
            }
        }
    };
    Ok(value)
}

/// Index of the upper bracketing order statistic: `lo` when the rank is exact
/// (`g = 0`), else `lo + 1`, clamped to the last valid index.
#[inline]
fn ceil_idx<T: Float>(lo_idx: usize, g: T, n: usize) -> usize {
    if g.is_zero() {
        lo_idx
    } else {
        (lo_idx + 1).min(n - 1)
    }
}
