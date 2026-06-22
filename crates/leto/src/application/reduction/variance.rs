//! Variance and standard-deviation reductions (ndarray-stats parity).

use num_traits::Float;

use crate::application::array::Array;
use crate::application::iter::AxisIter;
use crate::application::reduction::iter_elements;
use crate::application::reduction::mean::{mean_all, mean_axis};
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::domain::remove_axis::{RankMarker, RemoveAxis};
use crate::infrastructure::storage::{Storage, StorageMut, VecStorage};

/// Variance of all elements with `ddof` delta degrees of freedom.
///
/// # Theorem (two-pass variance)
/// For samples `x₁…xₙ` with mean `x̄`, the (Bessel-corrected) variance is
/// `s² = (1/(n−ddof)) Σᵢ (xᵢ − x̄)²` — `ddof = 0` is the population variance,
/// `ddof = 1` the unbiased sample variance.
/// *Why two passes:* the algebraically equal one-pass form
/// `(Σxᵢ² − (Σxᵢ)²/n)/(n−ddof)` subtracts two large nearly-equal quantities and
/// loses catastrophic precision for data with large mean; computing `x̄` first
/// and accumulating `Σ(xᵢ − x̄)²` keeps each summand `O(variance)` and is
/// numerically stable. This implementation uses the two-pass form. ∎
///
/// # Errors
/// [`LetoError`] if `arr` is empty or `n − ddof ≤ 0`.
pub fn var_all<T, S, const N: usize>(arr: &Array<T, S, N>, ddof: T) -> Result<T>
where
    T: Float + for<'a> std::ops::AddAssign<&'a T>,
    S: Storage<T>,
{
    let n = arr.size();
    if n == 0 {
        return Err(LetoError::StorageError {
            reason: "variance over empty array is undefined".to_string(),
        });
    }
    let mean = mean_all(arr)?;
    let view = arr.view();
    let mut acc = T::zero();
    for elem in iter_elements(&view) {
        let deviation = *elem - mean;
        acc = acc + deviation * deviation;
    }
    let denom = degrees_of_freedom(n, ddof)?;
    Ok(acc / denom)
}

/// Standard deviation of all elements: `√(var_all)`.
///
/// # Errors
/// [`LetoError`] if `arr` is empty or `n − ddof ≤ 0`.
pub fn std_all<T, S, const N: usize>(arr: &Array<T, S, N>, ddof: T) -> Result<T>
where
    T: Float + for<'a> std::ops::AddAssign<&'a T>,
    S: Storage<T>,
{
    Ok(var_all(arr, ddof)?.sqrt())
}

/// Variance along `axis` (reducing rank by one) with `ddof` degrees of freedom.
///
/// Two-pass per output position: [`mean_axis`] then `Σ(x − mean)²` over the
/// axis, divided by `(axis_len − ddof)`. See [`var_all`] for the theorem.
///
/// # Errors
/// [`LetoError`] if `axis ≥ N`, the axis is empty, or `axis_len − ddof ≤ 0`.
pub fn var_axis<T, S, const N: usize, const M: usize>(
    arr: &Array<T, S, N>,
    axis: usize,
    ddof: T,
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
            reason: format!("axis {axis} has length 0; variance is undefined"),
        });
    }
    let denom = degrees_of_freedom(axis_len, ddof)?;

    // Per-output mean: `mean_axis` returns a C-contiguous array (offset 0) whose
    // `data()[flat]` is the mean at output position `flat` in C-order — index it
    // directly, no second gather buffer.
    let means = mean_axis::<T, S, N, M>(arr, axis)?;
    let means_view = means.view();
    let mean_data = means_view.data();
    let out_shape = RankMarker::<N>.remove_shape(arr.shape(), axis)?;
    let out_size: usize = out_shape.iter().product();

    // Accumulate squared deviations over the axis lanes.
    let view = arr.view();
    let iter: AxisIter<'_, T, N, M> = AxisIter::new(&view, axis, RankMarker::<N>)?;
    let mut buf = vec![T::zero(); out_size];
    let mean_slice = &mean_data[..out_size];
    for lane in iter {
        if let Some(slice) = lane.as_slice() {
            for ((slot, &lane_val), &mean_val) in buf.iter_mut().zip(slice).zip(mean_slice) {
                let deviation = lane_val - mean_val;
                *slot = *slot + deviation * deviation;
            }
        } else {
            for ((slot, &lane_val), &mean_val) in buf.iter_mut().zip(lane.iter()).zip(mean_slice) {
                let deviation = lane_val - mean_val;
                *slot = *slot + deviation * deviation;
            }
        }
    }
    for value in buf.iter_mut() {
        *value = *value / denom;
    }

    let out_layout = Layout::c_contiguous(out_shape)?;
    Array::new(out_layout, VecStorage::new(buf))
}

/// Standard deviation along `axis`: elementwise `√(var_axis)`.
///
/// # Errors
/// [`LetoError`] if `axis ≥ N`, the axis is empty, or `axis_len − ddof ≤ 0`.
pub fn std_axis<T, S, const N: usize, const M: usize>(
    arr: &Array<T, S, N>,
    axis: usize,
    ddof: T,
) -> Result<Array<T, VecStorage<T>, M>>
where
    T: Float + std::ops::Add<Output = T> + Copy,
    S: Storage<T>,
    RankMarker<N>: RemoveAxis<N, SmallerShape = [usize; M], SmallerStrides = [isize; M]>,
{
    let mut variance = var_axis::<T, S, N, M>(arr, axis, ddof)?;
    for value in variance.storage_mut().as_mut_slice().iter_mut() {
        *value = value.sqrt();
    }
    Ok(variance)
}

/// `count − ddof`, rejecting a non-positive (or non-representable) result.
///
/// Shared by the variance reductions and the [`statistics`](crate::application::statistics)
/// covariance kernel (SSOT for the degrees-of-freedom contract).
pub(crate) fn degrees_of_freedom<T: Float>(count: usize, ddof: T) -> Result<T> {
    if !ddof.is_finite() {
        return Err(LetoError::StorageError {
            reason: "variance degrees of freedom must be finite".to_string(),
        });
    }
    let count = T::from(count).ok_or(LetoError::StorageError {
        reason: "element count exceeds float precision range".to_string(),
    })?;
    let denom = count - ddof;
    if denom <= T::zero() {
        return Err(LetoError::StorageError {
            reason: "variance degrees of freedom (n - ddof) must be positive".to_string(),
        });
    }
    Ok(denom)
}
