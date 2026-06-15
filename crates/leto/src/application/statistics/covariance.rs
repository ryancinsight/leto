//! Covariance matrix of a set of variables (ndarray-stats `cov` parity).

use num_traits::Float;

use crate::application::array::Array;
use crate::application::reduction::variance::degrees_of_freedom;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::infrastructure::storage::{Storage, VecStorage};

/// Covariance matrix of the `v` variables in a `v × n` observation matrix.
///
/// Following the ndarray-stats / numpy `rowvar = true` convention, **each row is
/// a variable** and **each column an observation**: `arr[i, k]` is the `k`-th
/// observation of variable `i`. The result is the symmetric `v × v` matrix `C`
/// with `C[i, j] = (1 / (n − ddof)) Σₖ (xᵢₖ − x̄ᵢ)(xⱼₖ − x̄ⱼ)`.
///
/// # Theorem (covariance ↔ variance on the diagonal)
/// `C[i, i] = (1 / (n − ddof)) Σₖ (xᵢₖ − x̄ᵢ)²` is exactly the `ddof`-variance of
/// row `i`; the off-diagonal entries are the symmetric pairwise covariances. `C`
/// is symmetric positive-semidefinite (`zᵀ C z = (1/(n−ddof))‖Xᶜᵀz‖² ≥ 0` for
/// the centered data `Xᶜ`). ∎
///
/// # Numerical note
/// Two-pass: variables are centered by their means before the cross-products are
/// accumulated, avoiding the catastrophic cancellation of the one-pass
/// `Σxᵢₖxⱼₖ − n x̄ᵢx̄ⱼ` form (see [`var_all`](crate::application::reduction::var_all)).
/// One `v × n` centered buffer is allocated; cross-products read it contiguously.
///
/// # Errors
/// [`LetoError`] if either dimension is zero or `n − ddof ≤ 0`.
pub fn covariance<T, S>(arr: &Array<T, S, 2>, ddof: T) -> Result<Array<T, VecStorage<T>, 2>>
where
    T: Float,
    S: Storage<T>,
{
    let [v, n] = arr.shape();
    if v == 0 || n == 0 {
        return Err(LetoError::StorageError {
            reason: "covariance over an empty matrix is undefined".to_string(),
        });
    }
    let denom = degrees_of_freedom(n, ddof)?;
    let nf = T::from(n).ok_or(LetoError::StorageError {
        reason: "observation count exceeds float precision range".to_string(),
    })?;

    let view = arr.view();
    let layout = view.layout();
    let data = view.data();

    // Center each variable by its mean into a contiguous v × n buffer.
    let mut centered = vec![T::zero(); v * n];
    for i in 0..v {
        let mut sum = T::zero();
        for k in 0..n {
            sum = sum + data[layout.offset_of([i, k])?];
        }
        let mean = sum / nf;
        for k in 0..n {
            centered[i * n + k] = data[layout.offset_of([i, k])?] - mean;
        }
    }

    // Symmetric cross-products: compute the upper triangle, mirror to the lower.
    let mut out = vec![T::zero(); v * v];
    for i in 0..v {
        for j in i..v {
            let mut acc = T::zero();
            for k in 0..n {
                acc = acc + centered[i * n + k] * centered[j * n + k];
            }
            let c = acc / denom;
            out[i * v + j] = c;
            out[j * v + i] = c;
        }
    }

    Array::new(Layout::c_contiguous([v, v])?, VecStorage::new(out))
}
