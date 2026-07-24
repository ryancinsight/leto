//! Pearson correlation matrix (ndarray-stats `pearson_correlation` parity).

use eunomia::FloatElement;

use crate::application::array::Array;
use crate::application::statistics::covariance::covariance;
use crate::domain::error::Result;
use crate::domain::layout::Layout;
use crate::infrastructure::storage::{Storage, VecStorage};

/// Pearson product-moment correlation matrix of a `v × n` observation matrix.
///
/// Same `rowvar = true` convention as [`covariance`]: each row is a variable.
/// The result is the symmetric `v × v` matrix `R` with
/// `R[i, j] = C[i, j] / (σᵢ σⱼ)`, where `C` is the covariance matrix and
/// `σᵢ = √C[i, i]`.
///
/// # Theorem (normalized covariance, ddof-invariant)
/// Writing `C` for the covariance with `ddof = d`, the `1/(n−d)` factor appears
/// once in `C[i, j]` and once in each of `σᵢ`, `σⱼ` under the square roots, so it
/// cancels exactly in the ratio: `R` is independent of `ddof`. By Cauchy–Schwarz
/// on the centered data, `R[i, j] ∈ [−1, 1]` with `R[i, i] = 1`. This kernel uses
/// the population covariance (`ddof = 0`) since the choice is immaterial. ∎
///
/// # Special values
/// A constant variable has zero variance, so its correlations are `0 / 0 = NaN`
/// (matching `numpy.corrcoef`); correlation is genuinely undefined there.
///
/// # Errors
/// [`LetoError`](crate::LetoError) if either dimension is zero (propagated from
/// [`covariance`]).
pub fn pearson_correlation<T, S>(arr: &Array<T, S, 2>) -> Result<Array<T, VecStorage<T>, 2>>
where
    T: FloatElement,
    S: Storage<T>,
{
    // ddof is immaterial (cancels in the ratio); population covariance keeps the
    // denominator positive for every n ≥ 1.
    let cov = covariance(arr, T::ZERO)?;
    let v = cov.shape()[0];
    let cov_view = cov.view();
    let cov_data = cov_view.data(); // C-contiguous, offset 0: [i*v + j]

    let mut std = vec![T::ZERO; v];
    for (i, s) in std.iter_mut().enumerate() {
        *s = cov_data[i * v + i].sqrt();
    }

    let mut out = vec![T::ZERO; v * v];
    for i in 0..v {
        for j in 0..v {
            out[i * v + j] = cov_data[i * v + j] / (std[i] * std[j]);
        }
    }

    Array::new(Layout::c_contiguous([v, v])?, VecStorage::new(out))
}
