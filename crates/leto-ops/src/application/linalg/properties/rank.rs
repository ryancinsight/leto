//! Numerical matrix rank via the singular-value spectrum.

use crate::domain::real::RealScalar;
use leto::{ArrayView2, Result};

/// Numerical rank: the count of singular values above a relative threshold.
///
/// ```
/// use leto::Array2;
/// use leto_ops::{matrix_rank, matrix_rank_with_tolerance};
///
/// let full = Array2::from_shape_vec([2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]).unwrap();
/// assert_eq!(matrix_rank(&full.view()).unwrap(), 2);
///
/// let nearly_zero = Array2::from_shape_vec([2, 2], vec![1.0_f64, 0.0, 0.0, 1.0e-12]).unwrap();
/// assert_eq!(matrix_rank_with_tolerance(&nearly_zero.view(), 1.0e-9).unwrap(), 1);
/// ```
///
/// # Theorem (rank equals the number of nonzero singular values)
/// For `A = U Σ Vᵀ` with `U, V` having orthonormal columns,
/// `rank(A) = #{ i : σᵢ ≠ 0 }`.
/// *Proof:* orthonormal-column factors are injective linear maps, so they
/// preserve rank; hence `rank(A) = rank(Σ)`. A (rectangular) diagonal matrix's
/// rank is exactly the number of nonzero diagonal entries. ∎
///
/// Exact rank is discontinuous under round-off, so the numerical rank counts
/// `σᵢ > τ · σ_max` for a *relative* tolerance `τ`: any singular value below the
/// noise floor of the largest is treated as structurally zero. This is the same
/// SVD-truncation criterion nalgebra's `rank` uses.
///
/// Single source of truth: the spectrum is computed by
/// [`singular_values`](crate::singular_values) (which already returns zeros for
/// rank-deficient inputs); this function adds only the threshold count — there
/// is no second SVD path.
///
/// # Errors
/// Propagates [`LetoError`](leto::LetoError) for empty or non-finite input
/// (from the singular-value kernel).
#[inline]
pub fn matrix_rank_with_tolerance<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    relative_tolerance: T,
) -> Result<usize> {
    let spectrum = crate::singular_values(matrix)?;
    let max = spectrum
        .iter()
        .copied()
        .fold(T::ZERO, |m, s| if s > m { s } else { m });
    if max <= T::ZERO {
        return Ok(0);
    }
    let threshold = max.mul(relative_tolerance);
    Ok(spectrum.iter().filter(|&&s| s > threshold).count())
}

/// Numerical rank with a conservative default relative tolerance (`1e-9`).
///
/// See [`matrix_rank_with_tolerance`] for the criterion and proof.
///
/// # Errors
/// Propagates [`LetoError`](leto::LetoError) for empty or non-finite input.
#[inline]
pub fn matrix_rank<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<usize> {
    matrix_rank_with_tolerance(matrix, default_rank_tolerance::<T>())
}

/// Conservative relative singular-value floor used by [`matrix_rank`].
#[inline]
fn default_rank_tolerance<T: RealScalar>() -> T {
    T::ONE.div(T::from_usize(1_000_000_000))
}
