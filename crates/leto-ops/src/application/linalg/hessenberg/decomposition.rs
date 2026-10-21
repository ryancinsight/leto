use super::reduce;
use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, LetoError, Result};

/// Upper Hessenberg decomposition `A = Q H Qᵀ`.
#[derive(Debug, Clone)]
pub struct HessenbergDecomposition<T> {
    q: Array2<T>,
    h: Array2<T>,
}

impl<T: RealScalar> HessenbergDecomposition<T> {
    /// Orthogonal factor `Q` (`n × n`).
    #[must_use]
    pub fn q(&self) -> &Array2<T> {
        &self.q
    }

    /// Upper Hessenberg factor `H` (`n × n`), zero below the first subdiagonal.
    #[must_use]
    pub fn h(&self) -> &Array2<T> {
        &self.h
    }
}

/// Validate square shape, returning the dimension `n`. Finiteness is checked in
/// the reduction's single bulk input copy (SSOT), not by a second pre-scan here.
fn validate_square<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<usize> {
    let [rows, cols] = matrix.shape();
    if rows != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows, rows],
        });
    }
    Ok(rows)
}

/// Reduce a square matrix to upper Hessenberg form `A = Q H Qᵀ`, accumulating `Q`.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] for non-square input;
/// [`LetoError::StorageError`] for a non-finite entry.
pub fn hessenberg<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<HessenbergDecomposition<T>> {
    validate_square(matrix)?;
    let (h, q, n) = reduce::reduce_to_hessenberg::<T, true>(matrix)?;
    Ok(HessenbergDecomposition {
        q: Array2::from_shape_vec([n, n], q).expect("Q shape matches storage"),
        h: Array2::from_shape_vec([n, n], h).expect("H shape matches storage"),
    })
}

/// Reduce a square matrix to upper Hessenberg form, returning only the row-major
/// `H` factor and dimension `n` — the `Q` accumulation is elided at compile time.
///
/// For the eigenvalue problem `H = Qᵀ A Q` is similar to `A` (spectrum
/// preserved), so the Schur/Francis stage that follows never reads `Q`. Skipping
/// its accumulation removes an `O(n³)` cost (one `apply_right` per reflector plus
/// the `n²` allocation) — about half the reduction work. SSOT: shares the single
/// [`reduce::reduce_to_hessenberg`] kernel via its `ACCUMULATE_Q = false`
/// instantiation.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] for non-square input;
/// [`LetoError::StorageError`] for a non-finite entry.
pub(crate) fn hessenberg_values<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<(Vec<T>, usize)> {
    validate_square(matrix)?;
    let (h, _q, n) = reduce::reduce_to_hessenberg::<T, false>(matrix)?;
    Ok((h, n))
}
