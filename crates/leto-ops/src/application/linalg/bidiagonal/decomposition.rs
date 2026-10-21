use super::{colmajor, reduce};
use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, LetoError, Result};

/// Bidiagonal decomposition `A = U B Vᵀ` (`B` upper bidiagonal).
#[derive(Debug, Clone)]
pub struct BidiagonalDecomposition<T> {
    u: Array2<T>,
    b: Array2<T>,
    v: Array2<T>,
}

impl<T: RealScalar> BidiagonalDecomposition<T> {
    /// Left orthogonal factor `U` (`m × m`).
    #[must_use]
    pub fn u(&self) -> &Array2<T> {
        &self.u
    }

    /// Upper bidiagonal factor `B` (`m × n`).
    #[must_use]
    pub fn b(&self) -> &Array2<T> {
        &self.b
    }

    /// Right orthogonal factor `V` (`n × n`).
    #[must_use]
    pub fn v(&self) -> &Array2<T> {
        &self.v
    }
}

/// Golub–Kahan bidiagonalization of a tall-or-square matrix (`m ≥ n`).
///
/// # Errors
/// [`LetoError::ShapeMismatch`] when `m < n` (transpose wide inputs);
/// [`LetoError::StorageError`] for a non-finite entry.
pub fn bidiagonalize<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<BidiagonalDecomposition<T>> {
    let [m, n] = matrix.shape();
    if m < n {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![m, n],
            rhs: vec![n, n],
        });
    }
    // Finiteness is validated inside the reduction's single bulk input copy.
    let (u, b, v) = reduce::reduce_to_bidiagonal(matrix, m, n)?;
    Ok(BidiagonalDecomposition {
        u: Array2::from_shape_vec([m, m], u).expect("U shape matches storage"),
        b: Array2::from_shape_vec([m, n], b).expect("B shape matches storage"),
        v: Array2::from_shape_vec([n, n], v).expect("V shape matches storage"),
    })
}

/// Bidiagonal `(d, e)` of `A` (`m ≥ n`) via the column-major working buffer (the
/// SVD values path's locality experiment; `σ(A) = σ(B)`).
///
/// # Preconditions
/// Entries must be finite. Finiteness is the caller's responsibility — the sole
/// caller, [`crate::singular_values`], validates it once via `validate_input`
/// (SSOT), so re-scanning here would be a redundant `O(mn)` pass on the SVD hot
/// path; a `debug_assert` keeps the safety net in debug/test builds.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] for `m < n`.
pub(crate) fn bidiagonal_diag_colmajor<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<(Vec<T>, Vec<T>)> {
    let [m, n] = matrix.shape();
    if m < n {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![m, n],
            rhs: vec![n, n],
        });
    }
    debug_assert!(
        if let Some(slice) = matrix.as_slice() {
            slice.iter().all(|x| x.is_finite())
        } else {
            matrix.iter().all(|x| x.is_finite())
        },
        "bidiagonal_diag_colmajor precondition: caller (validate_input) guarantees finite entries"
    );
    Ok(colmajor::reduce_values(matrix))
}
