//! Solve and inverse from the complete-pivoting factors `P A Q = L U`.

use super::decompose::Factored;
use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, LetoError, Result};

/// Solve `A x = b` from the factored form, with `b`/`x` as raw slices.
///
/// `P A Q = L U`, so `A x = b ⇔ L U (Qᵀx) = P b`: forward-substitute `L z = Pb`
/// (unit lower), back-substitute `U y = z`, then `x = Q y`. The single source of
/// truth for both [`solve`] and [`inverse`].
fn solve_packed<T: RealScalar>(factored: &Factored<T>, rhs: &[T]) -> Result<Vec<T>> {
    let n = factored.n;
    if factored.rank < n {
        return Err(LetoError::StorageError {
            reason: "FullPivLU solve requires a full-rank (non-singular) matrix".to_string(),
        });
    }
    let a = &factored.lu;

    // z ← P b, then forward-substitute L z = P b in place (L is unit lower).
    let mut z = vec![T::ZERO; n];
    for (k, slot) in z.iter_mut().enumerate() {
        *slot = rhs[factored.row_perm[k]];
    }
    for i in 0..n {
        let mut s = z[i];
        for j in 0..i {
            s = s.sub(a[i * n + j].mul(z[j]));
        }
        z[i] = s;
    }

    // Back-substitute U y = z.
    let mut y = vec![T::ZERO; n];
    for i in (0..n).rev() {
        let mut s = z[i];
        for j in i + 1..n {
            s = s.sub(a[i * n + j].mul(y[j]));
        }
        y[i] = s.div(a[i * n + i]);
    }

    // x = Q y.
    let mut x = vec![T::ZERO; n];
    for (k, &yk) in y.iter().enumerate() {
        x[factored.col_perm[k]] = yk;
    }
    Ok(x)
}

/// Solve `A x = b`.
///
/// # Errors
/// [`LetoError`](leto::LetoError) on a rank-deficient matrix or shape mismatch.
pub(super) fn solve<T: RealScalar>(
    factored: &Factored<T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>> {
    let n = factored.n;
    if rhs.shape() != [n] {
        return Err(LetoError::ShapeMismatch {
            lhs: rhs.shape().to_vec(),
            rhs: vec![n],
        });
    }
    let mut b = vec![T::ZERO; n];
    for (i, slot) in b.iter_mut().enumerate() {
        *slot = *rhs.get([i])?;
    }
    let x = solve_packed(factored, &b)?;
    Array1::from_shape_vec([n], x)
}

/// Inverse `A⁻¹` (solve against each identity column).
///
/// # Errors
/// [`LetoError`](leto::LetoError) on a rank-deficient matrix.
pub(super) fn inverse<T: RealScalar>(factored: &Factored<T>) -> Result<Array2<T>> {
    let n = factored.n;
    let mut inv = vec![T::ZERO; n * n];
    let mut e = vec![T::ZERO; n];
    for col in 0..n {
        for slot in e.iter_mut() {
            *slot = T::ZERO;
        }
        e[col] = T::ONE;
        let x = solve_packed(factored, &e)?;
        for (row, &value) in x.iter().enumerate() {
            inv[row * n + col] = value;
        }
    }
    Array2::from_shape_vec([n, n], inv)
}
