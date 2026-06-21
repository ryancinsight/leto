//! Solve and inverse from the complete-pivoting factors `P A Q = L U`.

use super::decompose::Factored;
use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, LetoError, Result};

/// Solve `A x = b` from the factored form, with `b`/`x` as raw slices.
///
/// `P A Q = L U`, so `A x = b ⇔ L U (Qᵀx) = P b`: forward-substitute `L z = Pb`
/// (unit lower), back-substitute `U y = z`, then `x = Q y`. The single source of
/// truth for both [`solve`] and [`inverse`].
fn solve_packed_impl<T: RealScalar>(
    factored: &Factored<T>,
    z: &mut [T],
    x: &mut [T],
) -> Result<()> {
    let n = factored.n;
    if factored.rank < n {
        return Err(LetoError::StorageError {
            reason: "FullPivLU solve requires a full-rank (non-singular) matrix".to_string(),
        });
    }
    let a = &factored.lu;

    // z ← P b, then forward-substitute L z = P b in place (L is unit lower).
    for i in 0..n {
        let mut s = z[i];
        for j in 0..i {
            s = s.sub(a[i * n + j].mul(z[j]));
        }
        z[i] = s;
    }

    // Back-substitute U y = z in place.
    for i in (0..n).rev() {
        let mut s = z[i];
        for j in i + 1..n {
            s = s.sub(a[i * n + j].mul(z[j]));
        }
        z[i] = s.div(a[i * n + i]);
    }

    // x = Q y.
    for k in 0..n {
        x[factored.col_perm[k]] = z[k];
    }
    Ok(())
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

    let mut z_stack = [T::ZERO; 128];
    let mut z_vec = Vec::new();
    let z = if n <= 128 {
        &mut z_stack[..n]
    } else {
        z_vec.resize(n, T::ZERO);
        &mut z_vec[..]
    };

    if let Some(rhs_slice) = rhs.as_slice() {
        for (k, z_k) in z.iter_mut().enumerate().take(n) {
            *z_k = rhs_slice[factored.row_perm[k]];
        }
    } else {
        for (k, z_k) in z.iter_mut().enumerate().take(n) {
            *z_k = *rhs.get([factored.row_perm[k]])?;
        }
    }

    let mut x = vec![T::ZERO; n];
    solve_packed_impl(factored, z, &mut x)?;
    Array1::from_shape_vec([n], x)
}

/// Inverse `A⁻¹` (solve against each identity column).
///
/// # Errors
/// [`LetoError`](leto::LetoError) on a rank-deficient matrix.
pub(super) fn inverse<T: RealScalar>(factored: &Factored<T>) -> Result<Array2<T>> {
    let n = factored.n;
    let mut inv = vec![T::ZERO; n * n];

    let mut z_stack = [T::ZERO; 128];
    let mut z_vec = Vec::new();
    let z = if n <= 128 {
        &mut z_stack[..n]
    } else {
        z_vec.resize(n, T::ZERO);
        &mut z_vec[..]
    };

    let mut x_stack = [T::ZERO; 128];
    let mut x_vec = Vec::new();
    let x = if n <= 128 {
        &mut x_stack[..n]
    } else {
        x_vec.resize(n, T::ZERO);
        &mut x_vec[..]
    };

    for col in 0..n {
        // e = basis vector where e[col] = 1, others 0.
        // row permuted e: z[k] = e[row_perm[k]] = 1 if row_perm[k] == col else 0.
        for (k, z_k) in z.iter_mut().enumerate().take(n) {
            *z_k = if factored.row_perm[k] == col { T::ONE } else { T::ZERO };
        }
        solve_packed_impl(factored, z, x)?;
        for row in 0..n {
            inv[row * n + col] = x[row];
        }
    }
    Array2::from_shape_vec([n, n], inv)
}
