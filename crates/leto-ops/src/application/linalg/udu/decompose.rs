//! Unpivoted `A = U D Uᵀ` factorization kernel.

use crate::domain::real::RealScalar;
use leto::{ArrayView2, LetoError, Result};

/// Factored form: unit upper-triangular `U` (row-major `n×n`) and the diagonal
/// `D` (length `n`).
pub(super) struct Factored<T> {
    pub(super) u: Vec<T>,
    pub(super) d: Vec<T>,
    pub(super) n: usize,
}

/// Compute `A = U D Uᵀ` for a symmetric matrix.
///
/// Columns are processed from last to first: `D[j] = A[j][j] − Σ_{k>j} U[j][k]²
/// D[k]`, then `U[i][j] = (A[i][j] − Σ_{k>j} U[i][k] U[j][k] D[k]) / D[j]` for
/// `i < j`. Unlike Cholesky, `D[j]` may be negative (symmetric indefinite). A
/// zero pivot (`D[j] ≈ 0`) means the unpivoted factorization does not exist and
/// is reported as an error (full generality needs Bunch–Kaufman pivoting).
pub(super) fn factor<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Factored<T>> {
    let [n, cols] = matrix.shape();
    if n != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![n, cols],
            rhs: vec![n, n],
        });
    }

    // Symmetry + finiteness validation, and a scale for the zero-pivot floor.
    let mut scale = T::ZERO;
    for i in 0..n {
        for j in 0..n {
            let value = *matrix.get([i, j])?;
            if !value.is_finite() {
                return Err(LetoError::StorageError {
                    reason: "UDU input contains a non-finite value".to_string(),
                });
            }
            if value.abs() > scale {
                scale = value.abs();
            }
        }
    }
    let sym_tol = scale.mul(T::ONE.div(T::from_usize(1_000_000_000)));
    for i in 0..n {
        for j in (i + 1)..n {
            if matrix.get([i, j])?.sub(*matrix.get([j, i])?).abs() > sym_tol {
                return Err(LetoError::StorageError {
                    reason: "UDU requires a symmetric matrix".to_string(),
                });
            }
        }
    }

    let pivot_tol = scale.mul(T::ONE.div(T::from_usize(1_000_000_000_000)));
    let mut u = vec![T::ZERO; n * n];
    let mut d = vec![T::ZERO; n];

    for j in (0..n).rev() {
        let mut dj = *matrix.get([j, j])?;
        for k in (j + 1)..n {
            let ujk = u[j * n + k];
            dj = dj.sub(ujk.mul(ujk).mul(d[k]));
        }
        if dj.abs() <= pivot_tol {
            return Err(LetoError::StorageError {
                reason: "UDU encountered a zero pivot (needs symmetric pivoting)".to_string(),
            });
        }
        d[j] = dj;
        u[j * n + j] = T::ONE;

        for i in (0..j).rev() {
            let mut uij = *matrix.get([i, j])?;
            for k in (j + 1)..n {
                uij = uij.sub(u[i * n + k].mul(u[j * n + k]).mul(d[k]));
            }
            u[i * n + j] = uij.div(dj);
        }
    }

    Ok(Factored { u, d, n })
}
