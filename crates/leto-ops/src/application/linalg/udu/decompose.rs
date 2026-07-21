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

    // One bulk row-major copy only when not already contiguous.
    let contiguous;
    let a = if let Some(slice) = matrix.as_slice() {
        slice
    } else {
        contiguous = matrix.to_contiguous();
        leto::Storage::as_slice(contiguous.storage())
    };

    // Symmetry + finiteness validation, and a scale for the zero-pivot floor.
    let mut scale = T::ZERO;
    for i in 0..n {
        for j in 0..n {
            let value = a[i * n + j];
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
            if a[i * n + j].sub(a[j * n + i]).abs() > sym_tol {
                return Err(LetoError::StorageError {
                    reason: "UDU requires a symmetric matrix".to_string(),
                });
            }
        }
    }

    let pivot_tol = scale.mul(T::ONE.div(T::from_usize(1_000_000_000_000)));
    let mut u = vec![T::ZERO; n * n];
    let mut d = vec![T::ZERO; n];
    // Reusable hoist buffer for the loop-invariant weights `w[k] = u[j][k]·d[k]`.
    let mut w = vec![T::ZERO; n];

    for j in (0..n).rev() {
        // Hoist `w = u[j][j+1..] ⊙ d[j+1..]`: these weights are loop-invariant
        // across the `i`-loop below and shared with this pivot reduction, so the
        // hoist drops the `O(n³)` `u[j][k]·d[k]` recompute and turns both
        // reductions into the SIMD `dot_slice` (SSOT with Cholesky/`solve`).
        let w = &mut w[..n - (j + 1)];
        for (wk, (&ujk, &dk)) in w
            .iter_mut()
            .zip(u[j * n + (j + 1)..j * n + n].iter().zip(&d[j + 1..n]))
        {
            *wk = ujk.mul(dk);
        }
        // dj = a[j][j] − Σ_{k>j} u[j][k]·w[k]  (≡ u[j][k]²·d[k])
        let dj = a[j * n + j].sub(T::dot_slice(&u[j * n + (j + 1)..j * n + n], w));
        if dj.abs() <= pivot_tol {
            return Err(LetoError::StorageError {
                reason: "UDU encountered a zero pivot (needs symmetric pivoting)".to_string(),
            });
        }
        d[j] = dj;
        u[j * n + j] = T::ONE;

        for i in (0..j).rev() {
            // uij = a[i][j] − Σ_{k>j} u[i][k]·w[k]
            let uij = a[i * n + j].sub(T::dot_slice(&u[i * n + (j + 1)..i * n + n], w));
            u[i * n + j] = uij.div(dj);
        }
    }

    Ok(Factored { u, d, n })
}
