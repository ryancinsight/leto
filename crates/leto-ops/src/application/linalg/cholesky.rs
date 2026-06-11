use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, ArrayView2, LetoError, Result};

/// Cholesky factorization of a symmetric positive-definite matrix:
/// `A = L · Lᵀ` with `L` lower-triangular.
///
/// Generic over `T: RealScalar`; the factorization runs in the native
/// precision of `T` (no hidden widening). Positive-definiteness is verified
/// constructively: a non-positive diagonal pivot during elimination rejects
/// the input, which simultaneously rejects indefinite and (to working
/// precision) singular matrices. Driver: CFDrs `cfd-math` SPD solver paths.
#[derive(Debug, Clone)]
pub struct CholeskyDecomposition<T> {
    lower: Array2<T>,
    dim: usize,
}

/// Compute the Cholesky factor of a symmetric positive-definite matrix.
///
/// The input may be strided/transposed; only the lower triangle (and the
/// diagonal) is read, so symmetric storage conventions that populate one
/// triangle work unchanged. Non-square and non-finite inputs are rejected;
/// a non-positive pivot rejects with a distinct "not positive-definite"
/// reason.
pub fn cholesky_decompose<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<CholeskyDecomposition<T>> {
    let [rows, cols] = matrix.shape();
    if rows != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows, rows],
        });
    }
    let n = rows;

    let mut l = vec![T::ZERO; n * n];
    for r in 0..n {
        for c in 0..=r {
            let a_rc = *matrix.get([r, c])?;
            if !a_rc.is_finite() {
                return Err(LetoError::StorageError {
                    reason: "Cholesky input contains a non-finite value".to_string(),
                });
            }
            // acc = A[r][c] - Σ_{k<c} L[r][k]·L[c][k]
            let mut acc = a_rc;
            for k in 0..c {
                acc = acc.sub(l[r * n + k].mul(l[c * n + k]));
            }
            if r == c {
                if acc <= T::ZERO {
                    return Err(LetoError::StorageError {
                        reason: format!(
                            "Cholesky pivot {r} is non-positive: matrix is not positive-definite"
                        ),
                    });
                }
                l[r * n + c] = acc.sqrt();
            } else {
                l[r * n + c] = acc.div(l[c * n + c]);
            }
        }
    }

    Ok(CholeskyDecomposition {
        lower: Array2::from_shape_vec([n, n], l).expect("square factor storage"),
        dim: n,
    })
}

impl<T: RealScalar> CholeskyDecomposition<T> {
    /// Matrix dimension `n`.
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The lower-triangular factor `L` (strict upper triangle is zero).
    #[must_use]
    #[inline]
    pub fn lower(&self) -> &Array2<T> {
        &self.lower
    }

    /// Solve `A · x = rhs` via `L · y = rhs` then `Lᵀ · x = y`.
    pub fn solve(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        let n = self.dim;
        if rhs.shape() != [n] {
            return Err(LetoError::ShapeMismatch {
                lhs: rhs.shape().to_vec(),
                rhs: vec![n],
            });
        }
        let l = |r: usize, c: usize| *self.lower.get([r, c]).expect("factor in bounds");

        let mut x = Vec::with_capacity(n);
        for k in 0..n {
            x.push(*rhs.get([k])?);
        }
        // Forward: L · y = rhs.
        for r in 0..n {
            let mut acc = x[r];
            for (c, &solved) in x[..r].iter().enumerate() {
                acc = acc.sub(l(r, c).mul(solved));
            }
            x[r] = acc.div(l(r, r));
        }
        // Backward: Lᵀ · x = y (Lᵀ[r][c] = L[c][r]).
        for r in (0..n).rev() {
            let mut acc = x[r];
            for (offset, &solved) in x[(r + 1)..n].iter().enumerate() {
                acc = acc.sub(l(r + 1 + offset, r).mul(solved));
            }
            x[r] = acc.div(l(r, r));
        }
        Array1::from_shape_vec([n], x)
    }
}
