use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, ArrayView2, LetoError, Result, Storage};

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

    // One bulk row-major copy + a single finiteness scan, then contiguous
    // indexing in the factorization — replacing per-element bounds-checked,
    // stride-recomputing `matrix.get` calls inside the hot loop.
    let contiguous = matrix.to_contiguous();
    let a = contiguous.storage().as_slice();
    if !a.iter().all(|value| value.is_finite()) {
        return Err(LetoError::StorageError {
            reason: "Cholesky input contains a non-finite value".to_string(),
        });
    }

    let mut l = vec![T::ZERO; n * n];
    for r in 0..n {
        for c in 0..=r {
            // acc = A[r][c] - Σ_{k<c} L[r][k]·L[c][k]
            let mut acc = a[r * n + c];
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

    /// Determinant of the original SPD matrix: `Π diag(L)^2`.
    #[must_use]
    pub fn det(&self) -> T {
        let mut det = T::ONE;
        for k in 0..self.dim {
            let diagonal = *self.lower.get([k, k]).expect("factor diagonal in bounds");
            det = det.mul(diagonal.mul(diagonal));
        }
        det
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
        let mut x = Vec::with_capacity(n);
        for k in 0..n {
            x.push(*rhs.get([k])?);
        }
        self.solve_in_place(&mut x);
        Array1::from_shape_vec([n], x)
    }

    /// Inverse of the original SPD matrix, solving against identity columns.
    pub fn inv(&self) -> Result<Array2<T>> {
        let n = self.dim;
        let mut out = vec![T::ZERO; n * n];
        let mut column = vec![T::ZERO; n];
        for col in 0..n {
            column.fill(T::ZERO);
            column[col] = T::ONE;
            self.solve_in_place(&mut column);
            for row in 0..n {
                out[row * n + col] = column[row];
            }
        }
        Array2::from_shape_vec([n, n], out)
    }

    fn solve_in_place(&self, x: &mut [T]) {
        let n = self.dim;
        let l = |r: usize, c: usize| *self.lower.get([r, c]).expect("factor in bounds");

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
    }
}

/// Convenience: decompose and solve `A · x = rhs` through Cholesky.
#[inline]
pub fn cholesky_solve<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>> {
    cholesky_decompose(matrix)?.solve(rhs)
}

/// Convenience: determinant via Cholesky factorization.
#[inline]
pub fn cholesky_det<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<T> {
    Ok(cholesky_decompose(matrix)?.det())
}

/// Convenience: inverse via Cholesky factorization.
#[inline]
pub fn cholesky_inv<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>> {
    cholesky_decompose(matrix)?.inv()
}
