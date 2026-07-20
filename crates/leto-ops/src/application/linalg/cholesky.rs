use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, LetoError, Result, Storage};

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
    let mut a = matrix.to_contiguous().into_storage().into_inner();
    if !a.iter().all(|value| value.is_finite()) {
        return Err(LetoError::StorageError {
            reason: "Cholesky input contains a non-finite value".to_string(),
        });
    }

    for r in 0..n {
        for c in 0..=r {
            // acc = A[r][c] − Σ_{k<c} L[r][k]·L[c][k]: the Cholesky–Crout inner
            // product over two contiguous row prefixes, dispatched through the
            // SIMD `dot_slice` (SSOT with `solve_in_place`). The scalar reduction
            // carries a loop-borne FP dependency and never autovectorizes, so
            // this is the dominant `O(n³/3)` win. Reduction reorder is
            // backward-stable, within the Cholesky differential oracle's tolerance.
            let dot = T::dot_slice(&a[r * n..r * n + c], &a[c * n..c * n + c]);
            let acc = a[r * n + c].sub(dot);
            if r == c {
                if acc <= T::ZERO {
                    return Err(LetoError::StorageError {
                        reason: format!(
                            "Cholesky pivot {r} is non-positive: matrix is not positive-definite"
                        ),
                    });
                }
                a[r * n + c] = acc.sqrt();
            } else {
                a[r * n + c] = acc.div(a[c * n + c]);
            }
        }
        for c in (r + 1)..n {
            a[r * n + c] = T::ZERO;
        }
    }

    Ok(CholeskyDecomposition {
        lower: Array2::from_shape_vec([n, n], a).expect("square factor storage"),
        dim: n,
    })
}

impl<T: RealScalar> CholeskyDecomposition<T> {
    /// Construct a Cholesky decomposition directly from its raw lower factor.
    #[must_use]
    #[inline]
    pub fn from_raw_parts(lower: Array2<T>) -> Self {
        let dim = lower.shape()[0];
        Self { lower, dim }
    }

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

    /// Solve `A · x = rhs` directly into a caller-owned view `out`.
    #[allow(clippy::needless_range_loop)]
    pub fn solve_into(
        &self,
        rhs: &ArrayView1<'_, T>,
        out: &mut ArrayViewMut1<'_, T>,
    ) -> Result<()> {
        let n = self.dim;
        if rhs.shape() != [n] {
            return Err(LetoError::ShapeMismatch {
                lhs: rhs.shape().to_vec(),
                rhs: vec![n],
            });
        }
        if out.shape() != [n] {
            return Err(LetoError::ShapeMismatch {
                lhs: out.shape().to_vec(),
                rhs: vec![n],
            });
        }

        if let Some(out_slice) = out.as_mut_slice() {
            if let Some(rhs_slice) = rhs.as_slice() {
                out_slice[..n].copy_from_slice(&rhs_slice[..n]);
            } else {
                for k in 0..n {
                    out_slice[k] = *rhs.get([k])?;
                }
            }
            self.solve_in_place(out_slice);
        } else {
            if let Some(rhs_slice) = rhs.as_slice() {
                for k in 0..n {
                    *out.get_mut([k])? = rhs_slice[k];
                }
            } else {
                for k in 0..n {
                    *out.get_mut([k])? = *rhs.get([k])?;
                }
            }
            let l = self.lower.storage().as_slice();

            // Forward: L · y = rhs. yᵣ = (xᵣ − Σ_{c<r} L[r,c]·y_c) / L[r,r].
            for r in 0..n {
                let mut dot = T::ZERO;
                for c in 0..r {
                    let l_rc = l[r * n + c];
                    let x_c = *out.get([c])?;
                    dot = dot.add(l_rc.mul(x_c));
                }
                let out_r = out.get_mut([r])?;
                *out_r = out_r.sub(dot).div(l[r * n + r]);
            }
            // Backward: Lᵀ · x = y (Lᵀ[r][c] = L[c][r], a strided column read).
            for r in (0..n).rev() {
                let mut acc = *out.get([r])?;
                for c in (r + 1)..n {
                    let l_cr = l[c * n + r];
                    let solved = *out.get([c])?;
                    acc = acc.sub(l_cr.mul(solved));
                }
                let out_r = out.get_mut([r])?;
                *out_r = acc.div(l[r * n + r]);
            }
        }
        Ok(())
    }

    /// Solve `A · x = rhs` via `L · y = rhs` then `Lᵀ · x = y`.
    pub fn solve(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        let n = self.dim;
        let mut out = Array1::from_elem([n], T::ZERO);
        self.solve_into(rhs, &mut out.view_mut())?;
        Ok(out)
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
        // Row-major `L`; the per-element bounds-checked logical `get` is `O(n³)`
        // under `inv` (n solves), so read the contiguous slice once. The forward
        // sweep reduces over a contiguous row (SIMD `dot_slice`, SSOT); the
        // backward sweep reduces down a column (strided), so it stays a direct
        // indexed scalar loop — still free of the `get` overhead.
        let l = self.lower.storage().as_slice();

        // Forward: L · y = rhs. yᵣ = (xᵣ − Σ_{c<r} L[r,c]·y_c) / L[r,r].
        for r in 0..n {
            let dot = T::dot_slice(&l[r * n..r * n + r], &x[..r]);
            x[r] = x[r].sub(dot).div(l[r * n + r]);
        }
        // Backward: Lᵀ · x = y (Lᵀ[r][c] = L[c][r], a strided column read).
        for r in (0..n).rev() {
            let mut acc = x[r];
            for (offset, &solved) in x[(r + 1)..n].iter().enumerate() {
                acc = acc.sub(l[(r + 1 + offset) * n + r].mul(solved));
            }
            x[r] = acc.div(l[r * n + r]);
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
