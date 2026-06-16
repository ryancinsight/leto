use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, ArrayView2, LetoError, Result};

/// LU decomposition with partial pivoting: `P · A = L · U`.
///
/// `L` (unit lower-triangular) and `U` (upper-triangular) are packed into one
/// row-major `n × n` factor matrix (`L` strictly below the diagonal, `U` on
/// and above it); `pivots` records the row permutation; `sign` is the
/// permutation parity (`+1`/`-1`), used by [`det`](Self::det).
///
/// Generic over `T: RealScalar`; elimination runs in the native precision of
/// `T` per the `Scalar` contract (no hidden widening — a caller needing a
/// wider working precision converts the input explicitly). Driver: CFDrs
/// `cfd-math` dense solver paths (nalgebra replacement Stage A1).
#[derive(Debug, Clone)]
pub struct LuDecomposition<T> {
    factors: Array2<T>,
    pivots: Vec<usize>,
    sign: i8,
}

/// Compute the partially pivoted LU decomposition of a square matrix.
///
/// The input may be strided/transposed; it is copied once into row-major
/// working storage. Returns [`LetoError::ShapeMismatch`] for non-square
/// input and [`LetoError::StorageError`] when a pivot column is exactly
/// zero (the matrix is singular to working precision).
pub fn lu_decompose<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<LuDecomposition<T>> {
    let [rows, cols] = matrix.shape();
    if rows != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows, rows],
        });
    }
    let n = rows;

    // One copy into row-major working storage; elimination is in place.
    let mut a = Vec::with_capacity(n * n);
    for r in 0..n {
        for c in 0..n {
            let value = *matrix.get([r, c])?;
            if !value.is_finite() {
                return Err(LetoError::StorageError {
                    reason: "LU input contains a non-finite value".to_string(),
                });
            }
            a.push(value);
        }
    }

    let mut pivots: Vec<usize> = (0..n).collect();
    let mut sign = 1i8;

    for k in 0..n {
        // Partial pivot: row with the largest |a[r][k]| for r >= k.
        let mut pivot_row = k;
        let mut pivot_mag = a[k * n + k].abs();
        for r in (k + 1)..n {
            let mag = a[r * n + k].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = r;
            }
        }
        if pivot_mag == T::ZERO {
            return Err(LetoError::StorageError {
                reason: format!("LU pivot column {k} is exactly zero: matrix is singular"),
            });
        }
        if pivot_row != k {
            for c in 0..n {
                a.swap(k * n + c, pivot_row * n + c);
            }
            pivots.swap(k, pivot_row);
            sign = -sign;
        }

        let pivot = a[k * n + k];
        for r in (k + 1)..n {
            let factor = a[r * n + k].div(pivot);
            a[r * n + k] = factor;
            for c in (k + 1)..n {
                let update = factor.mul(a[k * n + c]);
                a[r * n + c] = a[r * n + c].sub(update);
            }
        }
    }

    Ok(LuDecomposition {
        factors: Array2::from_shape_vec([n, n], a).expect("square factor storage"),
        pivots,
        sign,
    })
}

impl<T: RealScalar> LuDecomposition<T> {
    /// Matrix dimension `n`.
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.pivots.len()
    }

    /// The packed `L`/`U` factor matrix (unit `L` strictly below the
    /// diagonal, `U` on and above it), in row-major order.
    #[must_use]
    #[inline]
    pub fn factors(&self) -> &Array2<T> {
        &self.factors
    }

    /// The permutation pivots vector.
    #[must_use]
    #[inline]
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }

    /// Determinant of the original matrix: parity × product of `U`'s diagonal.
    #[must_use]
    pub fn det(&self) -> T {
        let n = self.dim();
        let mut det = if self.sign >= 0 {
            T::ONE
        } else {
            T::ZERO.sub(T::ONE)
        };
        for k in 0..n {
            det = det.mul(*self.factors.get([k, k]).expect("diagonal in bounds"));
        }
        det
    }

    /// Solve `A · x = rhs` for one right-hand-side vector.
    pub fn solve(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        let n = self.dim();
        if rhs.shape() != [n] {
            return Err(LetoError::ShapeMismatch {
                lhs: rhs.shape().to_vec(),
                rhs: vec![n],
            });
        }

        // Apply the row permutation while gathering the RHS.
        let mut x = Vec::with_capacity(n);
        for k in 0..n {
            x.push(*rhs.get([self.pivots[k]])?);
        }
        self.solve_in_place(&mut x);
        Array1::from_shape_vec([n], x)
    }

    /// Inverse of the original matrix, solving against the identity columns.
    pub fn inv(&self) -> Result<Array2<T>> {
        let n = self.dim();
        let mut out = vec![T::ZERO; n * n];
        let mut column = vec![T::ZERO; n];
        for j in 0..n {
            // Permuted j-th identity column.
            for (k, slot) in column.iter_mut().enumerate() {
                *slot = if self.pivots[k] == j { T::ONE } else { T::ZERO };
            }
            self.solve_in_place(&mut column);
            for r in 0..n {
                out[r * n + j] = column[r];
            }
        }
        Array2::from_shape_vec([n, n], out)
    }

    /// Forward- then back-substitution over the packed factors. `x` arrives
    /// already permuted and leaves holding the solution.
    fn solve_in_place(&self, x: &mut [T]) {
        let n = self.dim();
        let a = |r: usize, c: usize| *self.factors.get([r, c]).expect("factor in bounds");

        // Forward: L · y = P · rhs (unit diagonal).
        for r in 1..n {
            let mut acc = x[r];
            for (c, &solved) in x[..r].iter().enumerate() {
                acc = acc.sub(a(r, c).mul(solved));
            }
            x[r] = acc;
        }
        // Backward: U · x = y.
        for r in (0..n).rev() {
            let mut acc = x[r];
            for (offset, &solved) in x[(r + 1)..n].iter().enumerate() {
                acc = acc.sub(a(r, r + 1 + offset).mul(solved));
            }
            x[r] = acc.div(a(r, r));
        }
    }
}

/// Convenience: decompose and solve `A · x = rhs` in one call.
#[inline]
pub fn solve<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>> {
    lu_decompose(matrix)?.solve(rhs)
}

/// Convenience: determinant via LU. Singular matrices return zero rather
/// than an error (the zero-pivot rejection applies to solving, not to the
/// determinant value itself).
pub fn det<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<T> {
    match lu_decompose(matrix) {
        Ok(decomposition) => Ok(decomposition.det()),
        Err(LetoError::StorageError { reason }) if reason.contains("singular") => Ok(T::ZERO),
        Err(other) => Err(other),
    }
}

/// Convenience: matrix inverse via LU.
#[inline]
pub fn inv<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>> {
    lu_decompose(matrix)?.inv()
}
