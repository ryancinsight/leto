use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, ArrayView2, LetoError, Result, Storage};

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

    // One bulk row-major copy into working storage + a single finiteness scan
    // (replacing per-element bounds-checked gets); elimination is in place.
    let contiguous = matrix.to_contiguous();
    let src = contiguous.storage().as_slice();
    if !src.iter().all(|value| value.is_finite()) {
        return Err(LetoError::StorageError {
            reason: "LU input contains a non-finite value".to_string(),
        });
    }
    let mut a = src.to_vec();

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
        // Rank-1 elimination of the trailing submatrix. For each row r > k,
        // `a[r, k+1..] -= (a[r,k]/pivot) · a[k, k+1..]` — a fused axpy over two
        // contiguous, disjoint rows (k < r), dispatched through the SIMD
        // `axpy_slice` (SSOT with the matmul/QR row updates). `split_at_mut`
        // separates the pivot row (in `head`) from the trailing rows (`tail`).
        let (head, tail) = a.split_at_mut((k + 1) * n);
        let pivot_row = &head[k * n + (k + 1)..k * n + n];
        for r in (k + 1)..n {
            let base = (r - (k + 1)) * n;
            let factor = tail[base + k].div(pivot);
            tail[base + k] = factor;
            let target = &mut tail[base + (k + 1)..base + n];
            // target += (−factor)·pivot_row  ≡  target −= factor·pivot_row.
            T::axpy_slice(T::ZERO.sub(factor), pivot_row, target);
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
    ///
    /// The packed `L`/`U` is row-major, so each substitution row is a contiguous
    /// slice and the row·partial-solution reduction is a contiguous dot product
    /// dispatched through the SIMD [`Scalar::dot_slice`] (SSOT with the other
    /// contraction paths) — replacing the per-element bounds-checked logical
    /// `Array2::get`, whose `O(n³)` invocation by [`inv`](Self::inv) dominated the
    /// triangular solve. Inverse/solve are correspondingly faster (e.g. they back
    /// the matrix exponential's Padé denominator inverse).
    fn solve_in_place(&self, x: &mut [T]) {
        let n = self.dim();
        let f = self.factors.storage().as_slice(); // row-major packed L (below) / U (on+above)

        // Forward: L · y = P · rhs (unit diagonal). yᵣ = xᵣ − Σ_{c<r} L[r,c]·y_c.
        for r in 1..n {
            let dot = T::dot_slice(&f[r * n..r * n + r], &x[..r]);
            x[r] = x[r].sub(dot);
        }
        // Backward: U · x = y. xᵣ = (yᵣ − Σ_{c>r} U[r,c]·x_c) / U[r,r].
        for r in (0..n).rev() {
            let dot = T::dot_slice(&f[r * n + r + 1..r * n + n], &x[r + 1..n]);
            x[r] = x[r].sub(dot).div(f[r * n + r]);
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
