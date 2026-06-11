use crate::domain::real::RealScalar;
use leto::{Array1, ArrayView1, ArrayView2, LetoError, Result};

/// Householder QR factorization of an `m × n` matrix with `m ≥ n`:
/// `A = Q · R` with `Q` orthogonal (`m × m`, held implicitly as reflectors)
/// and `R` upper-triangular.
///
/// The factor storage is the standard compact form: `R` occupies the upper
/// triangle of the working matrix, each Householder vector's tail occupies
/// the column below the diagonal, and the vector heads and `β = 2/(vᵀv)`
/// coefficients are stored alongside. `Q` is never materialized — solves
/// apply the reflectors directly, which is both the fast and the
/// memory-lean form.
///
/// Generic over `T: RealScalar`, native-precision arithmetic. Driver: CFDrs
/// `cfd-math` least-squares paths.
#[derive(Debug, Clone)]
pub struct QrDecomposition<T> {
    /// Row-major `m × n` packed factors (R upper, reflector tails below).
    packed: Vec<T>,
    /// Householder vector head components `v_k[k]` (diagonal slots hold R).
    heads: Vec<T>,
    /// `β_k = 2 / (v_kᵀ v_k)` per reflector.
    betas: Vec<T>,
    rows: usize,
    cols: usize,
}

/// Compute the Householder QR factorization of an `m × n` matrix, `m ≥ n`.
///
/// The input may be strided/transposed; it is copied once into row-major
/// working storage. Underdetermined shapes (`m < n`), non-finite values, and
/// rank-deficient columns (zero pivot norm to working precision) are
/// rejected with distinct error reasons.
pub fn qr_decompose<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<QrDecomposition<T>> {
    let [rows, cols] = matrix.shape();
    if rows < cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![cols, cols],
        });
    }

    let mut a = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let value = *matrix.get([r, c])?;
            if !value.is_finite() {
                return Err(LetoError::StorageError {
                    reason: "QR input contains a non-finite value".to_string(),
                });
            }
            a.push(value);
        }
    }

    let mut heads = vec![T::ZERO; cols];
    let mut betas = vec![T::ZERO; cols];

    for k in 0..cols {
        // ‖x‖ for the pivot column below (and including) the diagonal.
        let mut norm_sq = T::ZERO;
        for r in k..rows {
            let x = a[r * cols + k];
            norm_sq = norm_sq.add(x.mul(x));
        }
        let norm = norm_sq.sqrt();
        if norm == T::ZERO {
            return Err(LetoError::StorageError {
                reason: format!("QR pivot column {k} has zero norm: matrix is rank-deficient"),
            });
        }

        // alpha = -sign(x₀)·‖x‖ for cancellation-free head computation.
        let pivot = a[k * cols + k];
        let alpha = if pivot > T::ZERO {
            T::ZERO.sub(norm)
        } else {
            norm
        };
        let head = pivot.sub(alpha);

        // vᵀv = head² + Σ tail²  (tail entries stay in place below the diagonal).
        let mut v_norm_sq = head.mul(head);
        for r in (k + 1)..rows {
            let x = a[r * cols + k];
            v_norm_sq = v_norm_sq.add(x.mul(x));
        }
        let beta = T::ONE.add(T::ONE).div(v_norm_sq);

        // Apply H = I − β·v·vᵀ to the trailing columns.
        for c in (k + 1)..cols {
            let mut s = head.mul(a[k * cols + c]);
            for r in (k + 1)..rows {
                s = s.add(a[r * cols + k].mul(a[r * cols + c]));
            }
            let bs = beta.mul(s);
            a[k * cols + c] = a[k * cols + c].sub(bs.mul(head));
            for r in (k + 1)..rows {
                let update = bs.mul(a[r * cols + k]);
                a[r * cols + c] = a[r * cols + c].sub(update);
            }
        }

        a[k * cols + k] = alpha; // R diagonal; v's tail remains below.
        heads[k] = head;
        betas[k] = beta;
    }

    Ok(QrDecomposition {
        packed: a,
        heads,
        betas,
        rows,
        cols,
    })
}

impl<T: RealScalar> QrDecomposition<T> {
    /// `(rows, cols)` of the factored matrix.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Solve `min ‖A·x − rhs‖₂` (least squares; exact solve when `m = n`).
    ///
    /// Applies the stored reflectors to `rhs` (computing `Qᵀ·rhs` without
    /// materializing `Q`), then back-substitutes against `R`.
    pub fn solve_least_squares(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        let (m, n) = (self.rows, self.cols);
        if rhs.shape() != [m] {
            return Err(LetoError::ShapeMismatch {
                lhs: rhs.shape().to_vec(),
                rhs: vec![m],
            });
        }

        let mut y = Vec::with_capacity(m);
        for k in 0..m {
            y.push(*rhs.get([k])?);
        }

        // y ← Qᵀ·y, one reflector at a time.
        for k in 0..n {
            let mut s = self.heads[k].mul(y[k]);
            for r in (k + 1)..m {
                s = s.add(self.packed[r * n + k].mul(y[r]));
            }
            let bs = self.betas[k].mul(s);
            y[k] = y[k].sub(bs.mul(self.heads[k]));
            for r in (k + 1)..m {
                let update = bs.mul(self.packed[r * n + k]);
                y[r] = y[r].sub(update);
            }
        }

        // Back-substitute R·x = y[..n].
        let mut x = y;
        x.truncate(n);
        for r in (0..n).rev() {
            let mut acc = x[r];
            for (offset, &solved) in x[(r + 1)..n].iter().enumerate() {
                acc = acc.sub(self.packed[r * n + r + 1 + offset].mul(solved));
            }
            x[r] = acc.div(self.packed[r * n + r]);
        }
        Array1::from_shape_vec([n], x)
    }
}

/// Convenience: factor and solve `min ‖A·x − rhs‖₂` in one call.
#[inline]
pub fn solve_least_squares<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>> {
    qr_decompose(matrix)?.solve_least_squares(rhs)
}
