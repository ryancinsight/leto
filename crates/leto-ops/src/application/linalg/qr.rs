//! Householder QR factorization `A = Q R` (compact reflector storage).
//!
//! # Theorem (existence and the Householder construction)
//! Every `A ∈ ℝ^{m×n}` with `m ≥ n` admits `A = Q R` with `Q ∈ ℝ^{m×m}`
//! orthogonal and `R ∈ ℝ^{m×n}` upper-triangular.
//!
//! *Proof (constructive — exactly the reflectors [`qr_decompose`] applies).* For
//! pivot column `k`, let `x = A[k.., k]`. The Householder reflector
//! `H_k = I − β_k v_k v_kᵀ`, with `v_k = x − α e₁`, `α = −sign(x₀)‖x‖₂`, and
//! `β_k = 2 / (v_kᵀ v_k)`, is symmetric and orthogonal (`H_kᵀ H_k = I − 2β v vᵀ +
//! β² v (vᵀv) vᵀ = I` since `β vᵀv = 2`) and satisfies `H_k x = α e₁`, zeroing
//! entries `k+1..m` of column `k` while fixing rows/columns `0..k`. Hence
//! `H_{n-1} ⋯ H_0 A = R` is upper-triangular, and with
//! `Q = H_0 ⋯ H_{n-1}` (a product of orthogonal matrices, therefore orthogonal)
//! we obtain `A = Q R`. Choosing `α = −sign(x₀)‖x‖₂` makes `v₀ = x₀ − α` a sum of
//! like-signed magnitudes (no cancellation), giving a backward-stable reflector. ∎
//!
//! # Corollary (least-squares normal-equation-free solve)
//! For full-column-rank `A` (`m ≥ n`), `x̂ = argminₓ ‖A x − b‖₂` is the unique
//! solution of `R₁ x̂ = (Qᵀ b)[0..n]`, where `R₁` is the top `n × n` block of `R`.
//!
//! *Proof.* `Qᵀ` is orthogonal, so `‖A x − b‖₂ = ‖Qᵀ(A x − b)‖₂ = ‖R x − Qᵀb‖₂`.
//! Partitioning into the first `n` and last `m−n` rows, `R x` only reaches the
//! first `n` (lower rows of `R` are zero), so the last `m−n` rows contribute the
//! fixed residual `‖(Qᵀb)[n..m]‖₂` independent of `x`; the total is minimized by
//! annihilating the first `n` rows, i.e. `R₁ x̂ = (Qᵀb)[0..n]`, solvable by
//! back-substitution since full rank ⇒ `R₁` nonsingular. [`QrDecomposition::solve_least_squares`]
//! realizes this: it forms `Qᵀb` by applying the stored reflectors and
//! back-substitutes — `Q` is never materialized. ∎
//!
//! Evidence tier: theorem/proof sketch above plus value-semantic and differential
//! (vs ndarray/nalgebra) tests for `A = Q R` reconstruction, `Q` orthogonality,
//! and least-squares agreement. Generic over [`crate::RealScalar`], native precision.

use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, ArrayView2, LetoError, Result};

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
/// exactly-zero pivot-column norms are rejected with distinct error reasons.
/// The zero-norm rejection is an exact contract: near rank-deficiency leaves
/// a tiny floating-point residue rather than an exact zero and manifests as
/// ill-conditioning of the solve — detecting it requires column pivoting or
/// an SVD, which this unpivoted factorization deliberately does not do.
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

    /// Materialize the orthogonal factor `Q` (`m × m`).
    #[must_use]
    pub fn q(&self) -> Array2<T> {
        let (m, n) = (self.rows, self.cols);
        let mut q = vec![T::ZERO; m * m];
        for i in 0..m {
            q[i * m + i] = T::ONE;
        }

        let limit = n.min(m);
        for k in (0..limit).rev() {
            let beta = self.betas[k];
            if beta == T::ZERO {
                continue;
            }

            for col_idx in 0..m {
                let mut dot = self.heads[k].mul(q[k * m + col_idx]);
                for offset in 1..(m - k) {
                    let r = k + offset;
                    let v_val = self.packed[r * n + k];
                    dot = dot.add(v_val.mul(q[r * m + col_idx]));
                }

                let bs = beta.mul(dot);

                q[k * m + col_idx] = q[k * m + col_idx].sub(bs.mul(self.heads[k]));
                for offset in 1..(m - k) {
                    let r = k + offset;
                    let v_val = self.packed[r * n + k];
                    q[r * m + col_idx] = q[r * m + col_idx].sub(bs.mul(v_val));
                }
            }
        }

        Array2::from_shape_vec([m, m], q).expect("Q shape matches storage")
    }

    /// Materialize the upper-triangular factor `R` (`m × n`).
    #[must_use]
    pub fn r(&self) -> Array2<T> {
        let (m, n) = (self.rows, self.cols);
        let mut r = vec![T::ZERO; m * n];
        for i in 0..m {
            for j in i..n {
                r[i * n + j] = self.packed[i * n + j];
            }
        }
        Array2::from_shape_vec([m, n], r).expect("R shape matches storage")
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
            for (offset, &value) in y[(k + 1)..m].iter().enumerate() {
                s = s.add(self.packed[(k + 1 + offset) * n + k].mul(value));
            }
            let bs = self.betas[k].mul(s);
            y[k] = y[k].sub(bs.mul(self.heads[k]));
            for (offset, slot) in y[(k + 1)..m].iter_mut().enumerate() {
                let update = bs.mul(self.packed[(k + 1 + offset) * n + k]);
                *slot = slot.sub(update);
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
