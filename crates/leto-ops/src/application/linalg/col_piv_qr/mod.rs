//! QR with column pivoting: `A P = Q R` (rank-revealing).
//!
//! Plain Householder QR ([`crate::qr_decompose`]) factors `A = Q R`; column pivoting adds
//! a permutation `P` that, at each step, moves the column of largest remaining
//! norm to the front. This forces `|R₀₀| ≥ |R₁₁| ≥ … ≥ |R_{r-1,r-1}| > 0` with
//! the trailing diagonal ≈ 0, so the factorization is **rank-revealing**.
//!
//! # Theorem (column-pivoted QR)
//! For every `A ∈ ℝᵐˣⁿ` there is a permutation `P`, an orthogonal `Q`, and an
//! upper-triangular `R` with `A P = Q R` and the diagonal of `R` non-increasing
//! in magnitude. *Proof (constructive):* at step `k`, choosing the
//! largest-tail-norm column as pivot guarantees `|Rₖₖ| ≥ ‖(R P)[k.., j]‖` for
//! all `j > k`; the Householder reflector that follows sets `Rₖₖ` to that pivot
//! column's tail norm, which therefore dominates every later diagonal. ∎
//!
//! # Corollary (rank)
//! The number of above-threshold diagonal entries is `rank(A)`; the monotone
//! diagonal means the first negligible one reveals the rank — more reliably than
//! a Gram-spectrum count for borderline cases.
//!
//! Leaf modules: `decompose` (the pivoted reduction, on the shared
//! `householder` primitive) and the solve logic here.
//! Generic over [`crate::RealScalar`], native precision.

mod decompose;

use crate::domain::real::RealScalar;
use leto::{Array1, Array2, ArrayView1, ArrayView2, LetoError, Result};

/// Column-pivoted QR decomposition `A P = Q R`.
#[derive(Debug, Clone)]
pub struct ColPivQrDecomposition<T> {
    q: Vec<T>,
    r: Vec<T>,
    perm: Vec<usize>,
    rank: usize,
    m: usize,
    n: usize,
}

impl<T: RealScalar> ColPivQrDecomposition<T> {
    /// Numerical rank (count of above-threshold `R` diagonal entries).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Orthogonal factor `Q` (`m × m`).
    #[must_use]
    pub fn q(&self) -> Array2<T> {
        Array2::from_shape_vec([self.m, self.m], self.q.clone()).expect("Q shape matches storage")
    }

    /// Upper-triangular factor `R` (`m × n`).
    #[must_use]
    pub fn r(&self) -> Array2<T> {
        Array2::from_shape_vec([self.m, self.n], self.r.clone()).expect("R shape matches storage")
    }

    /// Column permutation: `permutation()[k]` is the original column at position `k`.
    #[must_use]
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }

    /// Least-squares solution of `min ‖A x − b‖₂` for full-column-rank `A`.
    ///
    /// `A = Q R Pᵀ`, so `min‖Ax−b‖ = min‖R(Pᵀx) − Qᵀb‖`; with full column rank
    /// the top `n×n` of `R` is invertible — back-substitute `R y = (Qᵀb)[0..n]`,
    /// then `x = P y`.
    ///
    /// # Errors
    /// [`LetoError`] on a rank-deficient matrix (rank-deficient
    /// least squares is a follow-up) or shape mismatch.
    pub fn solve_least_squares(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        let (m, n) = (self.m, self.n);
        if rhs.shape() != [m] {
            return Err(LetoError::ShapeMismatch {
                lhs: rhs.shape().to_vec(),
                rhs: vec![m],
            });
        }
        if self.rank < n {
            return Err(LetoError::StorageError {
                reason: "ColPivQR least squares requires full column rank".to_string(),
            });
        }

        let mut qtb_stack = [T::ZERO; 128];
        let mut qtb_vec = Vec::new();
        let qtb = if m <= 128 {
            &mut qtb_stack[..m]
        } else {
            qtb_vec.resize(m, T::ZERO);
            &mut qtb_vec[..]
        };

        let mut rhs_stack = [T::ZERO; 128];
        let mut rhs_vec = Vec::new();
        let rhs_slice = if let Some(slice) = rhs.as_slice() {
            slice
        } else {
            if m <= 128 {
                for (k, slot) in rhs_stack[..m].iter_mut().enumerate() {
                    *slot = *rhs.get([k])?;
                }
                &rhs_stack[..m]
            } else {
                rhs_vec.reserve_exact(m);
                for k in 0..m {
                    rhs_vec.push(*rhs.get([k])?);
                }
                &rhs_vec[..]
            }
        };

        // qtb = Qᵀ b (Qᵀ[i][k] = q[k][i]).
        for (i, slot) in qtb.iter_mut().enumerate() {
            let mut acc = T::ZERO;
            for (k, &rhs_val) in rhs_slice.iter().enumerate() {
                acc = acc.add(rhs_val.mul(self.q[k * m + i]));
            }
            *slot = acc;
        }

        // Back-substitute R y = qtb[0..n] in place in qtb.
        for i in (0..n).rev() {
            let mut s = qtb[i];
            for (j, &qtb_j) in qtb.iter().enumerate().take(n).skip(i + 1) {
                s = s.sub(self.r[i * n + j].mul(qtb_j));
            }
            qtb[i] = s.div(self.r[i * n + i]);
        }

        // x = P y.
        let mut x = vec![T::ZERO; n];
        for k in 0..n {
            x[self.perm[k]] = qtb[k];
        }
        Array1::from_shape_vec([n], x)
    }
}

/// Factor a matrix with column-pivoted (rank-revealing) Householder QR.
///
/// # Errors
/// [`LetoError::StorageError`] for a non-finite entry.
pub fn col_piv_qr<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<ColPivQrDecomposition<T>> {
    let f = decompose::factor(matrix)?;
    Ok(ColPivQrDecomposition {
        q: f.q,
        r: f.r,
        perm: f.perm,
        rank: f.rank,
        m: f.m,
        n: f.n,
    })
}
