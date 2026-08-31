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
//! back-substitution since full rank ⇒ `R₁` nonsingular.
//! [`QrDecomposition::solve_least_squares`] realizes this: it forms `Qᵀb` by
//! applying the stored reflectors and
//! back-substitutes — `Q` is never materialized. ∎
//!
//! Vertical structure (SoC, mirroring `bunch_kaufman/{decompose, solve}`): the
//! `decompose` leaf owns the factorization kernel ([`qr_decompose`]); the `solve`
//! leaf owns the least-squares solve ([`solve_least_squares`]). This module owns
//! the [`QrDecomposition`] type and its `Q`/`R` materialization accessors. Generic
//! over [`crate::RealScalar`], native precision.

pub mod decompose;
pub mod solve;

pub use decompose::qr_decompose;
pub use solve::solve_least_squares;

use crate::domain::real::RealScalar;
use leto::Array2;

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
    pub(super) packed: Vec<T>,
    /// Householder vector head components `v_k[k]` (diagonal slots hold R).
    pub(super) heads: Vec<T>,
    /// `β_k = 2 / (v_kᵀ v_k)` per reflector.
    pub(super) betas: Vec<T>,
    pub(super) rows: usize,
    pub(super) cols: usize,
}

impl<T: RealScalar> QrDecomposition<T> {
    /// Construct a QR decomposition directly from its raw components.
    #[must_use]
    #[inline]
    pub fn from_raw_parts(
        packed: Vec<T>,
        heads: Vec<T>,
        betas: Vec<T>,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self {
            packed,
            heads,
            betas,
            rows,
            cols,
        }
    }

    /// `(rows, cols)` of the factored matrix.
    #[must_use]
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The row-major `m × n` packed factors: element `(i, j)` is at `i * n + j`.
    ///
    /// `R` occupies the upper triangle (`j ≥ i`). Reflector `k`'s vector spans
    /// rows `k..m` of column `k`, but its **head** `v_k[k]` is *not* here: the
    /// diagonal slot `k * n + k` holds `R[k][k]` instead, so the head comes from
    /// [`Self::heads`] and only the tail entries `packed[r * n + k]`, `r > k`,
    /// come from this slice. [`Self::q`] is the canonical application of the
    /// three slices together.
    #[must_use]
    #[inline]
    pub fn packed(&self) -> &[T] {
        &self.packed
    }

    /// Householder vector head components `v_k[k]`, indexed by reflector `k`
    /// (`k < min(m, n)`); the diagonal slots of [`Self::packed`] hold `R`
    /// instead. Applied by [`Self::q`].
    #[must_use]
    #[inline]
    pub fn heads(&self) -> &[T] {
        &self.heads
    }

    /// Reflector coefficients `β_k = 2 / (v_kᵀ v_k)`, indexed by reflector `k`
    /// (`k < min(m, n)`), for `H_k = I − β_k v_k v_kᵀ`. Applied by [`Self::q`].
    #[must_use]
    #[inline]
    pub fn betas(&self) -> &[T] {
        &self.betas
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the documented indexing of `packed`/`heads`/`betas`: a consumer that
    /// applies the reflectors itself (the hephaestus device-side `Q` driver) must
    /// reproduce `q()` exactly. Reconstruction is written out here rather than
    /// delegated, so a storage-layout change breaks this test instead of silently
    /// breaking consumers.
    #[test]
    fn reflector_accessors_reproduce_q() {
        let a = Array2::from_vec(
            [4, 3],
            vec![
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 10.0, //
                11.0, 13.0, 17.0,
            ],
        )
        .expect("4x3 literal matches the declared shape");
        let qr = qr_decompose(&a.view()).expect("full-rank 4x3 matrix factors");

        let (m, n) = qr.shape();
        let packed = qr.packed();
        let heads = qr.heads();
        let betas = qr.betas();

        // Q = H_0 ⋯ H_{limit-1} applied to the identity, accumulated right to
        // left: start from I and apply reflector k for k = limit-1 down to 0.
        let mut q = vec![0.0_f64; m * m];
        for i in 0..m {
            q[i * m + i] = 1.0;
        }

        let limit = n.min(m);
        for k in (0..limit).rev() {
            let beta = betas[k];
            if beta == 0.0 {
                continue;
            }
            for col in 0..m {
                // vᵀ q[:, col]: head from `heads[k]`, tail from column k of the
                // packed factors below the diagonal.
                let mut dot = heads[k] * q[k * m + col];
                for r in (k + 1)..m {
                    dot += packed[r * n + k] * q[r * m + col];
                }
                let scaled = beta * dot;
                q[k * m + col] -= scaled * heads[k];
                for r in (k + 1)..m {
                    q[r * m + col] -= scaled * packed[r * n + k];
                }
            }
        }

        // Bitwise equality, no tolerance: this reconstruction performs the same
        // IEEE-754 operations on the same values in the same order as `q()`, so
        // every intermediate rounds identically. Any difference means the
        // accessors do not expose what `q()` consumes — the defect this pins.
        let expected = qr.q();
        assert_eq!(
            q.as_slice(),
            expected
                .as_slice()
                .expect("q() returns dense row-major storage")
        );
    }
}
