//! Compact-WY block Householder reflectors: aggregate `r` reflectors into
//! `Q = I − V T Vᵀ` and apply the block as matrix–matrix products (BLAS-3).
//!
//! A single Householder reflector of width `w` applied to a trailing matrix is a
//! rank-1 update with `O(1)` arithmetic intensity — bandwidth-bound, no
//! contiguous SIMD span when `w` is small (the eig/SVD residual, ADR 0010). The
//! compact-WY representation fuses `r` consecutive reflectors so their combined
//! action on a trailing block is three GEMMs through `Scalar::tiled_gemm`, raising
//! arithmetic intensity to `O(r)` and routing the work onto the tuned SIMD
//! micro-kernel.
//!
//! # Theorem (compact-WY representation, Schreiber–Van Loan 1989)
//! For reflectors `Hⱼ = I − βⱼ vⱼ vⱼᵀ` (`j = 1…r`) with the `vⱼ` as the columns of
//! `V ∈ ℝ^{m×r}`, there is an upper-triangular `T ∈ ℝ^{r×r}` with
//! `H₁ H₂ … H_r = I − V T Vᵀ`. *Proof:* see [`accumulate::build_t`] — induction on
//! `r`, with `T` built columnwise by `T_{0:j,j} = −βⱼ T_{0:j,0:j}(V_{:,0:j}ᵀ vⱼ)`.
//!
//! # Corollary (BLAS-3 application)
//! `Qᵀ C = C − V (Tᵀ (Vᵀ C))` and `C Q = C − ((C V) T) Vᵀ`. Each parenthesised
//! product is a GEMM; the rank-`r` correction is a third GEMM accumulated in
//! place. Applying `Q`/`Qᵀ` is an orthogonal transform, so it preserves the
//! 2-norm/spectrum/singular values up to the standard blocked backward error
//! `O(r·ε‖C‖)` (the summation reorder vs sequential application is bounded, not
//! bitwise — admissible under the differential-tolerance contracts).
//!
//! Leaf modules: [`accumulate`] builds `T` (SSOT). The block apply depends only on
//! the [`crate::Scalar::tiled_gemm`] backend seam (DIP). Generic over
//! [`crate::RealScalar`]; native precision throughout.

mod accumulate;

use crate::domain::real::RealScalar;

/// Transpose a row-major `rows × cols` matrix into a fresh `cols × rows` buffer.
fn transpose<T: RealScalar>(src: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}

/// Apply `Qᵀ` (where `Q = I − V T Vᵀ` is the panel's `r` reflectors) to the
/// trailing block `c` (`m × ncols`, row-major) in place: `C ← Qᵀ C`.
///
/// `v` is the `m × r` reflector panel (row-major), `beta` the `r` reflector
/// scalars. Three GEMMs: `W = Vᵀ C` (`r × ncols`), `W₂ = Tᵀ W`, then the rank-`r`
/// correction `C −= V W₂` (accumulated in place via the negated `W₂`). The
/// `tiled_gemm` accumulate semantics (`C += A·B`) make the final step a single
/// fused call.
pub(super) fn apply_block_left<T: RealScalar>(
    v: &[T],
    beta: &[T],
    c: &mut [T],
    m: usize,
    ncols: usize,
    r: usize,
) {
    if r == 0 || ncols == 0 {
        return;
    }
    let t = accumulate::build_t(v, beta, m, r);

    // W = Vᵀ C  (r × ncols). tiled_gemm needs A row-major (r × m) = Vᵀ.
    let vt = transpose(v, m, r); // r × m
    let mut w = vec![T::ZERO; r * ncols];
    T::tiled_gemm(&vt, c, &mut w, r, ncols, m);

    // W₂ = Tᵀ W  (r × ncols). Tᵀ is r × r.
    let tt = transpose(&t, r, r);
    let mut w2 = vec![T::ZERO; r * ncols];
    T::tiled_gemm(&tt, &w, &mut w2, r, ncols, r);

    // C −= V W₂  ≡  C += V·(−W₂); tiled_gemm accumulates into C.
    for x in w2.iter_mut() {
        *x = T::ZERO.sub(*x);
    }
    T::tiled_gemm(v, &w2, c, m, ncols, r);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `apply_block_left` must equal applying the `r` reflectors sequentially
    /// (`Qᵀ C = H_r … H_1 C`) within the blocked backward-error bound.
    #[test]
    fn block_left_matches_sequential_reflector_application() {
        let (m, r, ncols) = (7usize, 3usize, 4usize);
        // Deterministic reflector panel: lower-trapezoidal columns with a nonzero
        // head on the diagonal; beta_j = 2/(v_jᵀ v_j).
        let mut v = vec![0.0f64; m * r];
        let mut beta = vec![0.0f64; r];
        for j in 0..r {
            let mut nrm = 0.0;
            for row in j..m {
                let val =
                    ((row * 3 + j * 5 + 1) % 7) as f64 - 3.0 + if row == j { 4.0 } else { 0.0 };
                v[row * r + j] = val;
                nrm += val * val;
            }
            beta[j] = 2.0 / nrm;
        }
        let c0: Vec<f64> = (0..m * ncols)
            .map(|i| ((i * 11 + 3) % 13) as f64 - 6.0)
            .collect();

        // Block path.
        let mut c_block = c0.clone();
        apply_block_left(&v, &beta, &mut c_block, m, ncols, r);

        // Sequential reference: C ← H_j C for j = 0..r (this realises Qᵀ C).
        let mut c_seq = c0.clone();
        for j in 0..r {
            // s = v_jᵀ C  (length ncols)
            let mut s = vec![0.0f64; ncols];
            for row in 0..m {
                let vrj = v[row * r + j];
                if vrj == 0.0 {
                    continue;
                }
                for (col, sc) in s.iter_mut().enumerate() {
                    *sc += vrj * c_seq[row * ncols + col];
                }
            }
            for sc in &mut s {
                *sc *= beta[j];
            }
            for row in 0..m {
                let vrj = v[row * r + j];
                if vrj == 0.0 {
                    continue;
                }
                for (col, &sc) in s.iter().enumerate() {
                    c_seq[row * ncols + col] -= vrj * sc;
                }
            }
        }

        let norm_c: f64 = c0.iter().map(|x| x * x).sum::<f64>().sqrt();
        let tol = 16.0 * f64::EPSILON * norm_c * r as f64;
        for (a, b) in c_block.iter().zip(c_seq.iter()) {
            assert!(
                (a - b).abs() <= tol,
                "block {a} vs sequential {b} exceeds blocked-WY tol {tol:e}"
            );
        }
    }

    /// `Q = I − V T Vᵀ` is orthogonal, so `Qᵀ` preserves column 2-norms.
    #[test]
    fn block_left_preserves_column_norms() {
        let (m, r, ncols) = (6usize, 2usize, 3usize);
        let mut v = vec![0.0f64; m * r];
        let mut beta = vec![0.0f64; r];
        for j in 0..r {
            let mut nrm = 0.0;
            for row in j..m {
                let val = ((row + 2 * j) % 5) as f64 - 2.0 + if row == j { 3.0 } else { 0.0 };
                v[row * r + j] = val;
                nrm += val * val;
            }
            beta[j] = 2.0 / nrm;
        }
        let c0: Vec<f64> = (0..m * ncols).map(|i| (i % 7) as f64 - 3.0).collect();
        let mut c = c0.clone();
        apply_block_left(&v, &beta, &mut c, m, ncols, r);

        for col in 0..ncols {
            let n0: f64 = (0..m).map(|r0| c0[r0 * ncols + col].powi(2)).sum();
            let n1: f64 = (0..m).map(|r0| c[r0 * ncols + col].powi(2)).sum();
            assert!(
                (n0 - n1).abs() <= 1e-10 * (1.0 + n0),
                "column {col} norm not preserved: {n0} vs {n1}"
            );
        }
    }
}
