//! Compact-WY block application to a trailing matrix.

use crate::domain::real::RealScalar;

use super::accumulate::build_t;
use super::transpose::transpose_into;

/// Apply `Qᵀ` (where `Q = I − V T Vᵀ` is the panel's `r` reflectors) to the
/// trailing block `c` (`m × ncols`, row-major) in place: `C ← Qᵀ C`.
///
/// `v` is the `m × r` reflector panel (row-major), `beta` the `r` reflector
/// scalars. Three GEMMs: `W = Vᵀ C` (`r × ncols`), `W₂ = Tᵀ W`, then the rank-`r`
/// correction `C −= V W₂` (accumulated in place via the negated `W₂`). The
/// `tiled_gemm` accumulate semantics (`C += A·B`) make the final step a single
/// fused call.
pub(in crate::application::linalg) fn apply_block_left<T: RealScalar>(
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

    let t_len = r * r;
    let mut t_stack = [T::ZERO; 1024];
    let mut t_vec = Vec::new();
    let t = if t_len <= 1024 {
        &mut t_stack[..t_len]
    } else {
        t_vec.resize(t_len, T::ZERO);
        &mut t_vec[..]
    };
    build_t(v, beta, t, m, r);

    // W = Vᵀ C  (r × ncols). tiled_gemm needs A row-major (r × m) = Vᵀ.
    let vt_len = r * m;
    let mut vt_stack = [T::ZERO; 2048];
    let mut vt_vec = Vec::new();
    let vt = if vt_len <= 2048 {
        &mut vt_stack[..vt_len]
    } else {
        vt_vec.resize(vt_len, T::ZERO);
        &mut vt_vec[..]
    };
    transpose_into(v, vt, m, r);

    let w_len = r * ncols;
    let mut w_stack = [T::ZERO; 2048];
    let mut w_vec = Vec::new();
    let w = if w_len <= 2048 {
        &mut w_stack[..w_len]
    } else {
        w_vec.resize(w_len, T::ZERO);
        &mut w_vec[..]
    };
    w.fill(T::ZERO);

    T::tiled_gemm(vt, c, w, r, ncols, m);

    // W₂ = Tᵀ W  (r × ncols). Tᵀ is r × r.
    let tt_len = r * r;
    let mut tt_stack = [T::ZERO; 1024];
    let mut tt_vec = Vec::new();
    let tt = if tt_len <= 1024 {
        &mut tt_stack[..tt_len]
    } else {
        tt_vec.resize(tt_len, T::ZERO);
        &mut tt_vec[..]
    };
    transpose_into(t, tt, r, r);

    let mut w2_stack = [T::ZERO; 2048];
    let mut w2_vec = Vec::new();
    let w2 = if w_len <= 2048 {
        &mut w2_stack[..w_len]
    } else {
        w2_vec.resize(w_len, T::ZERO);
        &mut w2_vec[..]
    };
    w2.fill(T::ZERO);

    T::tiled_gemm(tt, w, w2, r, ncols, r);

    // C −= V W₂  ≡  C += V·(−W₂); tiled_gemm accumulates into C.
    for x in w2.iter_mut() {
        *x = T::ZERO.sub(*x);
    }
    T::tiled_gemm(v, w2, c, m, ncols, r);
}

/// Apply `Q` (where `Q = I − V T Vᵀ` is the panel's `r` reflectors) to the
/// trailing block `c` (`m × ncols`, row-major) on the right in place:
/// `C ← C·Q`.
///
/// This is the row-major transpose of `apply_block_left`:
/// `Q` is the same compact-WY form as above, and
/// `C·Q = C − (C V) T Vᵀ`.
/// The same temporary re-use and scratch layout apply.
#[cfg(test)]
pub(in crate::application::linalg) fn apply_block_right<T: RealScalar>(
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
    debug_assert_eq!(m, ncols, "block-right WY form expects C to be square");
    debug_assert!(c.len() >= m * ncols);

    let t_len = r * r;
    let mut t_stack = [T::ZERO; 1024];
    let mut t_vec = Vec::new();
    let t = if t_len <= 1024 {
        &mut t_stack[..t_len]
    } else {
        t_vec.resize(t_len, T::ZERO);
        &mut t_vec[..]
    };
    build_t(v, beta, t, m, r);

    let w_len = m * r;
    let mut w_stack = [T::ZERO; 2048];
    let mut w_vec = Vec::new();
    let w = if w_len <= 2048 {
        &mut w_stack[..w_len]
    } else {
        w_vec.resize(w_len, T::ZERO);
        &mut w_vec[..]
    };
    w.fill(T::ZERO);
    T::tiled_gemm(c, v, w, m, r, ncols);

    let mut w2_stack = [T::ZERO; 2048];
    let mut w2_vec = Vec::new();
    let w2 = if w_len <= 2048 {
        &mut w2_stack[..w_len]
    } else {
        w2_vec.resize(w_len, T::ZERO);
        &mut w2_vec[..]
    };
    w2.fill(T::ZERO);
    T::tiled_gemm(w, t, w2, m, r, r);

    let vt_len = r * m;
    let mut vt_stack = [T::ZERO; 1024];
    let mut vt_vec = Vec::new();
    let vt = if vt_len <= 1024 {
        &mut vt_stack[..vt_len]
    } else {
        vt_vec.resize(vt_len, T::ZERO);
        &mut vt_vec[..]
    };
    transpose_into(v, vt, m, r);

    for x in w2.iter_mut() {
        *x = T::ZERO.sub(*x);
    }
    T::tiled_gemm(w2, vt, c, m, ncols, r);
}

#[cfg(test)]
mod tests {
    use super::{apply_block_left, apply_block_right};
    use eunomia::FloatElement;

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
                let val = f64::from_count((row * 3 + j * 5 + 1) % 7) - 3.0
                    + if row == j { 4.0 } else { 0.0 };
                v[row * r + j] = val;
                nrm += val * val;
            }
            beta[j] = 2.0 / nrm;
        }
        let c0: Vec<f64> = (0..m * ncols)
            .map(|i| f64::from_count((i * 11 + 3) % 13) - 6.0)
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
        let tol = 16.0 * f64::EPSILON * norm_c * f64::from_count(r);
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
                let val =
                    f64::from_count((row + 2 * j) % 5) - 2.0 + if row == j { 3.0 } else { 0.0 };
                v[row * r + j] = val;
                nrm += val * val;
            }
            beta[j] = 2.0 / nrm;
        }
        let c0: Vec<f64> = (0..m * ncols)
            .map(|i| f64::from_count(i % 7) - 3.0)
            .collect();
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

    /// `apply_block_right` must equal applying the `r` reflectors sequentially
    /// (`C ← C·H_0·H_1·…·H_{r-1}`) within the blocked backward-error bound.
    #[test]
    fn block_right_matches_sequential_reflector_application() {
        let (m, r, ncols) = (7usize, 3usize, 7usize);
        let mut v = vec![0.0f64; m * r];
        let mut beta = vec![0.0f64; r];
        for j in 0..r {
            let mut nrm = 0.0;
            for row in j..m {
                let val = f64::from_count((row * 5 + j * 3 + 1) % 9) - 4.0
                    + if row == j { 3.0 } else { 0.0 };
                v[row * r + j] = val;
                nrm += val * val;
            }
            beta[j] = 2.0 / nrm;
        }
        let c0: Vec<f64> = (0..m * ncols)
            .map(|i| f64::from_count((i * 7 + 5) % 17) - 8.0)
            .collect();

        let mut c_block = c0.clone();
        apply_block_right(&v, &beta, &mut c_block, m, ncols, r);

        let mut c_seq = c0.clone();
        for j in 0..r {
            for row in 0..m {
                let row_start = row * ncols;
                let row = &mut c_seq[row_start..row_start + ncols];
                let mut dot = 0.0f64;
                for i in 0..m {
                    let vr = v[i * r + j];
                    if vr == 0.0 {
                        continue;
                    }
                    dot += row[i] * vr;
                }
                let scale = beta[j] * dot;
                for i in 0..m {
                    let vr = v[i * r + j];
                    if vr == 0.0 {
                        continue;
                    }
                    row[i] -= vr * scale;
                }
            }
        }

        for (a, b) in c_block.iter().zip(c_seq.iter()) {
            let norm_c: f64 = c0.iter().map(|x| x * x).sum::<f64>().sqrt();
            let tol = 16.0 * f64::EPSILON * norm_c * f64::from_count(r);
            assert!(
                (a - b).abs() <= tol,
                "block {a} vs sequential {b} exceeds blocked-WY tol {tol:e}"
            );
        }
    }

    /// `apply_block_right` preserves matrix Frobenius norm, as `Q` is orthogonal.
    #[test]
    fn block_right_preserves_frobenius_norm() {
        let (m, r, ncols) = (6usize, 3usize, 6usize);
        let mut v = vec![0.0f64; m * r];
        let mut beta = vec![0.0f64; r];
        for j in 0..r {
            let mut nrm = 0.0;
            for row in j..m {
                let val =
                    f64::from_count((row + 2 * j) % 7) - 3.0 + if row == j { 4.0 } else { 0.0 };
                v[row * r + j] = val;
                nrm += val * val;
            }
            beta[j] = 2.0 / nrm;
        }
        let c0: Vec<f64> = (0..m * ncols)
            .map(|i| f64::from_count(i % 11) - 5.0)
            .collect();
        let mut c = c0.clone();
        apply_block_right(&v, &beta, &mut c, m, ncols, r);

        let n0: f64 = c0.iter().map(|x| x * x).sum();
        let n1: f64 = c.iter().map(|x| x * x).sum();
        assert!(
            (n0 - n1).abs() <= 1e-10 * (1.0 + n0),
            "Frobenius norm drift: {n0} vs {n1}"
        );
    }
}
