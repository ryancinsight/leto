//! Column-major working-buffer Golub–Kahan bidiagonal reduction, values-only
//! (`m ≥ n`). Local to the SVD values path — leto's global array layout stays
//! row-major; this transposes the input once into a column-major scratch so the
//! **left (column) reflector** lives in contiguous memory, mirroring nalgebra's
//! column-major locality without regressing any other operation.
//!
//! # Theorem (singular values preserved)
//! Storing `Aᵀ`'s columns contiguously and applying the identical left/right
//! Householder reflectors yields the same bidiagonal `B` (hence the same singular
//! values) as the row-major reduction — only the memory layout of the working
//! buffer differs; the arithmetic and reflector sequence are identical. ∎
//!
//! Both applies are contiguous in this layout. The **left** reflector treats the
//! trailing column block as a row-major sub-matrix (rows = columns, length
//! `vlen`, row stride `m`) and applies it in two batched, register-blocked SIMD
//! calls — `gemv_strided` for the dots `dotⱼ = vᵀ·colⱼ` (TILE_M accumulators in
//! flight, ILP) and `axpy_rows` for the rank-1 update `colⱼ −= (τ·dotⱼ)·v` —
//! rather than `O(n)` per-column `dot`/`axpy` calls. The **right** reflector forms
//! `Aw = Σⱼ wⱼ·colⱼ` and updates `colⱼ -= τ·wⱼ·Aw`, column-contiguous. Only the
//! small `O(m)`/`O(n)` reflector gathers touch strided memory.

use crate::domain::real::RealScalar;

/// In-place Householder (`dlarfg`): `x[0] ← β`, `x[1..] ← v` (implicit `v[0]=1`);
/// returns `(τ, β)`.
fn larfg<T: RealScalar>(x: &mut [T]) -> (T, T) {
    let n = x.len();
    if n == 0 {
        return (T::ZERO, T::ZERO);
    }
    let alpha = x[0];
    if n == 1 {
        return (T::ZERO, alpha);
    }
    let mut xnorm_sq = T::ZERO;
    for &xi in &x[1..] {
        xnorm_sq = xnorm_sq.add(xi.mul(xi));
    }
    if xnorm_sq == T::ZERO {
        return (T::ZERO, alpha);
    }
    let beta = {
        let b = alpha.mul(alpha).add(xnorm_sq).sqrt();
        if alpha <= T::ZERO {
            b
        } else {
            T::ZERO.sub(b)
        }
    };
    let tau = beta.sub(alpha).div(beta);
    let scal = T::ONE.div(alpha.sub(beta));
    for xi in &mut x[1..] {
        *xi = xi.mul(scal);
    }
    x[0] = beta;
    (tau, beta)
}

/// Reduce row-major `a` (`m×n`, `m ≥ n`) to the bidiagonal `(d, e)` via a
/// column-major working buffer.
pub(super) fn reduce_values<T: RealScalar>(a: &[T], m: usize, n: usize) -> (Vec<T>, Vec<T>) {
    // Transpose into column-major: cm[j*m + i] = A[i][j]; column j is contiguous.
    let mut cm = vec![T::ZERO; m * n];
    for i in 0..m {
        let row = &a[i * n..i * n + n];
        for (j, &val) in row.iter().enumerate() {
            cm[j * m + i] = val;
        }
    }

    let mut d = vec![T::ZERO; n];
    let mut e = vec![T::ZERO; n];
    let mut vbuf = vec![T::ZERO; m]; // contiguous copy of the active left reflector
    let mut w = vec![T::ZERO; n]; // right reflector
    let mut aw = vec![T::ZERO; m]; // A·w column vector
    let mut dotbuf = vec![T::ZERO; n]; // batched left-reflector dots vᵀ·colⱼ

    for k in 0..n {
        // ---- Left reflector on column k, rows k..m (contiguous).
        let vlen = m - k;
        let (tau_q, beta_d) = {
            let col = &mut cm[k * m + k..k * m + m];
            larfg(col)
        };
        d[k] = beta_d;
        // v with implicit unit head: copy into vbuf[..vlen], vbuf[0] = 1.
        vbuf[0] = T::ONE;
        for r in 1..vlen {
            vbuf[r] = cm[k * m + k + r];
        }
        // Apply to trailing columns k+1..n: colⱼ −= (τ·vᵀcolⱼ)·v, batched as a
        // sub-matrix dot (gemv_strided) + rank-1 update (axpy_rows). Both keep
        // TILE_M independent SIMD accumulators in flight, where the prior
        // per-column loop serialized on each column's reduction/update — the
        // isolated dot batch measured ≈2× (hermes `reflector_dots` bench).
        if tau_q != T::ZERO {
            let ntrail = n - k - 1;
            if ntrail > 0 {
                // Batch the dots: dotbuf[t] = vᵀ·col_{k+1+t} over the trailing
                // column block (gemv_strided accumulates, so zero first).
                for slot in dotbuf[..ntrail].iter_mut() {
                    *slot = T::ZERO;
                }
                T::gemv_strided(
                    &cm[(k + 1) * m + k..],
                    &vbuf[..vlen],
                    &mut dotbuf[..ntrail],
                    ntrail,
                    vlen,
                    m,
                );
                // Scale in place to the per-column axpy coefficient, then batch
                // the rank-1 update via axpy_rows: colⱼ[i] += (−τ·dot[j])·v[i].
                for slot in dotbuf[..ntrail].iter_mut() {
                    *slot = T::ZERO.sub(tau_q.mul(*slot));
                }
                T::axpy_rows(
                    &dotbuf[..ntrail],
                    &vbuf[..vlen],
                    &mut cm[(k + 1) * m + k..],
                    m,
                    ntrail,
                    vlen,
                );
            }
        }

        // ---- Right reflector on row k, columns k+1..n (strided gather).
        if k + 1 < n {
            let rlen = n - k - 1;
            for (idx, j) in ((k + 1)..n).enumerate() {
                w[idx] = cm[j * m + k];
            }
            let (tau_p, beta_e) = larfg(&mut w[..rlen]);
            e[k] = beta_e;
            w[0] = T::ONE; // implicit unit head
            if tau_p != T::ZERO {
                let rows = m - k - 1; // trailing rows k+1..m
                if rows > 0 {
                    // Aw[k+1..m] = Σⱼ wⱼ · colⱼ[k+1..m]  (column-contiguous accumulate).
                    for v in aw[..rows].iter_mut() {
                        *v = T::ZERO;
                    }
                    for (idx, j) in ((k + 1)..n).enumerate() {
                        let seg = &cm[j * m + (k + 1)..j * m + m];
                        T::axpy_slice(w[idx], seg, &mut aw[..rows]);
                    }
                    // colⱼ[k+1..m] −= τ·wⱼ·Aw  (column-contiguous update).
                    for (idx, j) in ((k + 1)..n).enumerate() {
                        let scale = T::ZERO.sub(tau_p.mul(w[idx]));
                        let seg = &mut cm[j * m + (k + 1)..j * m + m];
                        T::axpy_slice(scale, &aw[..rows], seg);
                    }
                }
            }
        }
    }

    (d, e)
}
