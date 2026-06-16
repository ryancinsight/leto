//! Francis double-shift implicit QR iteration on a real upper-Hessenberg matrix,
//! accumulating the orthogonal similarity.
//!
//! Operates entirely in real arithmetic: complex eigenvalues surface as isolated
//! 2×2 diagonal blocks (standardized later), so no complex type is needed here.

use crate::domain::real::RealScalar;
use leto::{LetoError, Result};

/// Iteration cap before declaring non-convergence (Wilkinson + exceptional
/// shifts converge in `O(n)` steps; this is a safety bound).
const MAX_ITER: usize = 2000;

struct StackReflector<T> {
    v: [T; 3],
    len: usize,
    beta: T,
}

fn stack_reflector<T: RealScalar>(x: &[T]) -> Option<(StackReflector<T>, T)> {
    let len = x.len();
    if len == 0 || len > 3 {
        return None;
    }
    let mut norm_sq = T::ZERO;
    for &xi in x {
        norm_sq = norm_sq.add(xi.mul(xi));
    }
    let norm = norm_sq.sqrt();
    if norm <= T::ZERO {
        return None;
    }

    let sign = if x[0] < T::ZERO {
        T::ZERO.sub(T::ONE)
    } else {
        T::ONE
    };
    let alpha = T::ZERO.sub(sign.mul(norm)); // α = −sign·‖x‖

    let mut v = [T::ZERO; 3];
    v[..len].copy_from_slice(&x[..len]);
    v[0] = v[0].sub(alpha); // v₀ = x₀ − α

    let mut vnorm_sq = T::ZERO;
    for &vi in &v[..len] {
        vnorm_sq = vnorm_sq.add(vi.mul(vi));
    }
    if vnorm_sq <= T::ZERO {
        return None;
    }
    let beta = T::ONE.add(T::ONE).div(vnorm_sq);
    Some((StackReflector { v, len, beta }, alpha))
}

#[inline]
fn at<T: Copy>(h: &[T], i: usize, j: usize, n: usize) -> T {
    h[i * n + j]
}

/// Left-apply a Householder reflector `P = I − β v vᵀ` (positioned at base row
/// `k`, `v.len()` rows) across columns `c_lo..=c_hi`: `H ← P H`.
fn apply_left<T: RealScalar>(
    h: &mut [T],
    v: &[T],
    beta: T,
    k: usize,
    n: usize,
    c_lo: usize,
    c_hi: usize,
) {
    for j in c_lo..=c_hi {
        let mut w = T::ZERO;
        for (i, &vi) in v.iter().enumerate() {
            w = w.add(vi.mul(h[(k + i) * n + j]));
        }
        w = w.mul(beta);
        for (i, &vi) in v.iter().enumerate() {
            let cell = (k + i) * n + j;
            h[cell] = h[cell].sub(vi.mul(w));
        }
    }
}

/// Right-apply a Householder reflector (base column `k`) across rows
/// `r_lo..=r_hi`: `H ← H P`.
fn apply_right<T: RealScalar>(
    h: &mut [T],
    v: &[T],
    beta: T,
    k: usize,
    n: usize,
    r_lo: usize,
    r_hi: usize,
) {
    for i in r_lo..=r_hi {
        let mut w = T::ZERO;
        for (c, &vc) in v.iter().enumerate() {
            w = w.add(h[i * n + (k + c)].mul(vc));
        }
        w = w.mul(beta);
        for (c, &vc) in v.iter().enumerate() {
            let cell = i * n + (k + c);
            h[cell] = h[cell].sub(w.mul(vc));
        }
    }
}

/// One Francis double-shift step on the active block `[lo, hi]` (`hi − lo ≥ 2`),
/// updating `h` (the Hessenberg matrix) and `z` (the accumulated similarity).
///
/// The implicit shift forms the first column of `(H − μ₁I)(H − μ₂I)` from the
/// trailing-2×2 shift pair `μ₁, μ₂` (or an exceptional ad-hoc pair when the
/// iteration stalls), then chases the resulting bulge down the band with size-3
/// (and a final size-2) Householder reflectors — a single orthogonal similarity
/// equal to one double-shifted QR step (the implicit-Q theorem).
fn francis_step<T: RealScalar, const ACCUMULATE_Q: bool>(
    h: &mut [T],
    z: &mut [T],
    lo: usize,
    hi: usize,
    n: usize,
    exceptional: bool,
) {
    // Shift sum `s = μ₁ + μ₂` and product `t = μ₁ μ₂`.
    let (s, t) = if exceptional {
        // Ad-hoc Wilkinson shift to break cycles.
        let scale = at(h, hi, hi - 1, n)
            .abs()
            .add(at(h, hi - 1, hi - 2, n).abs());
        let three_halves = T::from_f64(1.5);
        (three_halves.mul(scale), scale.mul(scale))
    } else {
        let a = at(h, hi - 1, hi - 1, n);
        let d = at(h, hi, hi, n);
        let b = at(h, hi - 1, hi, n);
        let c = at(h, hi, hi - 1, n);
        (a.add(d), a.mul(d).sub(b.mul(c)))
    };

    // First column of (H² − sH + tI) restricted to the block top.
    let h00 = at(h, lo, lo, n);
    let h01 = at(h, lo, lo + 1, n);
    let h10 = at(h, lo + 1, lo, n);
    let h11 = at(h, lo + 1, lo + 1, n);
    let h21 = at(h, lo + 2, lo + 1, n);
    let mut x = h00.mul(h00).add(h01.mul(h10)).sub(s.mul(h00)).add(t);
    let mut y = h10.mul(h00.add(h11).sub(s));
    let mut zz = h10.mul(h21);

    for k in lo..=(hi - 1) {
        let len = if k < hi - 1 { 3 } else { 2 };
        let refl_opt = if len == 3 {
            let arr = [x, y, zz];
            stack_reflector(&arr)
        } else {
            let arr = [x, y, T::ZERO];
            stack_reflector(&arr[..2])
        };
        if let Some((refl, _alpha)) = refl_opt {
            let v_slice = &refl.v[..refl.len];
            apply_left(h, v_slice, refl.beta, k, n, lo, n - 1);
            apply_right(h, v_slice, refl.beta, k, n, 0, hi);
            // Accumulate the similarity into `z` only when Schur vectors are
            // wanted; for eigenvalues-only this branch is DCE'd at
            // monomorphization (zero cost), and `z` may be empty.
            if ACCUMULATE_Q {
                apply_right(z, v_slice, refl.beta, k, n, 0, n - 1);
            }
        }
        if k + 1 < hi {
            x = at(h, k + 1, k, n);
            y = at(h, k + 2, k, n);
            zz = if k + 3 <= hi {
                at(h, k + 3, k, n)
            } else {
                T::ZERO
            };
        }
    }
}

/// Drive the Francis iteration to convergence: `h` becomes real
/// quasi-upper-triangular (real Schur form `T`) and `z` accumulates the
/// orthogonal similarity so that `H₀ = z T zᵀ`.
///
/// # Errors
/// [`LetoError::StorageError`] if a block fails to converge within [`MAX_ITER`].
pub(super) fn run<T: RealScalar, const ACCUMULATE_Q: bool>(
    h: &mut [T],
    z: &mut [T],
    n: usize,
) -> Result<()> {
    if n < 3 {
        return Ok(()); // 0/1: trivial; 2: a single block, standardized later.
    }
    let mut hi = n - 1;
    let mut iter = 0usize;
    loop {
        // Bottom-most unreduced block: scan up while the subdiagonal is
        // non-negligible (precision-exact `d + |sub| == d`).
        let mut lo = hi;
        while lo > 0 {
            let sub = at(h, lo, lo - 1, n).abs();
            let d = at(h, lo - 1, lo - 1, n).abs().add(at(h, lo, lo, n).abs());
            if d.add(sub) == d {
                h[lo * n + (lo - 1)] = T::ZERO;
                break;
            }
            lo -= 1;
        }

        if lo == hi {
            // 1×1 deflation (real eigenvalue).
            if hi == 0 {
                break;
            }
            hi -= 1;
            iter = 0;
            continue;
        }
        if lo == hi - 1 {
            // 2×2 deflation (real pair or complex conjugate pair).
            if lo == 0 {
                break;
            }
            hi = lo - 1;
            iter = 0;
            continue;
        }

        iter += 1;
        if iter > MAX_ITER {
            return Err(LetoError::StorageError {
                reason: "Schur QR iteration failed to converge".to_string(),
            });
        }
        francis_step::<T, ACCUMULATE_Q>(h, z, lo, hi, n, iter.is_multiple_of(10));
    }
    Ok(())
}
