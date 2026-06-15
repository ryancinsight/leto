//! Shared Householder reflector primitive (SSOT for two-sided orthogonal
//! transforms).
//!
//! A Householder reflector `P = I − β v vᵀ` (`β = 2/(vᵀv)`) is symmetric and
//! orthogonal. Given a vector `x`, the reflector built here maps `x` onto
//! `α·e₁` (`α = −sign(x₀)·‖x‖`, the cancellation-avoiding sign), zeroing every
//! component but the first. Both the Hessenberg reduction and the
//! bidiagonalization express their work as left/right applications of this one
//! primitive — there is no per-algorithm reflector code (QR's packed-reflector
//! scheme is the one remaining specialized variant, tracked in `gap_audit.md`).

use crate::domain::real::RealScalar;

/// A Householder reflector `P = I − β v vᵀ`, with `v` positioned relative to a
/// caller-supplied base index.
pub(crate) struct Reflector<T> {
    /// Reflector vector.
    pub(crate) v: Vec<T>,
    /// `β = 2/(vᵀv)`.
    pub(crate) beta: T,
}

/// Build the reflector mapping `x` to `α·e₁`; returns `(reflector, α)`.
///
/// `None` when `x` is already negligible (no reflection needed). The sign of
/// `α = −sign(x₀)·‖x‖` is chosen so `v₀ = x₀ − α` adds in magnitude rather than
/// cancels — the numerically stable convention.
pub(crate) fn reflector<T: RealScalar>(x: &[T]) -> Option<(Reflector<T>, T)> {
    if x.is_empty() {
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

    let mut v = x.to_vec();
    v[0] = v[0].sub(alpha); // v₀ = x₀ − α

    let mut vnorm_sq = T::ZERO;
    for &vi in &v {
        vnorm_sq = vnorm_sq.add(vi.mul(vi));
    }
    if vnorm_sq <= T::ZERO {
        return None;
    }
    let beta = T::ONE.add(T::ONE).div(vnorm_sq);
    Some((Reflector { v, beta }, alpha))
}

/// Left-apply `P` to rows `[base_row .. base_row + v.len())` of a row-major
/// `_ × cols` matrix, across columns `[c0 .. c1)` (`block ← P · block`).
pub(crate) fn apply_left<T: RealScalar>(
    refl: &Reflector<T>,
    m: &mut [T],
    cols: usize,
    base_row: usize,
    c0: usize,
    c1: usize,
) {
    let Reflector { v, beta } = refl;
    for j in c0..c1 {
        let mut dot = T::ZERO;
        for (i, &vi) in v.iter().enumerate() {
            dot = dot.add(vi.mul(m[(base_row + i) * cols + j]));
        }
        let scale = beta.mul(dot);
        for (i, &vi) in v.iter().enumerate() {
            let idx = (base_row + i) * cols + j;
            m[idx] = m[idx].sub(scale.mul(vi));
        }
    }
}

/// Right-apply `P` to columns `[base_col .. base_col + v.len())` of a row-major
/// `_ × cols` matrix, across rows `[r0 .. r1)` (`block ← block · P`). Because
/// `P = Pᵀ`, the same reflector serves left and right applications.
pub(crate) fn apply_right<T: RealScalar>(
    refl: &Reflector<T>,
    m: &mut [T],
    cols: usize,
    base_col: usize,
    r0: usize,
    r1: usize,
) {
    let Reflector { v, beta } = refl;
    for i in r0..r1 {
        let mut dot = T::ZERO;
        for (j, &vj) in v.iter().enumerate() {
            dot = dot.add(m[i * cols + base_col + j].mul(vj));
        }
        let scale = beta.mul(dot);
        for (j, &vj) in v.iter().enumerate() {
            let idx = i * cols + base_col + j;
            m[idx] = m[idx].sub(scale.mul(vj));
        }
    }
}
