//! Householder reflector primitive for two-sided orthogonal similarity.
//!
//! A Householder reflector is `P = I − β v vᵀ` with `β = 2/(vᵀv)`. It is
//! symmetric (`P = Pᵀ`) and orthogonal (`P² = I`), and chosen so that applying
//! it to a target sub-column maps that sub-column onto a single axis, zeroing
//! the rest. The QR factorization ([`super::super::qr`]) uses a packed-reflector
//! scheme specialized to building `R`; this primitive is the standalone form
//! needed for the *two-sided* application `P H P` that preserves similarity in
//! the Hessenberg reduction. (A future consolidation could share one reflector
//! core between the two; the access patterns differ enough that it is kept
//! local for now — recorded in `gap_audit.md`.)

use crate::domain::real::RealScalar;

/// A Householder reflector acting on the trailing index range `offset..n`
/// (`P = I − β v vᵀ`, with `v` indexed from `offset`).
pub(super) struct Reflector<T> {
    /// Reflector vector (length `n − offset`).
    pub(super) v: Vec<T>,
    /// `β = 2/(vᵀv)`.
    pub(super) beta: T,
    /// First affected index.
    pub(super) offset: usize,
}

/// Build the reflector that zeroes `H[offset+1 .. n][col]` (the entries strictly
/// below the subdiagonal in column `col`), where `offset = col + 1`.
///
/// Returns `None` when the sub-column is already negligible (nothing to zero).
/// The sign is chosen as `α = −sign(x₀)·‖x‖` so that `v₀ = x₀ − α` adds rather
/// than cancels, which is the numerically stable choice.
pub(super) fn reflector_for_column<T: RealScalar>(
    h: &[T],
    n: usize,
    col: usize,
) -> Option<Reflector<T>> {
    let offset = col + 1;
    let len = n - offset;

    let mut norm_sq = T::ZERO;
    let mut v = Vec::with_capacity(len);
    for i in 0..len {
        let x = h[(offset + i) * n + col];
        norm_sq = norm_sq.add(x.mul(x));
        v.push(x);
    }
    let norm = norm_sq.sqrt();
    if norm <= T::ZERO {
        return None;
    }

    let sign = if v[0] < T::ZERO {
        T::ZERO.sub(T::ONE)
    } else {
        T::ONE
    };
    // α = −sign·‖x‖; v₀ ← x₀ − α = x₀ + sign·‖x‖.
    v[0] = v[0].add(sign.mul(norm));

    let mut vnorm_sq = T::ZERO;
    for &value in &v {
        vnorm_sq = vnorm_sq.add(value.mul(value));
    }
    if vnorm_sq <= T::ZERO {
        return None;
    }
    let beta = T::ONE.add(T::ONE).div(vnorm_sq);

    Some(Reflector { v, beta, offset })
}

/// Apply `P` from the **left**: `H[offset.., :] ← P · H[offset.., :]`
/// (`P·H = H − β v (vᵀH)`), updating every column.
pub(super) fn apply_left<T: RealScalar>(refl: &Reflector<T>, h: &mut [T], n: usize) {
    let Reflector { v, beta, offset } = refl;
    for j in 0..n {
        let mut dot = T::ZERO;
        for (i, &vi) in v.iter().enumerate() {
            dot = dot.add(vi.mul(h[(offset + i) * n + j]));
        }
        let scale = beta.mul(dot);
        for (i, &vi) in v.iter().enumerate() {
            let idx = (offset + i) * n + j;
            h[idx] = h[idx].sub(scale.mul(vi));
        }
    }
}

/// Apply `P` from the **right**: `M[:, offset..] ← M[:, offset..] · P`
/// (`M·P = M − β (Mv) vᵀ`), updating every row.
pub(super) fn apply_right<T: RealScalar>(refl: &Reflector<T>, m: &mut [T], n: usize) {
    let Reflector { v, beta, offset } = refl;
    for i in 0..n {
        let mut dot = T::ZERO;
        for (j, &vj) in v.iter().enumerate() {
            dot = dot.add(m[i * n + (offset + j)].mul(vj));
        }
        let scale = beta.mul(dot);
        for (j, &vj) in v.iter().enumerate() {
            let idx = i * n + (offset + j);
            m[idx] = m[idx].sub(scale.mul(vj));
        }
    }
}
