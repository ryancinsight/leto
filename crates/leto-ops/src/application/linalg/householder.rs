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
    if c1 <= c0 {
        return;
    }
    // Row-oriented `P·m`: accumulate `w = vᵀ·m[rows, c0..c1]` by sweeping each
    // reflector row contiguously, then apply `m −= v·(β·w)` row by row — versus a
    // per-column dot that strides the reflector rows at `cols` apart. Both inner
    // sweeps are contiguous element-wise `y += a·x` updates, dispatched through the
    // SIMD `Scalar::axpy_slice` (SSOT with the LU/QR/matmul row updates). The
    // per-`w[j]` summation order (reflector rows ascending) and the `(β·w)` scaling
    // grouping are preserved; `axpy_slice` is bitwise-identical to the separate
    // `mul`+`add` it replaces (hermes `axpy` performs no FMA contraction), so the
    // result is unchanged to the last bit — the eigenvalue/SVD paths see no
    // rounding perturbation.
    let span = c1 - c0;
    let mut w = vec![T::ZERO; span];
    for (i, &vi) in v.iter().enumerate() {
        let row = &m[(base_row + i) * cols + c0..(base_row + i) * cols + c1];
        T::axpy_slice(vi, row, &mut w); // w += vᵢ · row
    }
    for wj in w.iter_mut() {
        *wj = beta.mul(*wj);
    }
    for (i, &vi) in v.iter().enumerate() {
        let row = &mut m[(base_row + i) * cols + c0..(base_row + i) * cols + c1];
        T::axpy_slice(T::ZERO.sub(vi), &w, row); // row += (−vᵢ)·w  ≡  row −= vᵢ·w
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
    let len = v.len();
    for i in r0..r1 {
        let row = &mut m[i * cols + base_col..i * cols + base_col + len];
        // Reduction `dot = rowᵀ·v` stays sequential (ascending `j`) to preserve the
        // summation order; vectorizing it via a tree reduction would reorder the
        // adds and perturb the rounding of the eigenvalue-feeding paths.
        let mut dot = T::ZERO;
        for (&mij, &vj) in row.iter().zip(v.iter()) {
            dot = dot.add(mij.mul(vj));
        }
        let scale = beta.mul(dot);
        // The element-wise update is contiguous; route it through the SIMD
        // `axpy_slice` (bitwise-identical to the separate `mul`+`sub`).
        T::axpy_slice(T::ZERO.sub(scale), v, row); // row += (−scale)·v ≡ row −= scale·v
    }
}
