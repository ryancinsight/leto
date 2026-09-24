//! Householder reduction of a symmetric matrix to tridiagonal form, and the
//! accumulation of the orthogonal factor's transpose.

use crate::application::linalg::householder::reflect_in_place;
use crate::domain::real::RealScalar;

/// Reduce the symmetric row-major `n × n` matrix `a` to tridiagonal form.
///
/// On return `diagonal[k] = T[k,k]`, `off_diagonal[k] = T[k,k+1]`
/// (`off_diagonal[n−1] = 0`), and row `k` of `a` holds reflector `Hₖ`'s vector
/// `v` in columns `k+1 ..`, with `scales[k] = β` (`0` where step `k` needed no
/// reflection). `scratch` is resized to `n`.
///
/// Step `k` forms `Hₖ = I − β v vᵀ` mapping `x = A[k, k+1..]` to `α e₁`, then
/// updates the trailing block `B ← Hₖ B Hₖ` as the symmetric rank-2 update
/// `B − v wᵀ − w vᵀ` with `p = β B v` and `w = p − (β pᵀv / 2) v`
/// (Golub & Van Loan Algorithm 8.3.1). `vᵢ wⱼ + wᵢ vⱼ` is one floating-point
/// expression symmetric in `(i, j)`, so `B` stays exactly symmetric and each
/// trailing row can be read as the column it mirrors.
pub(super) fn tridiagonalize<T: RealScalar>(
    a: &mut [T],
    n: usize,
    diagonal: &mut [T],
    off_diagonal: &mut [T],
    scales: &mut [T],
    scratch: &mut Vec<T>,
) {
    scratch.clear();
    scratch.resize(n, T::ZERO);
    let half = T::ONE.div(T::from_usize(2));
    for k in 0..n.saturating_sub(2) {
        diagonal[k] = a[k * n + k];
        let (head, trailing) = a.split_at_mut((k + 1) * n);
        let v = &mut head[k * n + k + 1..];
        let Some((beta, alpha)) = reflect_in_place(v) else {
            off_diagonal[k] = T::ZERO;
            scales[k] = T::ZERO;
            continue;
        };
        off_diagonal[k] = alpha;
        scales[k] = beta;
        let v = &*v;
        let width = n - k - 1;
        let p = &mut scratch[..width];
        for (pi, row) in p.iter_mut().zip(trailing.chunks_exact(n)) {
            *pi = beta.mul(T::dot_slice(&row[k + 1..], v));
        }
        let correction = half.mul(beta).mul(T::dot_slice(p, v));
        for (pi, &vi) in p.iter_mut().zip(v) {
            *pi = pi.sub(correction.mul(vi));
        }
        let w = &*p;
        for ((&vi, &wi), row) in v.iter().zip(w).zip(trailing.chunks_exact_mut(n)) {
            for ((entry, &wj), &vj) in row[k + 1..].iter_mut().zip(w).zip(v) {
                *entry = entry.sub(vi.mul(wj).add(wi.mul(vj)));
            }
        }
    }
    match n {
        0 => {}
        1 => {
            diagonal[0] = a[0];
            off_diagonal[0] = T::ZERO;
        }
        _ => {
            diagonal[n - 2] = a[(n - 2) * n + n - 2];
            off_diagonal[n - 2] = a[(n - 2) * n + n - 1];
            diagonal[n - 1] = a[n * n - 1];
            off_diagonal[n - 1] = T::ZERO;
        }
    }
}

/// Write `Qᵀ = H_{n−3} ⋯ H₀` into `transposed`, from the reflectors
/// [`tridiagonalize`] left in `a` and `scales`.
///
/// Backward accumulation: `M ← I`, then `M ← M Hₖ` for `k = n−3, …, 0`. Before
/// step `k`, `M = H_{n−3} ⋯ H_{k+1}` differs from `I` only in the block
/// `[k+2.., k+2..]`, so `M Hₖ` changes only rows and columns `k+1 ..`; each
/// row update is a contiguous dot and `axpy` over that window.
pub(super) fn accumulate_transposed_factor<T: RealScalar>(
    a: &[T],
    n: usize,
    scales: &[T],
    transposed: &mut [T],
) {
    transposed.fill(T::ZERO);
    for i in 0..n {
        transposed[i * n + i] = T::ONE;
    }
    for k in (0..n.saturating_sub(2)).rev() {
        let beta = scales[k];
        if beta == T::ZERO {
            continue;
        }
        let v = &a[k * n + k + 1..(k + 1) * n];
        for row in transposed[(k + 1) * n..].chunks_exact_mut(n) {
            let window = &mut row[k + 1..];
            let scale = beta.mul(T::dot_slice(window, v));
            T::axpy_slice(T::ZERO.sub(scale), v, window);
        }
    }
}
