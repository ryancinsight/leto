//! The Householder reduction loop `A → (Q, H)` with `H = Qᵀ A Q`.

use super::householder::{apply_left, apply_right, reflector_for_column};
use crate::domain::real::RealScalar;
use leto::{ArrayView2, Result};

/// Reduce `A` (n×n) to upper Hessenberg `H` with the accumulated orthogonal
/// `Q`, returning row-major `(h, q, n)` such that `A = Q H Qᵀ`.
///
/// For each column `k = 0 … n−3` a Householder reflector `Pₖ` zeroes
/// `H[k+2.. ][k]`; applying it on both sides (`H ← Pₖ H Pₖ`) is a similarity
/// transform, so the spectrum is preserved while the subdiagonal structure is
/// created. `Q = P₀ P₁ … P_{n-3}` is accumulated by right-multiplication.
/// Below-subdiagonal entries are explicitly zeroed at the end (they are already
/// negligible by construction); this yields the exact upper-Hessenberg form.
pub(super) fn reduce_to_hessenberg<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<(Vec<T>, Vec<T>, usize)> {
    let [n, _] = matrix.shape();

    // Row-major working copy H ← A and Q ← I.
    let mut h = vec![T::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] = *matrix.get([i, j])?;
        }
    }
    let mut q = vec![T::ZERO; n * n];
    for i in 0..n {
        q[i * n + i] = T::ONE;
    }

    // k runs to n−3 inclusive: the final subdiagonal entry needs no reflector.
    for k in 0..n.saturating_sub(2) {
        if let Some(refl) = reflector_for_column(&h, n, k) {
            apply_left(&refl, &mut h, n); // H ← Pₖ H
            apply_right(&refl, &mut h, n); // H ← (Pₖ H) Pₖ
            apply_right(&refl, &mut q, n); // Q ← Q Pₖ
        }
    }

    // Present the exact upper-Hessenberg form (zero the negligible tail).
    for i in 0..n {
        for j in 0..i.saturating_sub(1) {
            h[i * n + j] = T::ZERO;
        }
    }

    Ok((h, q, n))
}
