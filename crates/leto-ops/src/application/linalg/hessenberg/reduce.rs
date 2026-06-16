//! The Householder reduction loop `A → (Q, H)` with `H = Qᵀ A Q`.

use crate::application::linalg::householder::{apply_left, apply_right, reflector};
use crate::domain::real::RealScalar;
use leto::{ArrayView2, Result};

/// Reduce `A` (n×n) to upper Hessenberg `H`, optionally accumulating the
/// orthogonal `Q`, returning row-major `(h, q, n)` such that `A = Q H Qᵀ`.
///
/// For each column `k = 0 … n−3` a Householder reflector `Pₖ` zeroes
/// `H[k+2.. ][k]`; applying it on both sides (`H ← Pₖ H Pₖ`) is a similarity
/// transform, so the spectrum is preserved while the subdiagonal structure is
/// created. `Q = P₀ P₁ … P_{n-3}` is accumulated by right-multiplication.
/// Below-subdiagonal entries are explicitly zeroed at the end (already
/// negligible by construction); this yields the exact upper-Hessenberg form.
///
/// `ACCUMULATE_Q` is a compile-time switch: when `false` (the eigenvalue-only
/// path, which is similarity-invariant and never reads `Q`) the `Q` allocation
/// and its per-reflector `apply_right` update — together an `O(n³)` cost equal to
/// the `H` update itself — are eliminated by dead-code elimination, and the
/// returned `q` is empty. When `true` the full `Q` is built. The two
/// instantiations are monomorphized independently, so neither pays for the
/// other's branch.
pub(super) fn reduce_to_hessenberg<T: RealScalar, const ACCUMULATE_Q: bool>(
    matrix: &ArrayView2<'_, T>,
) -> Result<(Vec<T>, Vec<T>, usize)> {
    let [n, _] = matrix.shape();

    // Row-major working copy H ← A.
    let mut h = vec![T::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] = *matrix.get([i, j])?;
        }
    }
    // Q ← I only when requested; otherwise an empty, never-touched buffer.
    let mut q = if ACCUMULATE_Q {
        let mut q = vec![T::ZERO; n * n];
        for i in 0..n {
            q[i * n + i] = T::ONE;
        }
        q
    } else {
        Vec::new()
    };

    // k runs to n−3 inclusive: the final subdiagonal entry needs no reflector.
    for k in 0..n.saturating_sub(2) {
        // Sub-column below the subdiagonal: rows k+1..n of column k.
        let x: Vec<T> = (k + 1..n).map(|i| h[i * n + k]).collect();
        if let Some((refl, _alpha)) = reflector(&x) {
            // Left apply starts at column k: the reflector touches rows [k+1, n),
            // which are already zero in columns [0, k) (those columns are reduced
            // to Hessenberg form by prior steps), so the skipped columns are a
            // provable no-op. Column k itself is transformed (it is what this
            // reflector zeroes below the subdiagonal). The right apply must still
            // span all rows [0, n): the trailing columns [k+1, n) are unreduced and
            // dense above the diagonal. (Matches LAPACK dgehrd's trailing-submatrix
            // update.)
            apply_left(&refl, &mut h, n, k + 1, k, n); // H ← Pₖ H
            apply_right(&refl, &mut h, n, k + 1, 0, n); // H ← (Pₖ H) Pₖ
            if ACCUMULATE_Q {
                apply_right(&refl, &mut q, n, k + 1, 0, n); // Q ← Q Pₖ
            }
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
