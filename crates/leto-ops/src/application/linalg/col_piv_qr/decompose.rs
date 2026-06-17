//! Householder QR with column pivoting: `A P = Q R`.

use crate::application::linalg::householder::{apply_left, apply_right, reflector};
use crate::domain::real::RealScalar;
use leto::{ArrayView2, LetoError, Result};

/// Outcome of the pivoted QR: orthogonal `Q` (row-major `m×m`), upper-triangular
/// `R` (`m×n`), the column permutation (`perm[k]` = original column now at
/// position `k`), the numerical rank, and the dimensions.
pub(super) struct Factored<T> {
    pub(super) q: Vec<T>,
    pub(super) r: Vec<T>,
    pub(super) perm: Vec<usize>,
    pub(super) rank: usize,
    pub(super) m: usize,
    pub(super) n: usize,
}

/// Squared Euclidean norm of column `j` over rows `[r0 .. m)`.
fn tail_norm_sq<T: RealScalar>(r: &[T], n: usize, m: usize, j: usize, r0: usize) -> T {
    let mut acc = T::ZERO;
    for i in r0..m {
        let x = r[i * n + j];
        acc = acc.add(x.mul(x));
    }
    acc
}

/// Factor `A` (m×n) with column pivoting.
///
/// At step `k` the column of largest remaining (rows `k..m`) norm is pivoted to
/// position `k`, then a Householder reflector zeroes the sub-column below the
/// diagonal. Pivoting makes `|R₀₀| ≥ |R₁₁| ≥ …`, so the first diagonal entry
/// that drops below a relative threshold reveals the rank.
pub(super) fn factor<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Factored<T>> {
    let [m, n] = matrix.shape();

    let mut r = vec![T::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let value = *matrix.get([i, j])?;
            if !value.is_finite() {
                return Err(LetoError::StorageError {
                    reason: "ColPivQR input contains a non-finite value".to_string(),
                });
            }
            r[i * n + j] = value;
        }
    }

    // Q ← Iₘ.
    let mut q = vec![T::ZERO; m * m];
    for i in 0..m {
        q[i * m + i] = T::ONE;
    }
    let mut perm: Vec<usize> = (0..n).collect();

    let p = m.min(n);
    // Relative threshold from the largest initial full column norm.
    let mut ref_norm = T::ZERO;
    for j in 0..n {
        let nrm = tail_norm_sq(&r, n, m, j, 0).sqrt();
        if nrm > ref_norm {
            ref_norm = nrm;
        }
    }
    let tol = ref_norm.mul(T::ONE.div(T::from_usize(1_000_000_000_000)));
    let mut rank = p;

    let mut alw: Vec<T> = Vec::with_capacity(n);
    for k in 0..p {
        // Pivot: column with the largest tail norm among k..n.
        let mut best = k;
        let mut best_norm = tail_norm_sq(&r, n, m, k, k);
        for j in (k + 1)..n {
            let nrm = tail_norm_sq(&r, n, m, j, k);
            if nrm > best_norm {
                best_norm = nrm;
                best = j;
            }
        }
        if best_norm.sqrt() <= tol {
            rank = k;
            break;
        }
        if best != k {
            for i in 0..m {
                r.swap(i * n + k, i * n + best);
            }
            perm.swap(k, best);
        }

        // Householder on column k, rows k..m.
        let col: Vec<T> = (k..m).map(|i| r[i * n + k]).collect();
        if let Some((refl, _alpha)) = reflector(&col) {
            apply_left(&refl, &mut r, n, k, k, n, &mut alw); // rows k..m, cols k..n
            apply_right(&refl, &mut q, m, k, 0, m); // Q ← Q Hₖ
        }
    }

    // Present the exact upper-triangular R (zero the reflector tails below the diagonal).
    for i in 1..m {
        for j in 0..i.min(n) {
            r[i * n + j] = T::ZERO;
        }
    }

    Ok(Factored {
        q,
        r,
        perm,
        rank,
        m,
        n,
    })
}
