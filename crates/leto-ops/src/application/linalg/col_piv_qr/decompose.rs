//! Householder QR with column pivoting: `A P = Q R`.

mod column_norms;
mod norm_bounds;
#[cfg(test)]
mod tests;

use self::column_norms::{tail_norm_sq, ColumnNorms, PartialColumnNorms};

use crate::application::linalg::householder::{apply_left, apply_right, reflector};
use crate::application::linalg::thresholds::rank_pivot_ratio;
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

/// Factor `A` (m×n) with column pivoting.
///
/// At step `k` the column of largest remaining (rows `k..m`) norm is pivoted to
/// position `k`, then a Householder reflector zeroes the sub-column below the
/// diagonal. Pivoting makes `|R₀₀| ≥ |R₁₁| ≥ …`, so the first diagonal entry
/// that drops below a relative threshold reveals the rank.
pub(super) fn factor<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Factored<T>> {
    factor_with_norms::<T, PartialColumnNorms<T>>(matrix)
}

fn factor_with_norms<T: RealScalar, N: ColumnNorms<T>>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Factored<T>> {
    let [m, n] = matrix.shape();

    let mut r = if let Some(slice) = matrix.as_slice() {
        slice.to_vec()
    } else {
        matrix.to_contiguous().into_storage().into_inner()
    };
    for &value in &r {
        if !value.is_finite() {
            return Err(LetoError::StorageError {
                reason: "ColPivQR input contains a non-finite value".to_string(),
            });
        }
    }

    // Q ← Iₘ.
    let mut q = vec![T::ZERO; m * m];
    for i in 0..m {
        q[i * m + i] = T::ONE;
    }
    let mut perm: Vec<usize> = (0..n).collect();

    let p = m.min(n);
    let mut norms = N::new(&r, n, m);
    // Relative threshold from the largest initial full column norm.
    let mut ref_norm_sq = T::ZERO;
    for j in 0..n {
        let norm_squared = norms.squared_norm(&r, n, m, j, 0);
        if norm_squared > ref_norm_sq {
            ref_norm_sq = norm_squared;
        }
    }
    let tol = ref_norm_sq.sqrt().mul(rank_pivot_ratio::<T>());
    let mut rank = p;

    let mut alw: Vec<T> = Vec::with_capacity(n);
    let mut col_stack = [T::ZERO; 128];
    let mut col_vec: Vec<T> = Vec::new();
    for k in 0..p {
        // Pivot: column with the largest tail norm among k..n.
        #[cfg(test)]
        norms.assert_bounds_cover_exact_keys(&r, n, m, k);
        let certified = norms.certified_pivot(k, n, m - k);
        let best = if let Some(best) = certified {
            best
        } else {
            norms.refresh_remaining(&r, n, m, k);
            let mut best = k;
            let mut best_norm_sq = norms.squared_norm(&r, n, m, k, k);
            for j in (k + 1)..n {
                let norm_squared = norms.squared_norm(&r, n, m, j, k);
                if norm_squared > best_norm_sq {
                    best_norm_sq = norm_squared;
                    best = j;
                }
            }
            best
        };
        // The baseline rank contract compares the selected column's exact
        // native-precision tail sum, even when interval separation certified
        // its pivot without recomputing every competing column.
        let best_norm_sq = tail_norm_sq(&r, n, m, best, k);
        if best_norm_sq.sqrt() <= tol {
            rank = k;
            break;
        }
        if best != k {
            for i in 0..m {
                r.swap(i * n + k, i * n + best);
            }
            perm.swap(k, best);
            norms.swap(k, best);
        }

        // Householder on column k, rows k..m.
        let len = m - k;
        let col = if len <= 128 {
            for i in 0..len {
                col_stack[i] = r[(k + i) * n + k];
            }
            &col_stack[..len]
        } else {
            col_vec.clear();
            col_vec.reserve(len);
            for i in 0..len {
                col_vec.push(r[(k + i) * n + k]);
            }
            &col_vec[..]
        };

        if let Some((refl, _alpha)) = reflector(col) {
            apply_left(&refl, &mut r, n, k, k, n, &mut alw); // rows k..m, cols k..n
            apply_right(&refl, &mut q, m, k, 0, m); // Q ← Q Hₖ
            norms.remove_row(&r, n, m, k, Some(&refl));
        } else {
            norms.remove_row(&r, n, m, k, None);
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
