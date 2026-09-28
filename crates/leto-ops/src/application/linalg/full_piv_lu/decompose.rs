//! Complete-pivoting Gaussian elimination `P A Q = L U`.

use crate::application::linalg::thresholds::rank_pivot_ratio;
use crate::domain::real::RealScalar;
use leto::{ArrayView2, LetoError, Result};

/// Outcome of the elimination: the packed `L\U` factors (row-major `n×n`, unit
/// `L` below the diagonal, `U` on and above), the row and column permutations
/// (`perm[k]` = original index now at position `k`), the permutation-parity
/// sign, and the numerical rank.
pub(super) struct Factored<T> {
    pub(super) lu: Vec<T>,
    pub(super) row_perm: Vec<usize>,
    pub(super) col_perm: Vec<usize>,
    pub(super) sign: i8,
    pub(super) rank: usize,
    pub(super) n: usize,
}

/// Factor a square matrix with complete pivoting.
///
/// At step `k` the pivot is the largest-magnitude entry of the trailing
/// submatrix `[k.., k..]`, brought to `(k, k)` by a row and a column swap. When
/// the largest remaining entry falls below a relative threshold the matrix is
/// rank-deficient and the rank is fixed at `k`.
pub(super) fn factor<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Factored<T>> {
    let [n, cols] = matrix.shape();
    if n != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![n, cols],
            rhs: vec![n, n],
        });
    }

    let mut a = if let Some(slice) = matrix.as_slice() {
        slice.to_vec()
    } else {
        matrix.to_contiguous().into_storage().into_inner()
    };
    let mut global_max = T::ZERO;
    for &value in &a {
        if !value.is_finite() {
            return Err(LetoError::StorageError {
                reason: "FullPivLU input contains a non-finite value".to_string(),
            });
        }
        if value.abs() > global_max {
            global_max = value.abs();
        }
    }

    let mut row_perm: Vec<usize> = (0..n).collect();
    let mut col_perm: Vec<usize> = (0..n).collect();
    let mut sign = 1i8;
    let mut rank = n;
    // Relative pivot floor: a trailing block below this is treated as zero.
    let tol = global_max.mul(rank_pivot_ratio::<T>());

    for k in 0..n {
        // Locate the largest-magnitude entry in the trailing submatrix.
        let mut best_i = k;
        let mut best_j = k;
        let mut best = a[k * n + k].abs();
        for i in k..n {
            for j in k..n {
                let mag = a[i * n + j].abs();
                if mag > best {
                    best = mag;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        if best <= tol {
            rank = k;
            break;
        }

        if best_i != k {
            for j in 0..n {
                a.swap(k * n + j, best_i * n + j);
            }
            row_perm.swap(k, best_i);
            sign = -sign;
        }
        if best_j != k {
            for i in 0..n {
                a.swap(i * n + k, i * n + best_j);
            }
            col_perm.swap(k, best_j);
            sign = -sign;
        }

        let pivot = a[k * n + k];
        // Rank-1 elimination of the trailing submatrix: for each row `i > k`,
        // `a[i, k+1..] -= (a[i,k]/pivot) · a[k, k+1..]` — a fused axpy over
        // two contiguous, disjoint rows (`k < i`), dispatched through the
        // SIMD `axpy_slice` (SSOT with `lu.rs`'s partial-pivot row update).
        // `split_at_mut` separates the pivot row (`head`) from the trailing
        // rows (`tail`).
        let (head, tail) = a.split_at_mut((k + 1) * n);
        let pivot_row = &head[k * n + (k + 1)..k * n + n];
        for i in (k + 1)..n {
            let base = (i - (k + 1)) * n;
            let factor = tail[base + k].div(pivot);
            tail[base + k] = factor;
            let target = &mut tail[base + (k + 1)..base + n];
            // target += (−factor)·pivot_row  ≡  target −= factor·pivot_row.
            T::axpy_slice(T::ZERO.sub(factor), pivot_row, target);
        }
    }

    Ok(Factored {
        lu: a,
        row_perm,
        col_perm,
        sign,
        rank,
        n,
    })
}
