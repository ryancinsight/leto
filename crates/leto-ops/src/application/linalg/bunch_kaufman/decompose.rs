//! Bunch–Kaufman `P A Pᵀ = L D Lᵀ` factorization kernel (partial pivoting).

use crate::domain::real::RealScalar;
use leto::{ArrayView2, LetoError, Result};

/// Factored form: unit lower-triangular `L` (row-major `n×n`), block-diagonal
/// `D` (row-major `n×n`, with 1×1 and 2×2 blocks), the symmetric permutation
/// `perm` (`perm[i]` = original index now at position `i`), and `two[k] = true`
/// marking the start of a 2×2 pivot block at columns `(k, k+1)`.
#[derive(Debug, Clone)]
pub(super) struct Factored<T> {
    pub(super) l: Vec<T>,
    pub(super) d: Vec<T>,
    pub(super) perm: Vec<usize>,
    pub(super) two: Vec<bool>,
    pub(super) n: usize,
}

#[inline]
fn idx(i: usize, j: usize, n: usize) -> usize {
    i * n + j
}

/// Bunch–Kaufman partial-pivoting factorization of a symmetric matrix.
///
/// Maintains a full working copy `a` of the symmetric matrix (mirrored), applies
/// **symmetric** row+column interchanges, and eliminates one 1×1 or 2×2 pivot
/// block per step, accumulating the unit lower factor `L` and block-diagonal `D`.
pub(super) fn factor<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Factored<T>> {
    let [n, cols] = matrix.shape();
    if n != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![n, cols],
            rhs: vec![n, n],
        });
    }
    if n == 0 {
        return Ok(Factored {
            l: vec![],
            d: vec![],
            perm: vec![],
            two: vec![],
            n: 0,
        });
    }

    // Working full symmetric matrix.
    let mut a = vec![T::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            let value = *matrix.get([i, j])?;
            a[idx(i, j, n)] = value;
        }
    }

    // Symmetry + finiteness validation (same contract as UDU).
    let mut scale = T::ZERO;
    for &value in &a {
        if !value.is_finite() {
            return Err(LetoError::StorageError {
                reason: "Bunch-Kaufman input contains a non-finite value".to_string(),
            });
        }
        if value.abs() > scale {
            scale = value.abs();
        }
    }

    let sym_tol = scale.mul(T::ONE.div(T::from_usize(1_000_000_000)));
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = a[idx(i, j, n)].sub(a[idx(j, i, n)]).abs();
            if diff > sym_tol {
                return Err(LetoError::StorageError {
                    reason: "Bunch-Kaufman requires a symmetric matrix".to_string(),
                });
            }
        }
    }

    let mut l = vec![T::ZERO; n * n];
    for i in 0..n {
        l[idx(i, i, n)] = T::ONE; // unit diagonal
    }
    let mut d = vec![T::ZERO; n * n];
    let mut perm: Vec<usize> = (0..n).collect();
    let mut two = vec![false; n];

    // alpha = (1 + sqrt(17)) / 8, the Bunch–Kaufman pivot-growth threshold.
    let alpha = T::ONE.add(T::from_usize(17).sqrt()).div(T::from_usize(8));

    let mut k = 0usize;
    while k < n {
        // Largest off-diagonal magnitude in column k below the diagonal.
        let mut lambda = T::ZERO;
        let mut r = k;
        for i in (k + 1)..n {
            let mag = a[idx(i, k, n)].abs();
            if mag > lambda {
                lambda = mag;
                r = i;
            }
        }

        if lambda == T::ZERO {
            // Column already eliminated: 1×1 pivot, no interchange.
            d[idx(k, k, n)] = a[idx(k, k, n)];
            k += 1;
            continue;
        }

        let a_kk = a[idx(k, k, n)].abs();
        // Decide pivot kind and interchange target.
        let mut use_two = false;
        let mut swap_with = k; // 1×1 interchange target (k = none)
        if a_kk >= alpha.mul(lambda) {
            // 1×1, diagonal already acceptable.
        } else {
            // sigma = largest off-diagonal magnitude in column r (trailing block).
            let mut sigma = T::ZERO;
            for i in k..n {
                if i != r {
                    let mag = a[idx(i, r, n)].abs();
                    if mag > sigma {
                        sigma = mag;
                    }
                }
            }
            if a_kk.mul(sigma) >= alpha.mul(lambda).mul(lambda) {
                // 1×1 with the current diagonal.
            } else if a[idx(r, r, n)].abs() >= alpha.mul(sigma) {
                // 1×1 with interchange k ↔ r.
                swap_with = r;
            } else {
                // 2×2 pivot from {k, r}; bring r to position k+1.
                use_two = true;
                swap_with = r; // handled below as a (k+1 ↔ r) swap
            }
        }

        if use_two {
            symmetric_swap(&mut a, &mut l, &mut perm, k + 1, swap_with, k, n);
            eliminate_2x2(&mut a, &mut l, &mut d, &mut two, k, n)?;
            k += 2;
        } else {
            if swap_with != k {
                symmetric_swap(&mut a, &mut l, &mut perm, k, swap_with, k, n);
            }
            eliminate_1x1(&mut a, &mut l, &mut d, k, n);
            k += 1;
        }
    }

    Ok(Factored { l, d, perm, two, n })
}

/// Symmetric interchange of indices `p` and `q`: swap rows and columns `p,q` in
/// the working matrix `a`, swap rows `p,q` in already-computed columns of `L`,
/// and record the swap in `perm`.
fn symmetric_swap<T: RealScalar>(
    a: &mut [T],
    l: &mut [T],
    perm: &mut [usize],
    p: usize,
    q: usize,
    processed_cols: usize,
    n: usize,
) {
    if p == q {
        return;
    }
    for j in 0..n {
        a.swap(idx(p, j, n), idx(q, j, n));
    }
    for i in 0..n {
        a.swap(idx(i, p, n), idx(i, q, n));
    }
    for j in 0..processed_cols {
        l.swap(idx(p, j, n), idx(q, j, n));
    }
    perm.swap(p, q);
}

/// Eliminate a 1×1 pivot at column `k` and rank-1 update the trailing block.
fn eliminate_1x1<T: RealScalar>(a: &mut [T], l: &mut [T], d: &mut [T], k: usize, n: usize) {
    let pivot = a[idx(k, k, n)];
    d[idx(k, k, n)] = pivot;
    for i in (k + 1)..n {
        let factor = a[idx(i, k, n)].div(pivot);
        l[idx(i, k, n)] = factor;
    }

    let (row_k_part, trailing_part) = a.split_at_mut((k + 1) * n);
    let row_k = &row_k_part[k * n..];

    for i in (k + 1)..n {
        let factor = l[idx(i, k, n)];
        let row_i_idx = (i - (k + 1)) * n;
        let row_i = &mut trailing_part[row_i_idx..row_i_idx + n];
        for j in (k + 1)..n {
            let update = factor.mul(row_k[j]);
            row_i[j] = row_i[j].sub(update);
        }
    }
}

/// Eliminate a 2×2 pivot at columns `(k, k+1)` and rank-2 update the trailing
/// block.
fn eliminate_2x2<T: RealScalar>(
    a: &mut [T],
    l: &mut [T],
    d: &mut [T],
    two: &mut [bool],
    k: usize,
    n: usize,
) -> Result<()> {
    let e00 = a[idx(k, k, n)];
    let e01 = a[idx(k, k + 1, n)];
    let e11 = a[idx(k + 1, k + 1, n)];
    let det = e00.mul(e11).sub(e01.mul(e01));
    if det == T::ZERO {
        return Err(LetoError::StorageError {
            reason: "Bunch-Kaufman encountered a singular 2x2 pivot".to_string(),
        });
    }
    // D block.
    d[idx(k, k, n)] = e00;
    d[idx(k, k + 1, n)] = e01;
    d[idx(k + 1, k, n)] = e01;
    d[idx(k + 1, k + 1, n)] = e11;
    two[k] = true;

    // E⁻¹ = (1/det) [[e11, -e01], [-e01, e00]].
    let inv00 = e11.div(det);
    let inv01 = e01.neg().div(det);
    let inv11 = e00.div(det);

    for i in (k + 2)..n {
        let c0 = a[idx(i, k, n)];
        let c1 = a[idx(i, k + 1, n)];
        // [l_i] = [c0 c1] · E⁻¹  (E⁻¹ symmetric: inv10 = inv01).
        l[idx(i, k, n)] = c0.mul(inv00).add(c1.mul(inv01));
        l[idx(i, k + 1, n)] = c0.mul(inv01).add(c1.mul(inv11));
    }

    let (rows_k_k1, trailing_part) = a.split_at_mut((k + 2) * n);
    let row_k = &rows_k_k1[k * n..(k + 1) * n];
    let row_k1 = &rows_k_k1[(k + 1) * n..(k + 2) * n];

    for i in (k + 2)..n {
        let li0 = l[idx(i, k, n)];
        let li1 = l[idx(i, k + 1, n)];
        let row_i_idx = (i - (k + 2)) * n;
        let row_i = &mut trailing_part[row_i_idx..row_i_idx + n];
        for j in (k + 2)..n {
            let update = li0.mul(row_k[j]).add(li1.mul(row_k1[j]));
            row_i[j] = row_i[j].sub(update);
        }
    }
    Ok(())
}
