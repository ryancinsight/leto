//! LU elimination kernels: the shared partial-pivot step and the
//! right-looking blocked driver.

use crate::application::matrix::matmul_accumulate;
use crate::domain::real::RealScalar;
use leto::{ArrayView2, ArrayViewMut2, Layout, LetoError, Result};

/// Panel width of the blocked LU path: 64 columns per panel, the width the
/// 2026-07-20 experiment verified correct (`P·A = L·U` at n = 200). A panel
/// streams `n × 64` elements — L2-resident into the multi-thousand `n` where
/// the gate engages — while the trailing update's arithmetic intensity
/// (`2·64` flops per element) is fully BLAS-3.
const LU_PANEL_WIDTH: usize = 64;

/// One partial-pivot elimination step over row-major `a` (`n × n`): pivot
/// search in column `k`, full-row swap, column scale, and the rank-1 update
/// of row segments `k + 1..update_end` through the SIMD `axpy_slice`.
///
/// Both paths share it: unblocked elimination runs it with `update_end = n`,
/// and the blocked path's panel factorization with `update_end` at the panel
/// end (the trailing update is a GEMM there). Full-row swaps keep the
/// already-factored `L` columns consistent in both.
pub(crate) fn lu_eliminate_step<T: RealScalar>(
    a: &mut [T],
    n: usize,
    k: usize,
    update_end: usize,
    pivots: &mut [usize],
    sign: &mut i8,
) -> Result<()> {
    // Partial pivot: row with the largest |a[r][k]| for r >= k.
    let mut pivot_row = k;
    let mut pivot_mag = a[k * n + k].abs();
    for r in (k + 1)..n {
        let mag = a[r * n + k].abs();
        if mag > pivot_mag {
            pivot_mag = mag;
            pivot_row = r;
        }
    }
    if pivot_mag == T::ZERO {
        return Err(LetoError::StorageError {
            reason: format!("LU pivot column {k} is exactly zero: matrix is singular"),
        });
    }
    if pivot_row != k {
        for c in 0..n {
            a.swap(k * n + c, pivot_row * n + c);
        }
        pivots.swap(k, pivot_row);
        *sign = -*sign;
    }

    let pivot = a[k * n + k];
    // Rank-1 elimination of the trailing row segments. For each row r > k,
    // `a[r, k+1..update_end] -= (a[r,k]/pivot) · a[k, k+1..update_end]` — a
    // fused axpy over two contiguous, disjoint rows (k < r), dispatched
    // through the SIMD `axpy_slice` (SSOT with the matmul/QR row updates).
    // `split_at_mut` separates the pivot row (in `head`) from the trailing
    // rows (`tail`).
    let (head, tail) = a.split_at_mut((k + 1) * n);
    let pivot_row = &head[k * n + (k + 1)..k * n + update_end];
    for r in (k + 1)..n {
        let base = (r - (k + 1)) * n;
        let factor = tail[base + k].div(pivot);
        tail[base + k] = factor;
        let target = &mut tail[base + (k + 1)..base + update_end];
        // target += (−factor)·pivot_row  ≡  target −= factor·pivot_row.
        T::axpy_slice(T::ZERO.sub(factor), pivot_row, target);
    }
    Ok(())
}

/// Right-looking blocked LU (LAPACK `dgetrf` shape) over the row-major
/// working buffer: per panel, partial-pivot panel factorization through
/// [`lu_eliminate_step`], a unit-lower block-row solve, and a rank-`nb`
/// trailing update `A22 −= L21·U12` through [`matmul_accumulate`].
///
/// The GEMM inputs are packed contiguous copies — the negated `L21` and the
/// `U12` — into two buffers sized once to the widest panel and reused every
/// step, while the trailing block is a strided mutable view: no panel
/// extraction beyond the packing the GEMM's own dispatch performs for
/// strided inputs anyway (benchmarks.md "Blocked LU cache-resident
/// regression", condition b). The accumulation sign comes from the negated
/// `L21` pack because the GEMM has no alpha.
pub(crate) fn lu_eliminate_blocked<T: RealScalar>(
    a: &mut [T],
    n: usize,
    pivots: &mut [usize],
    sign: &mut i8,
) -> Result<()> {
    let mut neg_l21 = vec![T::ZERO; n * LU_PANEL_WIDTH];
    let mut u12 = vec![T::ZERO; n * LU_PANEL_WIDTH];
    let mut k = 0;
    while k < n {
        let nb = (n - k).min(LU_PANEL_WIDTH);
        let panel_end = k + nb;
        for j in k..panel_end {
            lu_eliminate_step(a, n, j, panel_end, pivots, sign)?;
        }
        if panel_end < n {
            let m = n - panel_end;
            // Top block row: solve L11·X = A12 in place (unit lower, forward
            // substitution down the panel rows for each trailing column).
            for c in panel_end..n {
                for r in k..panel_end {
                    let mut dot = T::ZERO;
                    for t in k..r {
                        dot = dot.add(a[r * n + t].mul(a[t * n + c]));
                    }
                    a[r * n + c] = a[r * n + c].sub(dot);
                }
            }
            // Pack the negated L21 (m × nb) and U12 (nb × m) panels.
            for (i, row) in (panel_end..n).enumerate() {
                for (j, col) in (k..panel_end).enumerate() {
                    neg_l21[i * nb + j] = T::ZERO.sub(a[row * n + col]);
                }
            }
            for (i, row) in (k..panel_end).enumerate() {
                for (j, col) in (panel_end..n).enumerate() {
                    u12[i * m + j] = a[row * n + col];
                }
            }
            // Trailing update through the GEMM: A22 += (−L21)·U12.
            let lhs: ArrayView2<'_, T> =
                ArrayView2::try_new(Layout::c_contiguous([m, nb])?, &neg_l21[..m * nb])?;
            let rhs: ArrayView2<'_, T> =
                ArrayView2::try_new(Layout::c_contiguous([nb, m])?, &u12[..nb * m])?;
            let stride = isize::try_from(n)
                .expect("LU dimension fits a view stride: n² elements are allocated");
            let mut trailing: ArrayViewMut2<'_, T> = ArrayViewMut2::try_new(
                Layout::try_new([m, m], [stride, 1], panel_end * n + panel_end)?,
                &mut *a,
            )?;
            matmul_accumulate(&lhs, &rhs, &mut trailing)?;
        }
        k = panel_end;
    }
    Ok(())
}
