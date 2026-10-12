use crate::application::matrix::matmul_accumulate;
use crate::domain::real::RealScalar;
use crate::infrastructure::cache::{
    cached_cache_geometry, working_set_spills_llc_for_geometry, CacheGeometry,
};
use leto::{
    Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Layout, LetoError,
    Result, Storage,
};

/// Panel width of the blocked LU path: 64 columns per panel, the width the
/// 2026-07-20 experiment verified correct (`P·A = L·U` at n = 200). A panel
/// streams `n × 64` elements — L2-resident into the multi-thousand `n` where
/// the gate engages — while the trailing update's arithmetic intensity
/// (`2·64` flops per element) is fully BLAS-3.
const LU_PANEL_WIDTH: usize = 64;

/// Whether an `n × n` LU over `element_bytes`-wide scalars runs the blocked
/// (BLAS-3) path: exactly when the factor working set spills past the shared
/// LLC, the same cache-aware threshold the parallel policy uses
/// ([`working_set_spills_llc_for_geometry`]). Cache-resident sizes keep the
/// unblocked SIMD-axpy path, which runs at cache bandwidth there.
pub(crate) fn lu_uses_blocked_path(
    geometry: CacheGeometry,
    n: usize,
    element_bytes: usize,
) -> bool {
    let working_set = n.saturating_mul(n).saturating_mul(element_bytes);
    working_set_spills_llc_for_geometry(geometry, working_set)
}

/// LU decomposition with partial pivoting: `P · A = L · U`.
///
/// `L` (unit lower-triangular) and `U` (upper-triangular) are packed into one
/// row-major `n × n` factor matrix (`L` strictly below the diagonal, `U` on
/// and above it); `pivots` records the row permutation; `sign` is the
/// permutation parity (`+1`/`-1`), used by [`det`](Self::det).
///
/// Generic over `T: RealScalar`; elimination runs in the native precision of
/// `T` per the `Scalar` contract (no hidden widening — a caller needing a
/// wider working precision converts the input explicitly). Driver: CFDrs
/// `cfd-math` dense solver paths (leto replacement Stage A1).
#[derive(Debug, Clone)]
pub struct LuDecomposition<T> {
    factors: Array2<T>,
    pivots: Vec<usize>,
    sign: i8,
}

/// Compute the partially pivoted LU decomposition of a square matrix.
///
/// The input may be strided/transposed; it is copied once into row-major
/// working storage. Elimination dispatches on the factor working set
/// (the crate-private `lu_uses_blocked_path` gate): past the shared LLC it
/// runs right-looking blocked LU with BLAS-3 trailing updates, while
/// cache-resident sizes keep
/// the unblocked SIMD-axpy path, which runs at cache bandwidth there
/// (benchmarks.md "Blocked LU cache-resident regression"). Returns
/// [`LetoError::ShapeMismatch`] for non-square input and
/// [`LetoError::StorageError`] when a pivot column is exactly zero (the
/// matrix is singular to working precision).
pub fn lu_decompose<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<LuDecomposition<T>> {
    let [rows, cols] = matrix.shape();
    if rows != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows, rows],
        });
    }
    let n = rows;

    // One bulk row-major copy into working storage + a single finiteness scan
    // (replacing per-element bounds-checked gets); elimination is in place.
    let mut a = matrix.to_contiguous().into_storage().into_inner();
    if !a.iter().all(|value| value.is_finite()) {
        return Err(LetoError::StorageError {
            reason: "LU input contains a non-finite value".to_string(),
        });
    }

    let mut pivots: Vec<usize> = (0..n).collect();
    let mut sign = 1i8;

    if lu_uses_blocked_path(cached_cache_geometry(), n, core::mem::size_of::<T>()) {
        lu_eliminate_blocked(&mut a, n, &mut pivots, &mut sign)?;
    } else {
        for k in 0..n {
            lu_eliminate_step(&mut a, n, k, n, &mut pivots, &mut sign)?;
        }
    }

    Ok(LuDecomposition {
        factors: Array2::from_shape_vec([n, n], a).expect("square factor storage"),
        pivots,
        sign,
    })
}

/// One partial-pivot elimination step over row-major `a` (`n × n`): pivot
/// search in column `k`, full-row swap, column scale, and the rank-1 update
/// of row segments `k + 1..update_end` through the SIMD `axpy_slice`.
///
/// Both paths share it: unblocked elimination runs it with `update_end = n`,
/// and the blocked path's panel factorization with `update_end` at the panel
/// end (the trailing update is a GEMM there). Full-row swaps keep the
/// already-factored `L` columns consistent in both.
fn lu_eliminate_step<T: RealScalar>(
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
fn lu_eliminate_blocked<T: RealScalar>(
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
impl<T: RealScalar> LuDecomposition<T> {
    /// Construct an LU decomposition directly from its raw components.
    #[must_use]
    #[inline]
    pub fn from_raw_parts(factors: Array2<T>, pivots: Vec<usize>, sign: i8) -> Self {
        Self {
            factors,
            pivots,
            sign,
        }
    }

    /// Matrix dimension `n`.
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.pivots.len()
    }

    /// The packed `L`/`U` factor matrix (unit `L` strictly below the
    /// diagonal, `U` on and above it), in row-major order.
    #[must_use]
    #[inline]
    pub fn factors(&self) -> &Array2<T> {
        &self.factors
    }

    /// The permutation pivots vector.
    #[must_use]
    #[inline]
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }

    /// Determinant of the original matrix: parity × product of `U`'s diagonal.
    #[must_use]
    pub fn det(&self) -> T {
        let n = self.dim();
        let mut det = if self.sign >= 0 {
            T::ONE
        } else {
            T::ZERO.sub(T::ONE)
        };
        for k in 0..n {
            det = det.mul(*self.factors.get([k, k]).expect("diagonal in bounds"));
        }
        det
    }

    /// Solve `A · x = rhs` directly into a caller-owned view `out`.
    #[allow(clippy::needless_range_loop)]
    pub fn solve_into(
        &self,
        rhs: &ArrayView1<'_, T>,
        out: &mut ArrayViewMut1<'_, T>,
    ) -> Result<()> {
        let n = self.dim();
        if rhs.shape() != [n] {
            return Err(LetoError::ShapeMismatch {
                lhs: rhs.shape().to_vec(),
                rhs: vec![n],
            });
        }
        if out.shape() != [n] {
            return Err(LetoError::ShapeMismatch {
                lhs: out.shape().to_vec(),
                rhs: vec![n],
            });
        }

        if let Some(out_slice) = out.as_mut_slice() {
            if let Some(rhs_slice) = rhs.as_slice() {
                for k in 0..n {
                    out_slice[k] = rhs_slice[self.pivots[k]];
                }
            } else {
                for k in 0..n {
                    out_slice[k] = *rhs.get([self.pivots[k]])?;
                }
            }
            self.solve_in_place(out_slice);
        } else {
            if let Some(rhs_slice) = rhs.as_slice() {
                for k in 0..n {
                    *out.get_mut([k])? = rhs_slice[self.pivots[k]];
                }
            } else {
                for k in 0..n {
                    *out.get_mut([k])? = *rhs.get([self.pivots[k]])?;
                }
            }
            let f = self.factors.storage().as_slice();

            // Forward: L · y = P · rhs (unit diagonal). yᵣ = xᵣ − Σ_{c<r} L[r,c]·y_c.
            for r in 1..n {
                let mut dot = T::ZERO;
                for c in 0..r {
                    let f_rc = f[r * n + c];
                    let x_c = *out.get([c])?;
                    dot = dot.add(f_rc.mul(x_c));
                }
                let out_r = out.get_mut([r])?;
                *out_r = out_r.sub(dot);
            }
            // Backward: U · x = y. xᵣ = (yᵣ − Σ_{c>r} U[r,c]·x_c) / U[r,r].
            for r in (0..n).rev() {
                let mut dot = T::ZERO;
                for c in (r + 1)..n {
                    let f_rc = f[r * n + c];
                    let x_c = *out.get([c])?;
                    dot = dot.add(f_rc.mul(x_c));
                }
                let out_r = out.get_mut([r])?;
                *out_r = out_r.sub(dot).div(f[r * n + r]);
            }
        }
        Ok(())
    }

    /// Solve `A · x = rhs` for one right-hand-side vector.
    pub fn solve(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        let n = self.dim();
        let mut out = Array1::from_elem([n], T::ZERO);
        self.solve_into(rhs, &mut out.view_mut())?;
        Ok(out)
    }

    /// Inverse of the original matrix, solving against the identity columns.
    pub fn inv(&self) -> Result<Array2<T>> {
        let n = self.dim();
        let mut out = vec![T::ZERO; n * n];
        let mut column = vec![T::ZERO; n];
        for j in 0..n {
            // Permuted j-th identity column.
            for (k, slot) in column.iter_mut().enumerate() {
                *slot = if self.pivots[k] == j { T::ONE } else { T::ZERO };
            }
            self.solve_in_place(&mut column);
            for r in 0..n {
                out[r * n + j] = column[r];
            }
        }
        Array2::from_shape_vec([n, n], out)
    }

    /// Forward- then back-substitution over the packed factors. `x` arrives
    /// already permuted and leaves holding the solution.
    ///
    /// The packed `L`/`U` is row-major, so each substitution row is a contiguous
    /// slice and the row·partial-solution reduction is a contiguous dot product
    /// dispatched through the SIMD [`Scalar::dot_slice`] (SSOT with the other
    /// contraction paths) — replacing the per-element bounds-checked logical
    /// `Array2::get`, whose `O(n³)` invocation by [`inv`](Self::inv) dominated the
    /// triangular solve. Inverse/solve are correspondingly faster (e.g. they back
    /// the matrix exponential's Padé denominator inverse).
    fn solve_in_place(&self, x: &mut [T]) {
        let n = self.dim();
        let f = self.factors.storage().as_slice(); // row-major packed L (below) / U (on+above)

        // Forward: L · y = P · rhs (unit diagonal). yᵣ = xᵣ − Σ_{c<r} L[r,c]·y_c.
        for r in 1..n {
            let dot = T::dot_slice(&f[r * n..r * n + r], &x[..r]);
            x[r] = x[r].sub(dot);
        }
        // Backward: U · x = y. xᵣ = (yᵣ − Σ_{c>r} U[r,c]·x_c) / U[r,r].
        for r in (0..n).rev() {
            let dot = T::dot_slice(&f[r * n + r + 1..r * n + n], &x[r + 1..n]);
            x[r] = x[r].sub(dot).div(f[r * n + r]);
        }
    }
}

/// Convenience: decompose and solve `A · x = rhs` in one call.
#[inline]
pub fn solve<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>> {
    lu_decompose(matrix)?.solve(rhs)
}

/// Convenience: determinant via LU. Singular matrices return zero rather
/// than an error (the zero-pivot rejection applies to solving, not to the
/// determinant value itself).
pub fn det<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<T> {
    match lu_decompose(matrix) {
        Ok(decomposition) => Ok(decomposition.det()),
        Err(LetoError::StorageError { reason }) if reason.contains("singular") => Ok(T::ZERO),
        Err(other) => Err(other),
    }
}

/// Convenience: matrix inverse via LU.
#[inline]
pub fn inv<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>> {
    lu_decompose(matrix)?.inv()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagonally_dominant(n: usize) -> Vec<f64> {
        let mut data = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                data[i * n + j] = ((i * 7 + j * 13) % 17) as f64 * 0.01 - 0.08;
            }
            data[i * n + i] += n as f64;
        }
        data
    }

    fn eliminate_unblocked(a: &mut [f64], n: usize, pivots: &mut [usize], sign: &mut i8) {
        for k in 0..n {
            lu_eliminate_step(a, n, k, n, pivots, sign).expect("fixture is nonsingular");
        }
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn blocked_path_matches_unblocked_within_rounding() {
        // n = 100: a full 64-wide panel, a partial panel, and two GEMM
        // trailing updates. Same pivot sequence, factors within ~n·ε.
        let n = 100usize;
        let data = diagonally_dominant(n);
        let mut blocked = data.clone();
        let mut blocked_pivots: Vec<usize> = (0..n).collect();
        let mut blocked_sign = 1i8;
        lu_eliminate_blocked(&mut blocked, n, &mut blocked_pivots, &mut blocked_sign)
            .expect("fixture is nonsingular");
        let mut plain = data.clone();
        let mut plain_pivots: Vec<usize> = (0..n).collect();
        let mut plain_sign = 1i8;
        eliminate_unblocked(&mut plain, n, &mut plain_pivots, &mut plain_sign);
        assert_eq!(blocked_pivots, plain_pivots);
        assert_eq!(blocked_sign, plain_sign);
        let err = max_abs_diff(&blocked, &plain);
        assert!(err < 1e-9, "blocked-vs-unblocked max factor diff {err}");
    }

    #[test]
    fn blocked_path_matches_unblocked_for_reduced_width_scalar() {
        // Same n = 100 shape in f32: ~n·ε₃₂ rounding over O(n) entries.
        let n = 100usize;
        let data: Vec<f32> = diagonally_dominant(n).iter().map(|&v| v as f32).collect();
        let mut blocked = data.clone();
        let mut blocked_pivots: Vec<usize> = (0..n).collect();
        let mut blocked_sign = 1i8;
        lu_eliminate_blocked(&mut blocked, n, &mut blocked_pivots, &mut blocked_sign)
            .expect("fixture is nonsingular");
        let mut plain = data.clone();
        let mut plain_pivots: Vec<usize> = (0..n).collect();
        let mut plain_sign = 1i8;
        for k in 0..n {
            lu_eliminate_step(&mut plain, n, k, n, &mut plain_pivots, &mut plain_sign)
                .expect("fixture is nonsingular");
        }
        assert_eq!(blocked_pivots, plain_pivots);
        assert_eq!(blocked_sign, plain_sign);
        let err: f32 = blocked
            .iter()
            .zip(&plain)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-2, "blocked-vs-unblocked max factor diff {err}");
    }

    #[test]
    fn sub_panel_orders_are_bitwise_identical_to_unblocked() {
        // At most one panel and no trailing update: the blocked kernel runs
        // the same step sequence as unblocked elimination, so the factors,
        // pivots, and sign agree bit for bit.
        for n in [1usize, 2, 63, 64] {
            let data = diagonally_dominant(n);
            let mut blocked = data.clone();
            let mut blocked_pivots: Vec<usize> = (0..n).collect();
            let mut blocked_sign = 1i8;
            lu_eliminate_blocked(&mut blocked, n, &mut blocked_pivots, &mut blocked_sign)
                .expect("fixture is nonsingular");
            let mut plain = data.clone();
            let mut plain_pivots: Vec<usize> = (0..n).collect();
            let mut plain_sign = 1i8;
            eliminate_unblocked(&mut plain, n, &mut plain_pivots, &mut plain_sign);
            assert_eq!(blocked, plain, "factors differ at n = {n}");
            assert_eq!(blocked_pivots, plain_pivots, "pivots differ at n = {n}");
            assert_eq!(blocked_sign, plain_sign, "sign differs at n = {n}");
        }
    }

    #[test]
    fn partial_second_panel_matches_unblocked() {
        // n = 65: full panel, a 1×1 GEMM update, then a 1-wide panel.
        let n = 65usize;
        let data = diagonally_dominant(n);
        let mut blocked = data.clone();
        let mut blocked_pivots: Vec<usize> = (0..n).collect();
        let mut blocked_sign = 1i8;
        lu_eliminate_blocked(&mut blocked, n, &mut blocked_pivots, &mut blocked_sign)
            .expect("fixture is nonsingular");
        let mut plain = data.clone();
        let mut plain_pivots: Vec<usize> = (0..n).collect();
        let mut plain_sign = 1i8;
        eliminate_unblocked(&mut plain, n, &mut plain_pivots, &mut plain_sign);
        assert_eq!(blocked_pivots, plain_pivots);
        assert_eq!(blocked_sign, plain_sign);
        let err = max_abs_diff(&blocked, &plain);
        assert!(err < 1e-9, "blocked-vs-unblocked max factor diff {err}");
    }

    #[test]
    fn blocked_path_rejects_singular_like_unblocked() {
        // Row 2 = 2 × row 1: rank deficient; both paths report the zero
        // pivot column with the same reason.
        let mut blocked = vec![1.0f64, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 0.0, 1.0];
        let mut pivots: Vec<usize> = (0..3).collect();
        let mut sign = 1i8;
        let blocked_err = lu_eliminate_blocked(&mut blocked, 3, &mut pivots, &mut sign)
            .expect_err("singular matrix must fail");
        assert!(
            format!("{blocked_err:?}").contains("singular"),
            "unexpected error: {blocked_err:?}"
        );
        let mut plain = vec![1.0f64, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 0.0, 1.0];
        let mut pivots: Vec<usize> = (0..3).collect();
        let mut sign = 1i8;
        let plain_err = (0..3)
            .try_for_each(|k| lu_eliminate_step(&mut plain, 3, k, 3, &mut pivots, &mut sign))
            .expect_err("singular matrix must fail");
        assert!(
            format!("{plain_err:?}").contains("singular"),
            "unexpected error: {plain_err:?}"
        );
    }

    #[test]
    fn gate_pins_the_spill_boundary() {
        // Against the 8 MiB fallback LLC: exactly 8 MiB stays unblocked
        // (the gate is `>`, not `>=`), one element over routes blocked.
        let fallback = CacheGeometry::fallback();
        assert!(!lu_uses_blocked_path(fallback, 1024, 8));
        assert!(lu_uses_blocked_path(fallback, 1025, 8));
        assert!(!lu_uses_blocked_path(fallback, 1448, 4));
        assert!(lu_uses_blocked_path(fallback, 1449, 4));
        assert!(!lu_uses_blocked_path(fallback, 0, 8));
        // Saturating arithmetic: absurd sizes route blocked, never wrap.
        assert!(lu_uses_blocked_path(fallback, usize::MAX, 8));
    }
}
