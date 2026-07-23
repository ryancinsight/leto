//! Two-sided Householder reduction `A → (U, B, V)` with `B = Uᵀ A V` upper
//! bidiagonal, for `m ≥ n`.

use crate::application::linalg::householder::{apply_left, apply_right, reflector};
use crate::domain::real::RealScalar;
use leto::{ArrayView2, LetoError, Result};

const BLOCK_WIDTH: usize = 16;

struct BidiagonalWork<T> {
    u: Option<Vec<T>>,
    b: Vec<T>,
    v: Option<Vec<T>>,
}

/// Reduce `A` (m×n, `m ≥ n`) to upper bidiagonal `B` with orthogonal `U` (m×m)
/// and `V` (n×n), returning row-major `(u, b, v)` such that `A = U B Vᵀ`.
///
/// At column `k` a **left** reflector zeroes `B[k+1.. ][k]` (building the
/// diagonal); for `k < n−1` a **right** reflector zeroes `B[k][k+2.. ]`
/// (building the superdiagonal). Each is an orthogonal transform, so the
/// singular values are preserved. `U = L₀…L_{n-1}` and `V = R₀…R_{n-2}` are
/// accumulated by right-multiplication. The off-bidiagonal tail is explicitly
/// zeroed at the end (already negligible by construction).
pub(super) fn reduce_to_bidiagonal<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    m: usize,
    n: usize,
) -> Result<(Vec<T>, Vec<T>, Vec<T>)> {
    let work = reduce_to_bidiagonal_impl::<T, true>(matrix, m, n)?;
    Ok((
        work.u
            .expect("invariant: ACCUMULATE_FACTORS=true returns U"),
        work.b,
        work.v
            .expect("invariant: ACCUMULATE_FACTORS=true returns V"),
    ))
}

fn reduce_to_bidiagonal_impl<T: RealScalar, const ACCUMULATE_FACTORS: bool>(
    matrix: &ArrayView2<'_, T>,
    m: usize,
    n: usize,
) -> Result<BidiagonalWork<T>> {
    // Working B ← A (m×n) via one bulk row-major copy, validating finiteness in
    // the same pass (SSOT — the caller no longer pre-scans element-by-element).
    // U ← I (m×m), V ← I (n×n) only when requested.
    let mut b = if let Some(slice) = matrix.as_slice() {
        slice.to_vec()
    } else {
        matrix.to_contiguous().into_storage().into_inner()
    };
    if !b.iter().all(|value| value.is_finite()) {
        return Err(LetoError::StorageError {
            reason: "bidiagonalization input contains a non-finite value".to_string(),
        });
    }
    let mut u = ACCUMULATE_FACTORS.then(|| identity::<T>(m));
    let mut v = ACCUMULATE_FACTORS.then(|| identity::<T>(n));

    // Reused scratch (allocation-free reduction hot loop): the left-apply
    // accumulator `w`.
    let mut alw: Vec<T> = Vec::with_capacity(n);
    let mut u_panel = Vec::<T>::new();
    let mut u_panel_beta = Vec::<T>::new();
    let mut v_panel = Vec::<T>::new();
    let mut v_panel_beta = Vec::<T>::new();

    let flush_u_panel =
        |u: &mut Option<Vec<T>>, u_rows: &mut Vec<T>, u_betas: &mut Vec<T>, dim: usize| {
            if !u_betas.is_empty() {
                if let Some(u) = u.as_mut() {
                    apply_reflectors_right(u_rows, u_betas, u, dim);
                }
                u_rows.clear();
                u_betas.clear();
            }
        };
    let flush_v_panel =
        |v: &mut Option<Vec<T>>, v_rows: &mut Vec<T>, v_betas: &mut Vec<T>, dim: usize| {
            if !v_betas.is_empty() {
                if let Some(v) = v.as_mut() {
                    apply_reflectors_right(v_rows, v_betas, v, dim);
                }
                v_rows.clear();
                v_betas.clear();
            }
        };

    for k in 0..n {
        // Left reflector: column k, rows k..m.
        let len_col = m - k;
        let mut col_stack = [T::ZERO; 128];
        let mut col_vec = Vec::new();
        let col = if len_col <= 128 {
            for i in 0..len_col {
                col_stack[i] = b[(k + i) * n + k];
            }
            &col_stack[..len_col]
        } else {
            col_vec.reserve_exact(len_col);
            for i in 0..len_col {
                col_vec.push(b[(k + i) * n + k]);
            }
            &col_vec[..]
        };

        if let Some((refl, _alpha)) = reflector(col) {
            apply_left(&refl, &mut b, n, k, k, n, &mut alw); // rows k..m, cols k..n
            if u.is_some() {
                for row in 0..m {
                    if row < k {
                        u_panel.push(T::ZERO);
                    } else {
                        u_panel.push(refl.v[row - k]);
                    }
                }
                u_panel_beta.push(refl.beta);
                if u_panel_beta.len() == BLOCK_WIDTH {
                    flush_u_panel(&mut u, &mut u_panel, &mut u_panel_beta, m);
                }
            }
        }

        // Right reflector: row k, columns k+1..n.
        if k + 1 < n {
            let len_row = n - (k + 1);
            let mut row_stack = [T::ZERO; 128];
            let mut row_vec = Vec::new();
            let row = if len_row <= 128 {
                for j in 0..len_row {
                    row_stack[j] = b[k * n + k + 1 + j];
                }
                &row_stack[..len_row]
            } else {
                row_vec.reserve_exact(len_row);
                for j in 0..len_row {
                    row_vec.push(b[k * n + k + 1 + j]);
                }
                &row_vec[..]
            };

            if let Some((refl, _alpha)) = reflector(row) {
                apply_right(&refl, &mut b, n, k + 1, k, m); // cols k+1..n, rows k..m
                if v.is_some() {
                    for row in 0..n {
                        if row < k + 1 {
                            v_panel.push(T::ZERO);
                        } else {
                            v_panel.push(refl.v[row - (k + 1)]);
                        }
                    }
                    v_panel_beta.push(refl.beta);
                    if v_panel_beta.len() == BLOCK_WIDTH {
                        flush_v_panel(&mut v, &mut v_panel, &mut v_panel_beta, n);
                    }
                }
            }
        }
    }

    flush_u_panel(&mut u, &mut u_panel, &mut u_panel_beta, m);
    flush_v_panel(&mut v, &mut v_panel, &mut v_panel_beta, n);

    // Present exact upper-bidiagonal form (keep (i,i) and (i,i+1) only). Needed
    // only when the `B` matrix is an output (factor path); the values-only path
    // reads just the diagonal/superdiagonal, so the `O(m·n)` zeroing of the
    // (negligible-by-construction) off-bidiagonal entries is wasted there and is
    // DCE'd by the const generic.
    if ACCUMULATE_FACTORS {
        for i in 0..m {
            for j in 0..n {
                if j < i || j > i + 1 {
                    b[i * n + j] = T::ZERO;
                }
            }
        }
    }

    Ok(BidiagonalWork { u, b, v })
}

fn identity<T: RealScalar>(n: usize) -> Vec<T> {
    let mut m = vec![T::ZERO; n * n];
    for i in 0..n {
        m[i * n + i] = T::ONE;
    }
    m
}

fn apply_reflectors_right<T: RealScalar>(vectors: &[T], betas: &[T], matrix: &mut [T], dim: usize) {
    let count = betas.len();
    debug_assert_eq!(vectors.len(), dim * count);
    debug_assert_eq!(matrix.len(), dim * dim);
    for reflector in 0..count {
        let vector = &vectors[reflector * dim..(reflector + 1) * dim];
        for row in 0..dim {
            let row_start = row * dim;
            let row_values = &mut matrix[row_start..row_start + dim];
            // Per-row reflector apply over full-dimension contiguous slices:
            // dot = row_values·vector (a loop-carried reduction → SIMD `dot_slice`,
            // the Cholesky-class win, SSOT with `householder::apply_right`), then
            // row_values −= (β·dot)·vector (axpy → `axpy_slice`). The slices are
            // length `dim`, long enough that the SIMD dispatch pays — unlike QR's
            // short within-panel slices, which regressed.
            let dot = T::dot_slice(row_values, vector);
            let scale = betas[reflector].mul(dot);
            T::axpy_slice(T::ZERO.sub(scale), vector, row_values); // row_values −= scale·vector
        }
    }
}
