//! Two-sided Householder reduction `A → (U, B, V)` with `B = Uᵀ A V` upper
//! bidiagonal, for `m ≥ n`.

use crate::application::linalg::householder::{apply_left, apply_right, reflector};
use crate::domain::real::RealScalar;
use leto::{ArrayView2, LetoError, Result};

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

    // Reused scratch (allocation-free reduction hot loop): the left-apply accumulator `w`.
    let mut alw: Vec<T> = Vec::with_capacity(n);
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
            if let Some(u) = u.as_mut() {
                apply_right(&refl, u, m, k, 0, m); // U ← U·Lₖ
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
                if let Some(v) = v.as_mut() {
                    apply_right(&refl, v, n, k + 1, 0, n); // V ← V·Rₖ
                }
            }
        }
    }

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
