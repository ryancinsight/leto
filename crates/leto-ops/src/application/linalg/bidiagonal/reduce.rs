//! Two-sided Householder reduction `A → (U, B, V)` with `B = Uᵀ A V` upper
//! bidiagonal, for `m ≥ n`.

use crate::application::linalg::householder::{apply_left, apply_right, reflector};
use crate::domain::real::RealScalar;
use leto::{ArrayView2, LetoError, Result, Storage};

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

/// Reduce `A` to upper bidiagonal `B` without accumulating `U`/`V`.
///
/// This is the values-only SVD path: the same Householder transforms are applied
/// to the working matrix, but factor updates are dead-code-eliminated by the
/// const-generic `ACCUMULATE_FACTORS = false` specialization.
pub(super) fn reduce_to_bidiagonal_values<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    m: usize,
    n: usize,
) -> Result<Vec<T>> {
    Ok(reduce_to_bidiagonal_impl::<T, false>(matrix, m, n)?.b)
}

fn reduce_to_bidiagonal_impl<T: RealScalar, const ACCUMULATE_FACTORS: bool>(
    matrix: &ArrayView2<'_, T>,
    m: usize,
    n: usize,
) -> Result<BidiagonalWork<T>> {
    // Working B ← A (m×n) via one bulk row-major copy, validating finiteness in
    // the same pass (SSOT — the caller no longer pre-scans element-by-element).
    // U ← I (m×m), V ← I (n×n) only when requested.
    let contiguous = matrix.to_contiguous();
    let b_src = contiguous.storage().as_slice();
    if !b_src.iter().all(|value| value.is_finite()) {
        return Err(LetoError::StorageError {
            reason: "bidiagonalization input contains a non-finite value".to_string(),
        });
    }
    let mut b = b_src.to_vec();
    let mut u = ACCUMULATE_FACTORS.then(|| identity::<T>(m));
    let mut v = ACCUMULATE_FACTORS.then(|| identity::<T>(n));

    for k in 0..n {
        // Left reflector: column k, rows k..m.
        let col: Vec<T> = (k..m).map(|i| b[i * n + k]).collect();
        if let Some((refl, _alpha)) = reflector(&col) {
            apply_left(&refl, &mut b, n, k, k, n); // rows k..m, cols k..n
            if let Some(u) = u.as_mut() {
                apply_right(&refl, u, m, k, 0, m); // U ← U·Lₖ
            }
        }

        // Right reflector: row k, columns k+1..n.
        if k + 1 < n {
            let row: Vec<T> = (k + 1..n).map(|j| b[k * n + j]).collect();
            if let Some((refl, _alpha)) = reflector(&row) {
                apply_right(&refl, &mut b, n, k + 1, k, m); // cols k+1..n, rows k..m
                if let Some(v) = v.as_mut() {
                    apply_right(&refl, v, n, k + 1, 0, n); // V ← V·Rₖ
                }
            }
        }
    }

    // Present exact upper-bidiagonal form (keep (i,i) and (i,i+1) only).
    for i in 0..m {
        for j in 0..n {
            if j < i || j > i + 1 {
                b[i * n + j] = T::ZERO;
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
