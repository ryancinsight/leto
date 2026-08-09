//! Rank-revealing SVD by one-sided Jacobi column orthogonalization.

use super::{SvdDecomposition, default_tolerance, validate_input};
use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, Result};

/// Safety cap on Jacobi sweeps; convergence is quadratic near the solution, so
/// well under this many sweeps are used in practice. The cap only guarantees
/// termination.
const MAX_SWEEPS: usize = 60;

/// Rank-revealing SVD `A = U Σ Vᵀ` via one-sided Jacobi (default tolerance).
///
/// Unlike [`svd_decompose`](super::svd_decompose), this **accepts rank-deficient
/// input**: zero singular values are surfaced honestly (their `U` columns are
/// left zero — those directions lie in the left null space and are not
/// materialized), while `V` is always fully orthonormal.
///
/// # Theorem (one-sided Jacobi converges to the SVD)
/// Orthogonalizing the columns of `A` by a product of plane rotations `V`
/// produces `AV = W` with mutually orthogonal columns; then `σᵢ = ‖wᵢ‖`,
/// `uᵢ = wᵢ/σᵢ` (for `σᵢ>0`), and `A = W Vᵀ = U Σ Vᵀ`.
/// *Convergence:* let `off(M) = Σ_{i≠j} (mᵢᵀmⱼ)²` measure column
/// non-orthogonality of `M`. A Jacobi rotation on columns `(p,q)` chosen to make
/// `wₚᵀwq = 0` zeroes that pair's contribution and leaves all other inner
/// products that don't involve `p` or `q` unchanged; a standard computation
/// shows `off` is **non-increasing** each rotation and strictly decreases while
/// any off-orthogonal pair remains. Since `off ≥ 0` is bounded below, the
/// sweeps converge to `off = 0`, i.e. exact column orthogonality — the SVD. ∎
///
/// # Errors
/// [`LetoError`](leto::LetoError) on empty or non-finite input.
pub fn svd_rank_revealing<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<SvdDecomposition<T>> {
    svd_rank_revealing_with_tolerance(matrix, default_tolerance::<T>())
}

/// Rank-revealing one-sided Jacobi SVD with an explicit relative orthogonality
/// tolerance. See [`svd_rank_revealing`] for the contract and proof.
///
/// # Errors
/// [`LetoError`](leto::LetoError) on empty or non-finite input.
pub fn svd_rank_revealing_with_tolerance<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    tolerance: T,
) -> Result<SvdDecomposition<T>> {
    validate_input(matrix, tolerance)?;
    let [rows, cols] = matrix.shape();
    // One-sided Jacobi orthogonalizes columns, so it needs at least as many rows
    // as columns. For wide inputs decompose Aᵀ and swap U ↔ V (`A = (Aᵀ)ᵀ`).
    one_sided_jacobi(matrix, rows >= cols, tolerance)
}

/// Core kernel. `tall` selects whether the working matrix is `A` (`m ≥ n`) or
/// `Aᵀ`; the result `U/V` are swapped accordingly so the returned
/// [`SvdDecomposition`] always describes the *original* `A`.
fn one_sided_jacobi<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    tall: bool,
    tolerance: T,
) -> Result<SvdDecomposition<T>> {
    let [rows, cols] = matrix.shape();
    // Working matrix `w` has shape wm × wn with wm ≥ wn.
    let (wm, wn) = if tall { (rows, cols) } else { (cols, rows) };

    // Row-major working copy of the matrix being orthogonalized (A or Aᵀ).
    let mut w = vec![T::ZERO; wm * wn];
    if let Some(slice) = matrix.as_slice() {
        if tall {
            w.copy_from_slice(&slice[..wm * wn]);
        } else {
            for i in 0..wm {
                for j in 0..wn {
                    w[i * wn + j] = slice[j * wm + i];
                }
            }
        }
    } else {
        for i in 0..wm {
            for j in 0..wn {
                w[i * wn + j] = if tall {
                    *matrix.get([i, j])?
                } else {
                    *matrix.get([j, i])?
                };
            }
        }
    }

    // Right factor accumulator, initialized to the identity.
    let mut v = vec![T::ZERO; wn * wn];
    for i in 0..wn {
        v[i * wn + i] = T::ONE;
    }

    for _sweep in 0..MAX_SWEEPS {
        let mut rotated = false;
        for p in 0..wn {
            for q in (p + 1)..wn {
                let (mut alpha, mut beta, mut gamma) = (T::ZERO, T::ZERO, T::ZERO);
                for i in 0..wm {
                    let wp = w[i * wn + p];
                    let wq = w[i * wn + q];
                    alpha = alpha.add(wp.mul(wp));
                    beta = beta.add(wq.mul(wq));
                    gamma = gamma.add(wp.mul(wq));
                }
                // Columns already orthogonal (relative threshold): skip.
                let threshold = tolerance.mul(alpha.mul(beta).sqrt());
                if gamma.abs() <= threshold {
                    continue;
                }

                // Jacobi rotation (c, s) that zeroes the new ⟨wₚ, wq⟩.
                let zeta = beta.sub(alpha).div(gamma.add(gamma));
                let sign = if zeta < T::ZERO {
                    T::ZERO.sub(T::ONE)
                } else {
                    T::ONE
                };
                let t = sign.div(zeta.abs().add(zeta.mul(zeta).add(T::ONE).sqrt()));
                let c = T::ONE.div(t.mul(t).add(T::ONE).sqrt());
                let s = c.mul(t);

                for i in 0..wm {
                    let wp = w[i * wn + p];
                    let wq = w[i * wn + q];
                    w[i * wn + p] = c.mul(wp).sub(s.mul(wq));
                    w[i * wn + q] = s.mul(wp).add(c.mul(wq));
                }
                for i in 0..wn {
                    let vp = v[i * wn + p];
                    let vq = v[i * wn + q];
                    v[i * wn + p] = c.mul(vp).sub(s.mul(vq));
                    v[i * wn + q] = s.mul(vp).add(c.mul(vq));
                }
                rotated = true;
            }
        }
        if !rotated {
            break;
        }
    }

    // Column norms are the singular values; normalized columns are U.
    let mut sigma = vec![T::ZERO; wn];
    for j in 0..wn {
        let mut norm_sq = T::ZERO;
        for i in 0..wm {
            let x = w[i * wn + j];
            norm_sq = norm_sq.add(x.mul(x));
        }
        sigma[j] = norm_sq.sqrt();
    }
    let mut u = vec![T::ZERO; wm * wn];
    for j in 0..wn {
        if sigma[j] > tolerance {
            for i in 0..wm {
                u[i * wn + j] = w[i * wn + j].div(sigma[j]);
            }
        }
        // σⱼ ≈ 0: leave the U column zero (left-null-space direction).
    }

    // Sort triplets by descending singular value.
    let mut order: Vec<usize> = (0..wn).collect();
    order.sort_by(|&a, &b| {
        sigma[b]
            .partial_cmp(&sigma[a])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let sigma_sorted: Vec<T> = order.iter().map(|&j| sigma[j]).collect();
    let u_sorted = permute_columns(&u, wm, wn, &order);
    let v_sorted = permute_columns(&v, wn, wn, &order);

    // Map back to the original A: tall keeps (U, V); wide swaps them.
    let (left, left_rows, right, right_rows) = if tall {
        (u_sorted, wm, v_sorted, wn)
    } else {
        (v_sorted, wn, u_sorted, wm)
    };

    Ok(SvdDecomposition {
        singular_values: sigma_sorted,
        left_singular_vectors: Array2::from_shape_vec([left_rows, wn], left)
            .expect("left singular vector shape matches storage"),
        right_singular_vectors: Array2::from_shape_vec([right_rows, wn], right)
            .expect("right singular vector shape matches storage"),
    })
}

/// Reorder the columns of a row-major `rows × cols` matrix by `order` (new
/// column `k` ← old column `order[k]`).
fn permute_columns<T: RealScalar>(src: &[T], rows: usize, cols: usize, order: &[usize]) -> Vec<T> {
    let mut out = vec![T::ZERO; rows * cols];
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..rows {
            out[row * cols + new_col] = src[row * cols + old_col];
        }
    }
    out
}
