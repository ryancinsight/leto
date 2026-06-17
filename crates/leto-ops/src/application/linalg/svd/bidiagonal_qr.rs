//! Singular values via the implicit-shift bidiagonal QR algorithm
//! (Golub–Kahan / Golub–Reinsch), the accuracy-preserving alternative to the
//! Gram-matrix path.
//!
//! # Theorem (singular values of `A` = of its bidiagonal factor)
//! Golub–Kahan bidiagonalization gives orthogonal `U, V` with `A = U B Vᵀ` and
//! `B` upper bidiagonal. Orthogonal factors preserve singular values
//! (`σ(A) = σ(B)`), so it suffices to compute `σ(B)`. The implicit-shift QR
//! iteration applies, per sweep, a sequence of Givens rotations equivalent to one
//! shifted QR step of `BᵀB` **without ever forming `BᵀB`** (the "Golub–Kahan SVD
//! step"); the off-diagonal of the implicit `BᵀB` is driven to zero, so a
//! superdiagonal of `B` deflates and `B → diag(σ)`. Avoiding `BᵀB` keeps the
//! conditioning at `κ(A)` rather than `κ(A)² = κ(AᵀA)`, so small singular values
//! retain accuracy the Gram path loses. ∎
//!
//! This module provides both the singular **values** (no `U`/`V` accumulation —
//! a zero-cost const-generic specialization) and the full thin SVD with `U`/`V`
//! (the rotations are accumulated into the bidiagonalization's orthogonal
//! factors). It is the default `svd_decompose`; the rank-revealing one-sided
//! Jacobi (`super::jacobi`) remains for rank-deficient / maximal-accuracy needs.

use super::{default_tolerance, validate_input, SvdDecomposition};
use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, Result, Storage};

/// Iteration cap before declaring non-convergence (a safety bound; shifted QR
/// converges in `O(n)` sweeps).
const MAX_ITER: usize = 4000;

/// Singular values of a finite matrix, sorted descending, via bidiagonal QR.
///
/// # Errors
/// [`LetoError`](leto::LetoError) on empty / non-finite input, or QR
/// non-convergence.
pub fn singular_values<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Vec<T>> {
    let tolerance = default_tolerance::<T>();
    validate_input(matrix, tolerance)?;
    let [rows, cols] = matrix.shape();

    // Bidiagonalization requires `m >= n`; σ(A) = σ(Aᵀ), so transpose wide input.
    // Uses the column-major working buffer (left reflector contiguous; local to
    // this path, no global layout change) and returns `(d, e)` directly.
    let (mut d, mut e) = if rows >= cols {
        crate::application::linalg::bidiagonal::bidiagonal_diag_colmajor(matrix)?
    } else {
        let transposed = transpose_to_owned(matrix)?;
        crate::application::linalg::bidiagonal::bidiagonal_diag_colmajor(&transposed.view())?
    };
    let k = rows.min(cols);

    let mut no_u: [T; 0] = [];
    let mut no_v: [T; 0] = [];
    qr_iterate::<T, false>(&mut d, &mut e, k, &mut no_u, 0, &mut no_v, 0)?;

    let mut sigmas: Vec<T> = d.into_iter().map(|x| x.abs()).collect();
    sigmas.sort_by(|a, b| {
        b.partial_cmp(a)
            .expect("singular values are finite (not NaN)")
    });
    Ok(sigmas)
}

/// Materialize `Aᵀ` (wide → tall) so the `m ≥ n` bidiagonalization applies.
fn transpose_to_owned<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>> {
    let [rows, cols] = matrix.shape();
    let mut values = vec![T::ZERO; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            values[j * rows + i] = *matrix.get([i, j])?;
        }
    }
    Array2::from_shape_vec([cols, rows], values)
}

/// Givens rotation `(c, s)` with `c·a + s·b = r`, `−s·a + c·b = 0`.
#[inline]
fn givens<T: RealScalar>(a: T, b: T) -> (T, T) {
    if b == T::ZERO {
        return (T::ONE, T::ZERO);
    }
    // One reciprocal + two mults rather than two divides by the same `r`
    // (division is several× a multiply; the sweep calls this O(n²) times). The
    // extra rounding of `1/r` is within the SVD's differential tolerance.
    let inv_r = T::ONE.div(a.mul(a).add(b.mul(b)).sqrt());
    (a.mul(inv_r), b.mul(inv_r))
}

/// Wilkinson shift: the eigenvalue of the trailing 2×2 of `T = BᵀB` (rows
/// `q-1, q`) nearest the corner `T[q,q]`.
fn wilkinson_shift<T: RealScalar>(d: &[T], e: &[T], p: usize, q: usize) -> T {
    let dq = d[q];
    let dq1 = d[q - 1];
    let eq1 = e[q - 1];
    let eq2 = if q >= p + 2 { e[q - 2] } else { T::ZERO };
    let t11 = dq1.mul(dq1).add(eq2.mul(eq2));
    let t22 = dq.mul(dq).add(eq1.mul(eq1));
    let t12 = dq1.mul(eq1);

    let half = T::from_f64(0.5);
    // δ = (t11 − t22)/2; μ = t22 − sign(δ)·t12² / (|δ| + √(δ²+t12²)) — the
    // cancellation-avoiding form of "eigenvalue of the 2×2 nearest t22".
    let delta = t11.sub(t22).mul(half);
    let denom = delta.abs().add(delta.mul(delta).add(t12.mul(t12)).sqrt());
    if denom == T::ZERO {
        return t22;
    }
    let sign = if delta < T::ZERO {
        T::ONE.neg()
    } else {
        T::ONE
    };
    t22.sub(sign.mul(t12.mul(t12)).div(denom))
}

/// Transpose a row-major `n × n` matrix into a fresh buffer.
fn transpose_square<T: RealScalar>(src: &[T], n: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = src[i * n + j];
        }
    }
    out
}

/// Apply the column Givens `col' = c·col + s·col₊₁`, `col₊₁' = c·col₊₁ − s·col`
/// to a matrix stored **transposed**, where the two columns are the two
/// **contiguous rows** `(row, row+1)` of length `len`.
///
/// # Theorem (transposed accumulation is bitwise-identical, and contiguous)
/// Accumulating `U` (or `V`) as its transpose `Uᵀ` while rotating two **rows** of
/// `Uᵀ` produces exactly the same factor `U` as rotating two **columns** of `U`,
/// bit for bit, and turns the strided column update into a contiguous one.
///
/// *Proof.* A plane rotation applied to columns `(k, k+1)` of `U` is the
/// right-multiplication `U ← U G` with `G` the embedded `2×2` Givens. Transposing,
/// `(U G)ᵀ = Gᵀ Uᵀ`, i.e. `Uᵀ ← Gᵀ Uᵀ`, which is a left-multiplication mixing
/// **rows** `(k, k+1)` of `Uᵀ` with the same scalar coefficients `(c, s)`. The two
/// updated entries in each position are the identical floating-point expression
/// `c·a + s·b` / `c·b − s·a` of the identical operands `a, b`, evaluated in the
/// identical order — so no rounding differs: the stored `Uᵀ` is the exact transpose
/// of the column-accumulated `U`. In row-major storage a row of `Uᵀ` is a
/// contiguous length-`len` slice, so the rotation is two contiguous disjoint
/// slices (cache-friendly, auto-vectorizable) instead of a stride-`len` column walk
/// (a cache line per element). The factors are recovered by reading rows of
/// `Uᵀ`/`Vᵀ` as columns of `U`/`V` at the `O(n²)` thin-extraction step — no
/// separate transpose pass — negligible against the `O(n³)` sweep. ∎
///
/// `U`/`V` are accumulated as `Uᵀ`/`Vᵀ` so every rotation hits this path.
#[inline]
fn rotate_rows<T: RealScalar>(mat: &mut [T], len: usize, row: usize, c: T, s: T) {
    let (head, tail) = mat.split_at_mut((row + 1) * len);
    let row_k = &mut head[row * len..row * len + len];
    let row_k1 = &mut tail[..len];
    for (a, b) in row_k.iter_mut().zip(row_k1.iter_mut()) {
        let (va, vb) = (*a, *b);
        *a = c.mul(va).add(s.mul(vb));
        *b = c.mul(vb).sub(s.mul(va));
    }
}

/// Drive the bidiagonal `(d, e)` to diagonal form (all superdiagonals deflated).
///
/// When `VEC`, the left/right Givens rotations are accumulated into `u`
/// (`m × m`) and `v` (`n × n`); otherwise those updates are DCE'd
/// (`u`/`v` may be empty) — a zero-cost specialization for the values-only path.
fn qr_iterate<T: RealScalar, const VEC: bool>(
    d: &mut [T],
    e: &mut [T],
    k: usize,
    u: &mut [T],
    m: usize,
    v: &mut [T],
    n: usize,
) -> Result<()> {
    if k <= 1 {
        return Ok(());
    }
    let mut q = k - 1;
    let mut iter = 0usize;
    loop {
        // Peel converged singular values off the bottom, deflating the bottom
        // superdiagonal in passing (precision-exact `s + |e| == s`). Only the
        // active region near the bottom is touched — already-converged blocks
        // above are not re-scanned each iteration (matches LAPACK/nalgebra
        // `delimit_subproblem`; the prior `0..q` rescan was `O(q)` per step).
        while q > 0 {
            let scale = d[q - 1].abs().add(d[q].abs());
            if scale.add(e[q - 1].abs()) == scale {
                e[q - 1] = T::ZERO;
            }
            if e[q - 1] != T::ZERO {
                break;
            }
            q -= 1;
        }
        if q == 0 {
            return Ok(());
        }
        // Top of the bottom-most unreduced block: scan up until a negligible
        // superdiagonal splits it (deflate that entry as the block boundary).
        let mut p = q;
        while p > 0 {
            let scale = d[p - 1].abs().add(d[p].abs());
            if scale.add(e[p - 1].abs()) == scale {
                e[p - 1] = T::ZERO;
                break;
            }
            p -= 1;
        }

        iter += 1;
        if iter > MAX_ITER {
            return Err(leto::LetoError::StorageError {
                reason: "bidiagonal SVD QR failed to converge".to_string(),
            });
        }
        qr_step::<T, VEC>(d, e, p, q, u, m, v, n);
    }
}

/// One implicit-shift Golub–Kahan SVD step on the block `d[p..=q]`, `e[p..q]`.
#[allow(clippy::too_many_arguments)]
fn qr_step<T: RealScalar, const VEC: bool>(
    d: &mut [T],
    e: &mut [T],
    p: usize,
    q: usize,
    u: &mut [T],
    m: usize,
    v: &mut [T],
    n: usize,
) {
    let mu = wilkinson_shift(d, e, p, q);
    // First column of (BᵀB − μI).
    let mut y = d[p].mul(d[p]).sub(mu);
    let mut z = d[p].mul(e[p]);

    for k in p..q {
        // Right rotation (mixes columns k, k+1) annihilating z → accumulate V.
        let (c, s) = givens(y, z);
        if VEC {
            rotate_rows(v, n, k, c, s); // V accumulated transposed
        }
        if k > p {
            e[k - 1] = c.mul(y).add(s.mul(z));
        }
        let mut f = c.mul(d[k]).add(s.mul(e[k]));
        e[k] = c.mul(e[k]).sub(s.mul(d[k]));
        let bulge_col = s.mul(d[k + 1]);
        d[k + 1] = c.mul(d[k + 1]);
        d[k] = f;

        // Left rotation (mixes rows k, k+1) annihilating the bulge → accumulate U.
        let (c, s) = givens(d[k], bulge_col);
        if VEC {
            rotate_rows(u, m, k, c, s); // U accumulated transposed
        }
        d[k] = c.mul(d[k]).add(s.mul(bulge_col));
        f = c.mul(e[k]).add(s.mul(d[k + 1]));
        d[k + 1] = c.mul(d[k + 1]).sub(s.mul(e[k]));
        e[k] = f;
        if k + 1 < q {
            let bulge_row = s.mul(e[k + 1]);
            e[k + 1] = c.mul(e[k + 1]);
            y = e[k];
            z = bulge_row;
        }
    }
}

/// Thin SVD `A = U Σ Vᵀ` for a **tall-or-square** matrix (`m ≥ n`) via the
/// implicit-shift bidiagonal QR with `U`/`V` accumulation.
///
/// Returns `(U, σ, V)` with `U` (`m × n`) and `V` (`n × n`) having orthonormal
/// columns and `σ` sorted descending (length `n`). Singular values are
/// non-negative (negative pivots are absorbed by flipping the matching `U`
/// column). `m ≥ n` is a precondition (the caller transposes wide input).
fn svd_tall<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<(Array2<T>, Vec<T>, Array2<T>)> {
    let [m, n] = matrix.shape();
    debug_assert!(m >= n, "svd_tall requires m >= n");
    let bidiag = crate::bidiagonalize(matrix)?;

    // Bidiagonalization factors `A = U_b B V_bᵀ`; accumulate the QR rotations on
    // top of them. To make every rotation a contiguous two-row update we hold the
    // factors **transposed**: `ut = U_bᵀ` (`m × m`), `vt = V_bᵀ` (`n × n`), so
    // after the sweep `ut = Uᵀ`, `vt = Vᵀ` (see `rotate_rows`). Column `j` of
    // `U`/`V` is therefore row `j` of `ut`/`vt`.
    let mut ut = transpose_square(bidiag.u().storage().as_slice(), m);
    let mut vt = transpose_square(bidiag.v().storage().as_slice(), n);
    let b = bidiag.b();
    let mut d = vec![T::ZERO; n];
    let mut e = vec![T::ZERO; n];
    for i in 0..n {
        d[i] = *b.get([i, i])?;
        if i + 1 < n {
            e[i] = *b.get([i, i + 1])?;
        }
    }

    qr_iterate::<T, true>(&mut d, &mut e, n, &mut ut, m, &mut vt, n)?;

    // Force σ ≥ 0: a negative pivot flips the sign of its left singular vector
    // (column `i` of `U` = row `i` of `ut`, contiguous).
    for i in 0..n {
        if d[i] < T::ZERO {
            d[i] = d[i].neg();
            for slot in &mut ut[i * m..i * m + m] {
                *slot = slot.neg();
            }
        }
    }

    // Descending sort of the singular values, carrying the U/V columns. Column
    // `old` of `U`/`V` is row `old` of `ut`/`vt` (a contiguous slice).
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&a, &b| d[b].partial_cmp(&d[a]).expect("singular values are finite"));

    let mut sigma = vec![T::ZERO; n];
    let mut u_thin = vec![T::ZERO; m * n]; // m × n
    let mut v_thin = vec![T::ZERO; n * n]; // n × n
    for (new_col, &old) in perm.iter().enumerate() {
        sigma[new_col] = d[old];
        for r in 0..m {
            u_thin[r * n + new_col] = ut[old * m + r];
        }
        for r in 0..n {
            v_thin[r * n + new_col] = vt[old * n + r];
        }
    }

    Ok((
        Array2::from_shape_vec([m, n], u_thin).expect("U shape matches storage"),
        sigma,
        Array2::from_shape_vec([n, n], v_thin).expect("V shape matches storage"),
    ))
}

/// Thin SVD for a **full-rank** matrix (default tolerance); rejects
/// rank-deficient input. Bidiagonal-QR backed (supersedes the former Gram path).
///
/// # Errors
/// [`LetoError`](leto::LetoError) on empty / non-finite / rank-deficient input.
pub fn svd_decompose<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<SvdDecomposition<T>> {
    svd_decompose_with_tolerance(matrix, default_tolerance::<T>())
}

/// Thin SVD for a full-rank matrix with an explicit rank tolerance.
///
/// # Errors
/// [`LetoError`](leto::LetoError) on empty / non-finite / rank-deficient input,
/// or invalid tolerance.
pub fn svd_decompose_with_tolerance<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
    tolerance: T,
) -> Result<SvdDecomposition<T>> {
    validate_input(matrix, tolerance)?;
    let svd = svd_via_bidiagonal(matrix)?;
    if let Some(&sigma_min) = svd.singular_values.last() {
        if sigma_min <= tolerance {
            return Err(leto::LetoError::StorageError {
                reason: "SVD input is rank-deficient".to_string(),
            });
        }
    }
    Ok(svd)
}

/// Thin SVD `A = U Σ Vᵀ` via implicit-shift bidiagonal QR (Golub–Reinsch).
///
/// Wide input (`m < n`) is handled by `σ(A) = σ(Aᵀ)` with `U(A) = V(Aᵀ)`,
/// `V(A) = U(Aᵀ)`: the SVD of the tall `Aᵀ` is computed and its factors swapped.
/// Faster and more accurate than the Gram path (conditioning `κ(A)`, not
/// `κ(A)²`); handles rank-deficient input (zero singular values emerge).
///
/// # Errors
/// [`LetoError`](leto::LetoError) on empty / non-finite input or QR
/// non-convergence.
pub fn svd_via_bidiagonal<T: RealScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<SvdDecomposition<T>> {
    validate_input(matrix, default_tolerance::<T>())?;
    let [m, n] = matrix.shape();
    if m >= n {
        let (u, sigma, v) = svd_tall(matrix)?;
        Ok(SvdDecomposition {
            singular_values: sigma,
            left_singular_vectors: u,
            right_singular_vectors: v,
        })
    } else {
        // Compute the SVD of the tall Aᵀ and swap U ↔ V.
        let transposed = transpose_to_owned(matrix)?;
        let (u_t, sigma, v_t) = svd_tall(&transposed.view())?;
        Ok(SvdDecomposition {
            singular_values: sigma,
            left_singular_vectors: v_t,
            right_singular_vectors: u_t,
        })
    }
}
