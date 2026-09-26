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
//! The shifted step above assumes a nonsingular block. Exact rank deficiency
//! breaks that assumption — it puts an exact zero on the diagonal of `B`, where
//! the implicit `BᵀB` is singular and the sweep reaches a fixed point at
//! `d = 0`, `e ≠ 0` instead of deflating. That case is handled separately, by
//! rotating the offending row (or trailing column) out of the block; see
//! `chase_negligible_diagonal_row` and `chase_negligible_diagonal_column`.
//!
//! This module provides both the singular **values** (no `U`/`V` accumulation —
//! a zero-cost const-generic specialization) and the full thin SVD with `U`/`V`
//! (the rotations are accumulated into the bidiagonalization's orthogonal
//! factors). It is the sole SVD implementation, rank-deficient input included.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::{validate_input, SvdDecomposition};
use crate::application::linalg::scaling::{self, GateBound};
use crate::application::linalg::thresholds;
use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, Result, Storage};
use rotation::TransposedFactors;
use sweep::qr_iterate;

mod deflation;
mod rotation;
mod sweep;
#[cfg(test)]
mod tests;
mod zero_shift;

/// The matrix-tier gate's power-of-two bound `2^f` for the SVD family
/// (degree 1): with the kernels' products scale-safe (`givens`,
/// `qr_step`'s shift, the bidiagonal reflectors), every remaining
/// intermediate is a sum of entry-scale terms. The largest is a reflector
/// application's dot product `vᵀ·col` (`bidiagonal/colmajor.rs`,
/// `householder::apply_left`): `|vᵀ·col| ≤ ‖v‖₂·‖A‖_F`, with `‖v‖₂ ≤ 4√M`
/// (`M = max(rows, cols)`; the shared reflector's `v` has entries below 4
/// after its `[1, 2)` normalization, `householder::reflect_in_place`; the
/// column-major one has `|vᵢ| ≤ 1`) and `‖A‖_F ≤ 2^r·‖A‖_max`
/// ([`scaling::norm_ratio_log2`]). The rotation updates and deflation sums
/// stay below `2‖B‖ ≤ 2‖A‖_F` (`|c|, |s| ≤ 1`), inside the same bound.
///
/// The range the gate applies is degree 2, LAPACK `dgesvd`'s
/// `[√safmin/ε, ε/√safmin]` form (`scaling::balanced(matrix, 2, …)`): the
/// bound needs only degree 1, and `(Ω·2^−f)^½ ≤ Ω·2^−f` keeps it, but the
/// sweep needs headroom below `‖A‖_max` for the small singular values it
/// converges on. At the degree-1 lower end `smlnum` they sit within a few
/// binades of `safmin` in the narrow formats, and Bf16 skew-symmetric
/// tridiagonals near `2⁻¹¹⁵` fail to converge even with the zero-shift sweep.
///
/// Deflation floor: [`qr_iterate`] also deflates at or below
/// [`deflation_floor`](deflation::deflation_floor)` = safmin`, at most `k = min(rows, cols)` entries, so
/// the gate's lower end is raised until `√k·safmin ≤ ε·2^l·‖A‖_max ≤ ε·‖A‖_F`
/// ([`thresholds::deflation_count_log2`], [`scaling::norm_ratio_floor_log2`]).
fn svd_bound<T: RealScalar>(rows: usize, cols: usize) -> impl FnOnce(&[T], T) -> GateBound {
    move |values, largest| {
        let half_log2_m = (thresholds::ceil_log2_count(rows.max(cols)) + 1) / 2;
        GateBound {
            factor_log2: 2 + half_log2_m + scaling::norm_ratio_log2(values, largest),
            floor_log2: thresholds::deflation_count_log2(rows.min(cols))
                - scaling::norm_ratio_floor_log2(values, largest),
        }
    }
}

/// Singular values of a finite matrix, sorted descending, via bidiagonal QR.
///
/// # Errors
/// [`LetoError`](leto::LetoError) on empty / non-finite input, or QR
/// non-convergence.
pub fn singular_values<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Vec<T>> {
    validate_input(matrix)?;
    let [rows, cols] = matrix.shape();
    if let Some((scaled, exponent)) = scaling::balanced(matrix, 2, svd_bound(rows, cols))? {
        let mut sigmas = singular_values_of_balanced(&scaled.view())?;
        scaling::restore(
            &mut sigmas,
            exponent,
            "singular value exceeds the scalar range",
        )?;
        return Ok(sigmas);
    }
    singular_values_of_balanced(matrix)
}

/// [`singular_values`] of a validated matrix whose entries are in range.
fn singular_values_of_balanced<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Vec<T>> {
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

    qr_iterate::<T, false>(&mut d, &mut e, k, &mut TransposedFactors::none())?;

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
    if let Some(slice) = matrix.as_slice() {
        for i in 0..rows {
            let row = &slice[i * cols..i * cols + cols];
            for (j, &val) in row.iter().enumerate() {
                values[j * rows + i] = val;
            }
        }
    } else {
        for i in 0..rows {
            for j in 0..cols {
                values[j * rows + i] = *matrix.get([i, j])?;
            }
        }
    }
    Array2::from_shape_vec([cols, rows], values)
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
    let b_slice = b.storage().as_slice();
    let cols = b.shape()[1];
    let mut d = vec![T::ZERO; n];
    let mut e = vec![T::ZERO; n];
    for i in 0..n {
        d[i] = b_slice[i * cols + i];
        if i + 1 < n {
            e[i] = b_slice[i * cols + i + 1];
        }
    }

    qr_iterate::<T, true>(
        &mut d,
        &mut e,
        n,
        &mut TransposedFactors::new(&mut ut, m, &mut vt, n),
    )?;

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

/// Thin SVD `A = U Σ Vᵀ` via implicit-shift bidiagonal QR (Golub–Reinsch).
///
/// Wide input (`m < n`) is handled by `σ(A) = σ(Aᵀ)` with `U(A) = V(Aᵀ)`,
/// `V(A) = U(Aᵀ)`: the SVD of the tall `Aᵀ` is computed and its factors swapped.
///
/// Rank-deficient input is accepted: rank deficiency is data, reported as
/// `σᵢ = 0` in the returned `Σ`, not an error. `U` and `V` keep orthonormal
/// columns at every rank because both are accumulated products of Householder
/// reflectors and Givens rotations, whose orthogonality does not depend on the
/// singular values being nonzero. Callers that require full rank test
/// `singular_values.last()` against their own threshold — the appropriate
/// threshold is the caller's noise floor, which this function cannot know.
///
/// # Errors
/// [`LetoError`](leto::LetoError) on empty / non-finite input or QR
/// non-convergence.
pub fn svd_decompose<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<SvdDecomposition<T>> {
    validate_input(matrix)?;
    let [rows, cols] = matrix.shape();
    if let Some((scaled, exponent)) = scaling::balanced(matrix, 2, svd_bound(rows, cols))? {
        let mut decomposition = svd_of_balanced(&scaled.view())?;
        scaling::restore(
            &mut decomposition.singular_values,
            exponent,
            "singular value exceeds the scalar range",
        )?;
        return Ok(decomposition);
    }
    svd_of_balanced(matrix)
}

/// [`svd_decompose`] of a validated matrix whose entries are in range.
fn svd_of_balanced<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<SvdDecomposition<T>> {
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
