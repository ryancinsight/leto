//! Givens rotation helpers for GMRES.
//!
//! The Hessenberg matrix is stored transposed — row `k` of the backing array
//! holds column `k` of `H̄` — so a rotation sweep, which walks one column,
//! traverses contiguous memory.

use eunomia::{NumericElement, RealField};
use leto::{LetoError, Result};

/// Apply the accumulated rotations `0..k` to the freshly built column `k`.
pub(super) fn apply_previous_rotations<T: RealField + Copy>(
    column: &mut [T],
    cosines: &[T],
    sines: &[T],
    k: usize,
) {
    for i in 0..k {
        let upper = column[i];
        let lower = column[i + 1];
        column[i] = cosines[i] * upper + sines[i] * lower;
        column[i + 1] = -sines[i] * upper + cosines[i] * lower;
    }
}

/// Compute the rotation `(c, s)` that annihilates `lower` against `upper`.
///
/// The tangent form divides the smaller magnitude by the larger, so `|t| ≤ 1`
/// and `√(1 + t²) ∈ [1, √2]`: no intermediate overflow or underflow for any
/// finite input, unlike the direct `√(upper² + lower²)` hypotenuse used by both
/// reference implementations. Finite inputs therefore yield finite output.
///
/// # Errors
/// Returns [`LetoError::NumericalBreakdown`] if either input is non-finite.
pub(super) fn compute_rotation<T: RealField + Copy>(upper: T, lower: T) -> Result<(T, T)> {
    if !upper.is_finite() || !lower.is_finite() {
        return Err(LetoError::NumericalBreakdown(
            "GMRES: non-finite Hessenberg entry in Givens rotation".into(),
        ));
    }
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    if lower == zero {
        return Ok((one, zero));
    }
    if upper.abs() > lower.abs() {
        let tangent = lower / upper;
        let cosine = one / (one + tangent * tangent).sqrt();
        Ok((cosine, cosine * tangent))
    } else {
        let tangent = upper / lower;
        let sine = one / (one + tangent * tangent).sqrt();
        Ok((sine * tangent, sine))
    }
}

/// Apply the new rotation to column `k` of `H̄` and to the transformed
/// right-hand side `g`, so that `|g[k+1]|` is the minimised residual norm over
/// the current Krylov subspace.
pub(super) fn apply_new_rotation<T: RealField + Copy>(
    column: &mut [T],
    g: &mut [T],
    cosine: T,
    sine: T,
    k: usize,
) {
    let upper = column[k];
    let lower = column[k + 1];
    column[k] = cosine * upper + sine * lower;
    column[k + 1] = <T as NumericElement>::ZERO;
    let transformed = g[k];
    g[k] = cosine * transformed;
    g[k + 1] = -sine * transformed;
}

/// Solve `R·y = g` for the leading `k × k` block, writing into `y`.
///
/// `hessenberg` is the transposed store: `H̄(row, column)` lives at
/// `hessenberg[column · stride + row]`.
///
/// # Errors
/// Returns [`LetoError::NumericalBreakdown`] on a vanishing or non-finite
/// diagonal entry, which means the rotations failed to produce a full-rank
/// triangular factor.
pub(super) fn solve_upper_triangular<T: RealField + Copy>(
    hessenberg: &[T],
    stride: usize,
    g: &[T],
    y: &mut [T],
    k: usize,
) -> Result<()> {
    for row in (0..k).rev() {
        let mut sum = g[row];
        for column in row + 1..k {
            sum -= hessenberg[column * stride + row] * y[column];
        }
        let diagonal = hessenberg[row * stride + row];
        if !sum.is_finite() || !diagonal.is_finite() || diagonal == <T as NumericElement>::ZERO {
            return Err(LetoError::NumericalBreakdown(format!(
                "GMRES: singular or non-finite triangular factor at row {row}"
            )));
        }
        y[row] = sum / diagonal;
        if !y[row].is_finite() {
            return Err(LetoError::NumericalBreakdown(format!(
                "GMRES: non-finite least-squares coefficient at row {row}"
            )));
        }
    }
    Ok(())
}
