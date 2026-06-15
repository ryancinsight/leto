//! Matrix exponential `e^A` via scaling-and-squaring with a diagonal Padé
//! approximant.

use super::dense::{add, identity, inf_norm, mul, scale};
use crate::domain::real::RealScalar;
use leto::{Array2, ArrayView2, LetoError, Result, Storage};

/// Degree of the diagonal Padé approximant.
const PADE_Q: usize = 6;
/// Scaling threshold: halve `A` until `‖A/2ˢ‖_∞ ≤ 1/2`, where the Padé-6
/// approximant is accurate to well below `f64` rounding.
const SCALE_THRESHOLD: f64 = 0.5;
/// Hard cap on the scaling exponent (guards against non-finite norms slipping
/// past the finiteness check; `2¹⁰²³` already overflows `f64`).
const MAX_SCALE: u32 = 1023;

/// Matrix exponential `e^A = Σ_{k≥0} Aᵏ / k!` of a square matrix.
///
/// # Numerical contract
/// `e^A = (e^{A/2ˢ})^{2ˢ}` because `A` commutes with itself, so
/// `e^{X+Y} = e^X e^Y` holds for the scaled copies and the identity follows by
/// `s`-fold squaring. The inner `e^{A/2ˢ}` is approximated by the diagonal
/// `(q,q)` Padé rational `r_q(B) = D_q(B)⁻¹ N_q(B)` with
/// `N_q(B) = Σ_{k=0}^q c_k Bᵏ`, `D_q(B) = Σ_{k=0}^q (−1)ᵏ c_k Bᵏ`, and
/// `c_k = \frac{(2q−k)!\,q!}{(2q)!\,k!\,(q−k)!}`. The implementation follows
/// the scaling-and-squaring Padé construction described by Higham, *Functions
/// of Matrices*, and selects `s` so `‖B‖_∞ = ‖A/2ˢ‖_∞ ≤ 1/2`.
///
/// Evidence tier: analytical identity for scaling/squaring plus value-semantic
/// closed-form and nalgebra differential tests. This is not a machine-checked
/// proof of the Padé truncation bound.
///
/// Reuses the caller-owned [`matmul`](crate::matmul) and the partial-pivot LU
/// inverse (SSOT): the only matrix operations are products and one inverse of
/// the well-conditioned denominator.
///
/// # Errors
/// [`LetoError::ShapeMismatch`] for non-square input; [`LetoError::StorageError`]
/// for non-finite entries, scaling overflow, or a singular Padé denominator.
pub fn matexp<T: RealScalar>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>> {
    let [rows, cols] = matrix.shape();
    if rows != cols {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![rows, cols],
            rhs: vec![rows, rows],
        });
    }
    let a = matrix.to_contiguous();
    if rows == 0 {
        return Ok(a);
    }
    if !a.storage().as_slice().iter().all(|x| x.is_finite()) {
        return Err(LetoError::StorageError {
            reason: "matrix exponential requires finite entries".to_string(),
        });
    }

    // 1. Scaling: choose s with ‖A / 2ˢ‖_∞ ≤ 1/2.
    let half = T::from_f64(SCALE_THRESHOLD);
    let two = T::from_f64(2.0);
    let mut scaled_norm = inf_norm(&a);
    let mut s: u32 = 0;
    while scaled_norm > half {
        scaled_norm = scaled_norm.div(two);
        s += 1;
        if s > MAX_SCALE {
            return Err(LetoError::StorageError {
                reason: "matrix exponential scaling exponent overflow".to_string(),
            });
        }
    }
    let b = scale(&a, T::from_f64(2f64.powi(-(s as i32)))); // B = A / 2ˢ

    // 2. Diagonal Padé(q, q) approximant of e^B.
    let coeffs = pade_coefficients::<T>();
    let mut b_power = identity::<T>(rows); // B⁰ = I
    let mut numerator = scale(&b_power, coeffs[0]);
    let mut denominator = scale(&b_power, coeffs[0]);
    for (k, &c_k) in coeffs.iter().enumerate().skip(1) {
        b_power = mul(&b_power, &b)?; // Bᵏ
        numerator = add(&numerator, &scale(&b_power, c_k));
        // (−1)ᵏ c_k on the denominator.
        let signed = if k % 2 == 0 { c_k } else { c_k.neg() };
        denominator = add(&denominator, &scale(&b_power, signed));
    }
    let inv_denominator = crate::application::linalg::lu::inv(&denominator.view())?;
    let mut result = mul(&inv_denominator, &numerator)?; // r_q(B) ≈ e^B

    // 3. Squaring: e^A = (e^B)^{2ˢ}.
    for _ in 0..s {
        result = mul(&result, &result)?;
    }
    Ok(result)
}

/// Diagonal Padé coefficients `c_k`, `k = 0..=q`, via the exact ratio recurrence
/// `c_k = c_{k-1} · (q − k + 1) / (k · (2q − k + 1))`, evaluated in `f64` and
/// converted to `T` (construction-time constants, per the `RealScalar::from_f64`
/// contract — not a compute-path widen-narrow).
fn pade_coefficients<T: RealScalar>() -> [T; PADE_Q + 1] {
    let q = PADE_Q as f64;
    let mut c = [0.0f64; PADE_Q + 1];
    c[0] = 1.0;
    for k in 1..=PADE_Q {
        let kf = k as f64;
        c[k] = c[k - 1] * (q - kf + 1.0) / (kf * (2.0 * q - kf + 1.0));
    }
    let mut out = [T::ZERO; PADE_Q + 1];
    for (slot, &value) in out.iter_mut().zip(c.iter()) {
        *slot = T::from_f64(value);
    }
    out
}
