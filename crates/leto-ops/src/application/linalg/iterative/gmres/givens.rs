//! Givens rotation helpers for GMRES.

use eunomia::{NumericElement, RealField};
use leto::{Array1, Array2, LetoError, Result};

/// Apply all previously accumulated Givens rotations to the new column `k` of H.
pub fn apply_previous_rotations<T: RealField + Copy>(
    h: &mut Array2<T>,
    c: &Array1<T>,
    s: &Array1<T>,
    k: usize,
) {
    for i in 0..k {
        let hi = h[[i, k]];
        let hi1 = h[[i + 1, k]];
        h[[i, k]] = c[i] * hi + s[i] * hi1;
        h[[i + 1, k]] = -s[i] * hi + c[i] * hi1;
    }
}

/// Compute the Givens rotation coefficients `(c, s)` that zero `h[k+1, k]`.
pub fn compute_rotation<T: RealField + Copy>(h_kk: T, h_kp1_k: T) -> (T, T) {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    if h_kp1_k == zero {
        (one, zero)
    } else if h_kk.abs() > h_kp1_k.abs() {
        let t = h_kp1_k / h_kk;
        let c = one / (one + t * t).sqrt();
        (c, c * t)
    } else {
        let t = h_kk / h_kp1_k;
        let s = one / (one + t * t).sqrt();
        (s * t, s)
    }
}

/// Apply the new Givens rotation to column `k` of H and the RHS vector `g`.
pub fn apply_new_rotation<T: RealField + Copy>(
    h: &mut Array2<T>,
    g: &mut Array1<T>,
    c: T,
    s: T,
    k: usize,
) {
    let hkk = h[[k, k]];
    let hkp1k = h[[k + 1, k]];
    h[[k, k]] = c * hkk + s * hkp1k;
    h[[k + 1, k]] = <T as NumericElement>::ZERO;
    let gk = g[k];
    let gk1 = g[k + 1];
    g[k] = c * gk + s * gk1;
    g[k + 1] = -s * gk + c * gk1;
}

/// Solve the upper triangular system H[0..k, 0..k] · y = g[0..k].
///
/// # Errors
/// Returns [`LetoError::NumericalBreakdown`] when a diagonal entry is ≤ ε.
pub fn solve_upper_triangular<T: RealField + Copy>(
    h: &Array2<T>,
    g: &Array1<T>,
    k: usize,
) -> Result<Array1<T>> {
    let mut y = Array1::from_elem([k], <T as NumericElement>::ZERO);
    for i in (0..k).rev() {
        let d = h[[i, i]];
        if d.abs() <= <T as RealField>::EPSILON {
            return Err(LetoError::NumericalBreakdown(format!(
                "GMRES: zero diagonal in triangular solve at position {i}"
            )));
        }
        let mut sum = g[i];
        for j in i + 1..k {
            sum -= h[[i, j]] * y[j];
        }
        y[i] = sum / d;
    }
    Ok(y)
}
