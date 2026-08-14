//! Complex linear algebra: Gaussian elimination for `Array2<Complex<f64>>`.
//!
//! These are the generic complex counterparts of the real LU-based routines in
//! `lu.rs`.  They complement [`leto_ops`] for beamforming, acoustics, and any
//! application that requires complex-valued linear system solves.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use eunomia::Complex;
use leto::{Array1, Array2, LetoError, Result};

type C64 = Complex<f64>;

/// Solve the complex linear system `A·x = b` via Gaussian elimination with
/// partial pivoting.
///
/// # Errors
/// - [`LetoError::InvalidInput`] if `A` is not square or dimensions are inconsistent.
/// - [`LetoError::NumericalBreakdown`] if `A` is singular (zero pivot).
pub fn complex_solve(a: &Array2<C64>, b: &Array1<C64>) -> Result<Array1<C64>> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(LetoError::InvalidInput(format!(
            "complex_solve: A must be square, got {}×{}",
            a.shape()[0],
            a.shape()[1]
        )));
    }
    if b.shape()[0] != n {
        return Err(LetoError::InvalidInput(format!(
            "complex_solve: b length {} != matrix size {n}",
            b.shape()[0]
        )));
    }

    let mut mat: Vec<C64> = a
        .as_slice()
        .ok_or_else(|| LetoError::InvalidInput("complex_solve: A must be contiguous".into()))?
        .to_vec();
    let mut rhs: Vec<C64> = b
        .as_slice()
        .ok_or_else(|| LetoError::InvalidInput("complex_solve: b must be contiguous".into()))?
        .to_vec();

    // Forward elimination with partial pivoting.
    for i in 0..n {
        let mut max_row = i;
        let mut max_val = mat[i * n + i].norm_sqr();
        for k in i + 1..n {
            let v = mat[k * n + i].norm_sqr();
            if v > max_val {
                max_val = v;
                max_row = k;
            }
        }
        if max_row != i {
            for j in 0..n {
                mat.swap(i * n + j, max_row * n + j);
            }
            rhs.swap(i, max_row);
        }
        let pivot = mat[i * n + i];
        if pivot.norm_sqr() < 1e-24 {
            return Err(LetoError::NumericalBreakdown(format!(
                "complex_solve: zero pivot at position {i}"
            )));
        }
        for k in i + 1..n {
            let fac = mat[k * n + i] / pivot;
            for j in i..n {
                let v = mat[k * n + j];
                mat[k * n + j] = v - fac * mat[i * n + j];
            }
            rhs[k] = rhs[k] - fac * rhs[i];
        }
    }

    // Back substitution.
    let mut x = vec![C64::new(0.0, 0.0); n];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for j in i + 1..n {
            s -= mat[i * n + j] * x[j];
        }
        x[i] = s / mat[i * n + i];
    }

    Array1::from_vec([n], x)
        .map_err(|e| LetoError::InvalidInput(format!("complex_solve: output reshape failed: {e}")))
}

/// Compute the inverse of a square complex matrix.
///
/// Solves for each column of the identity matrix via [`complex_solve`].
///
/// # Errors
/// Same as [`complex_solve`].
pub fn complex_inv(a: &Array2<C64>) -> Result<Array2<C64>> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(LetoError::InvalidInput(format!(
            "complex_inv: A must be square, got {}×{}",
            a.shape()[0],
            a.shape()[1]
        )));
    }

    let zero = C64::new(0.0, 0.0);
    let one = C64::new(1.0, 0.0);
    let mut result = vec![zero; n * n];

    for col in 0..n {
        let mut e = vec![zero; n];
        e[col] = one;
        let e_arr = Array1::from_vec([n], e).map_err(|e| LetoError::InvalidInput(e.to_string()))?;
        let x = complex_solve(a, &e_arr)?;
        let xs = x.as_slice().ok_or_else(|| {
            LetoError::InvalidInput("complex_inv: solve output not contiguous".into())
        })?;
        for row in 0..n {
            result[row * n + col] = xs[row];
        }
    }

    Array2::from_vec([n, n], result)
        .map_err(|e| LetoError::InvalidInput(format!("complex_inv: reshape failed: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_2x2() {
        // A = [[2+i, 1], [1, 2-i]], b = [3+i, 2-i]
        let a = Array2::from_vec(
            [2, 2],
            vec![
                C64::new(2.0, 1.0),
                C64::new(1.0, 0.0),
                C64::new(1.0, 0.0),
                C64::new(2.0, -1.0),
            ],
        )
        .unwrap();
        let b = Array1::from_vec([2], vec![C64::new(3.0, 1.0), C64::new(2.0, -1.0)]).unwrap();
        let x = complex_solve(&a, &b).expect("solve");
        // Verify A·x ≈ b
        for i in 0..2 {
            let mut ax = C64::new(0.0, 0.0);
            for j in 0..2 {
                ax += a[[i, j]] * x[j];
            }
            assert!(
                (ax - b[i]).norm() < 1e-12,
                "row {i}: Ax = {ax}, b = {}",
                b[i]
            );
        }
    }

    #[test]
    fn identity_inverse() {
        let eye = Array2::from_vec(
            [2, 2],
            vec![
                C64::new(1.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(1.0, 0.0),
            ],
        )
        .unwrap();
        let inv = complex_inv(&eye).expect("inverse of identity");
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[[i, j]].re - expected).abs() < 1e-12);
                assert!(inv[[i, j]].im.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn singular_returns_err() {
        let sing = Array2::from_vec(
            [2, 2],
            vec![
                C64::new(1.0, 0.0),
                C64::new(2.0, 0.0),
                C64::new(2.0, 0.0),
                C64::new(4.0, 0.0),
            ],
        )
        .unwrap();
        let b = Array1::from_vec([2], vec![C64::new(1.0, 0.0), C64::new(2.0, 0.0)]).unwrap();
        assert!(complex_solve(&sing, &b).is_err());
    }
}
