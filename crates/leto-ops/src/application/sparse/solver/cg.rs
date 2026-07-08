//! Conjugate gradient (CG) solver for symmetric positive-definite matrices.

use super::super::CsrMatrix;
use crate::domain::real::RealScalar;
use leto::LetoError;

/// Result of a CG solve.
#[derive(Debug, Clone, PartialEq)]
pub struct CgResult<T> {
    /// Solution vector (length `nrows`).
    pub x: Vec<T>,
    /// Iterations performed.
    pub iterations: usize,
    /// Final L2 norm of the residual.
    pub residual: T,
    /// `true` if the residual met the tolerance within the iteration limit.
    pub converged: bool,
}

/// Solve `A·x = b` via conjugate gradient.
///
/// `A` must be symmetric positive-definite (SPD). Uses a zero initial guess.
/// Convergence is checked on the relative residual: `‖r‖₂ ≤ tol · ‖b‖₂`.
///
/// # Errors
/// - [`LetoError::ShapeMismatch`] if `b.len() != nrows`.
/// - [`LetoError::ConvergenceError`] if `max_iters` is exceeded.
pub fn cg<T: RealScalar>(
    a: &CsrMatrix<T>,
    b: &[T],
    max_iters: usize,
    tol: T,
) -> Result<CgResult<T>, LetoError> {
    let n = a.nrows();
    if b.len() != n {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![b.len()],
            rhs: vec![n],
        });
    }

    // Trivial zero-rhs case.
    let b_norm_sq = T::dot_slice(b, b);
    if b_norm_sq == T::ZERO {
        return Ok(CgResult {
            x: vec![T::ZERO; n],
            iterations: 0,
            residual: T::ZERO,
            converged: true,
        });
    }

    // Workspace: initial guess x₀ = 0, r₀ = b - A·0 = b, p₀ = r₀.
    let mut x = vec![T::ZERO; n];
    let mut r = b.to_vec();
    let mut p = b.to_vec();

    let tol_sq = tol.mul(tol);
    let b_norm_sq_tol = b_norm_sq.mul(tol_sq);
    let mut rho = T::dot_slice(&r, &r);

    for iter in 0..max_iters {
        // Ap = A·p  (inlined CSR SpMV)
        let mut ap = vec![T::ZERO; n];
        spmv_into(a, &p, &mut ap);

        // α = (r·r) / (p·Ap)
        let p_ap = T::dot_slice(&p, &ap);
        if p_ap == T::ZERO {
            break;
        }
        let alpha = rho / p_ap;

        // x = x + α·p
        for (xi, pi) in x.iter_mut().zip(&p) {
            *xi = xi.add(alpha.mul(*pi));
        }

        // r = r - α·Ap
        for (ri, api) in r.iter_mut().zip(&ap) {
            *ri = ri.sub(alpha.mul(*api));
        }

        let rho_new = T::dot_slice(&r, &r);
        if rho_new < b_norm_sq_tol {
            return Ok(CgResult {
                x,
                iterations: iter + 1,
                residual: T::sqrt(rho_new),
                converged: true,
            });
        }

        // β = (r_new·r_new) / (r·r)
        let beta = rho_new / rho;

        // p = r + β·p
        for (pi, ri) in p.iter_mut().zip(&r) {
            *pi = ri.add(beta.mul(*pi));
        }

        rho = rho_new;
    }

    // Did not converge.
    Err(LetoError::ConvergenceError {
        max_iters,
        residual: rho.sqrt().to_f64(),
        tol: tol.to_f64(),
    })
}

/// CSR SpMV: y = A·x (no ArrayView wrapper — for hot inner loop).
#[inline]
pub(super) fn spmv_into<T: RealScalar>(a: &CsrMatrix<T>, x: &[T], y: &mut [T]) {
    let (values, col_indices, row_ptr) = a.as_parts();
    for (i, slot) in y.iter_mut().enumerate() {
        let mut acc = T::ZERO;
        for p in row_ptr[i]..row_ptr[i + 1] {
            acc = acc.add(values[p].mul(x[col_indices[p]]));
        }
        *slot = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use leto::Array2;

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut dense = Array2::<f64>::zeros([n, n]);
        for i in 0..n {
            *dense.get_mut([i, i]).unwrap() = 2.0;
            if i > 0 {
                *dense.get_mut([i, i - 1]).unwrap() = -1.0;
            }
            if i + 1 < n {
                *dense.get_mut([i, i + 1]).unwrap() = -1.0;
            }
        }
        CsrMatrix::from_dense(&dense.view())
    }

    #[test]
    fn cg_2x2_spd() {
        // A = [[2, 1], [1, 2]],  b = [3, 3]  →  x = [1, 1]
        let values = vec![2.0_f64, 1.0, 1.0, 2.0];
        let col_indices = vec![0usize, 1, 0, 1];
        let row_ptr = vec![0usize, 2, 4];
        let a = CsrMatrix::from_parts(values, col_indices, row_ptr, 2, 2).unwrap();
        let b = vec![3.0_f64, 3.0];

        let result = cg(&a, &b, 100, 1e-12).unwrap();
        assert!(result.converged);
        assert!(
            result.iterations <= 2,
            "CG should converge in ≤ 2 iterations for a 2×2 SPD system"
        );
        assert!((result.x[0] - 1.0).abs() < 1e-10, "x[0] = {}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 1e-10, "x[1] = {}", result.x[1]);
    }

    #[test]
    fn cg_5x5_laplacian() {
        let a = laplacian_1d(5);
        let b = vec![1.0_f64; 5];

        let result = cg(&a, &b, 200, 1e-10).unwrap();
        assert!(result.converged, "CG should converge for SPD Laplacian");
        assert!(result.residual < 1e-10, "residual = {:e}", result.residual);

        // Verify A·x ≈ b
        let mut ax = vec![0.0_f64; 5];
        spmv_into(&a, &result.x, &mut ax);
        for (axi, bi) in ax.iter().zip(&b) {
            assert!((axi - bi).abs() < 1e-8, "|Ax - b| = {}", (axi - bi).abs());
        }
    }

    #[test]
    fn cg_zero_rhs() {
        let a = laplacian_1d(5);
        let b = vec![0.0_f64; 5];

        let result = cg(&a, &b, 100, 1e-12).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        for xi in &result.x {
            assert_eq!(*xi, 0.0);
        }
    }

    #[test]
    fn cg_shape_mismatch_errors() {
        let a = laplacian_1d(5);
        let b = vec![1.0_f64; 3];
        let err = cg(&a, &b, 100, 1e-12).unwrap_err();
        assert!(matches!(err, LetoError::ShapeMismatch { .. }));
    }
}
