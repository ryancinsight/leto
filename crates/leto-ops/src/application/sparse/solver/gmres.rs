//! Restarted GMRES solver for general non-symmetric linear systems.

use super::super::CsrMatrix;
use super::cg::spmv_into;
use crate::domain::real::RealScalar;
use leto::LetoError;

/// Result of a GMRES solve.
#[derive(Debug, Clone, PartialEq)]
pub struct GmresResult<T> {
    /// Solution vector (length `nrows`).
    pub x: Vec<T>,
    /// Total Arnoldi iterations (matvecs) performed.
    pub iterations: usize,
    /// Final L2 norm of the residual.
    pub residual: T,
    /// `true` if the residual met the tolerance within the iteration limit.
    pub converged: bool,
}

/// Solve `A·x = b` via restarted GMRES.
///
/// Works for general non-symmetric matrices. Uses a zero initial guess.
/// The Krylov subspace is limited to `restart` vectors; if restart is reached
/// without convergence the solver restarts from the current approximation. The
/// total number of Arnoldi iterations (i.e. matrix–vector products) across all
/// restarts is capped at `max_iters`.
///
/// Convergence is checked on the relative residual:
/// `‖r‖₂ ≤ tol · ‖b‖₂`.
///
/// # Errors
/// - [`LetoError::ShapeMismatch`] if `b.len() != nrows`.
/// - [`LetoError::ConvergenceError`] if `max_iters` is exceeded.
pub fn gmres<T: RealScalar>(
    a: &CsrMatrix<T>,
    b: &[T],
    restart: usize,
    max_iters: usize,
    tol: T,
) -> Result<GmresResult<T>, LetoError> {
    let n = a.nrows();
    if b.len() != n {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![b.len()],
            rhs: vec![n],
        });
    }

    let b_norm = T::sqrt(T::dot_slice(b, b));
    if b_norm == T::ZERO {
        return Ok(GmresResult {
            x: vec![T::ZERO; n],
            iterations: 0,
            residual: T::ZERO,
            converged: true,
        });
    }

    let rs = restart.max(1);
    let tol_b = tol * b_norm;

    let mut x = vec![T::ZERO; n];
    let mut total_iters: usize = 0;

    loop {
        // r = b - A·x
        let mut r = vec![T::ZERO; n];
        spmv_into(a, &x, &mut r);
        for (ri, bi) in r.iter_mut().zip(b.iter()) {
            *ri = *bi - *ri;
        }

        let beta = T::sqrt(T::dot_slice(&r, &r));
        if beta < tol_b {
            return Ok(GmresResult {
                x,
                iterations: total_iters,
                residual: beta,
                converged: true,
            });
        }

        // Arnoldi basis V[0..rs]
        let mut v: Vec<Vec<T>> = Vec::with_capacity(rs);
        let inv_beta = T::ONE / beta;
        v.push(r.iter().map(|&ri| ri * inv_beta).collect());

        // Hessenberg matrix (rs+1) × rs, column-major
        let mut h = vec![T::ZERO; (rs + 1) * rs];
        // Givens rotation cosines / sines
        let mut cs = vec![T::ZERO; rs];
        let mut sn = vec![T::ZERO; rs];
        // Transformed RHS
        let mut g = vec![T::ZERO; rs + 1];
        g[0] = beta;

        let inner_limit = (max_iters - total_iters).min(rs);
        if inner_limit == 0 {
            break;
        }

        for j in 0..inner_limit {
            total_iters += 1;

            // w = A·v[j]
            let mut w = vec![T::ZERO; n];
            spmv_into(a, &v[j], &mut w);

            // Modified Gram–Schmidt against v[0..=j]
            for i in 0..=j {
                let h_ij = T::dot_slice(&w, &v[i]);
                h[i * rs + j] = h_ij;
                for (wk, v_ik) in w.iter_mut().zip(v[i].iter()) {
                    *wk -= h_ij * *v_ik;
                }
            }

            // h[j+1, j] = ‖w‖
            let h_j1j = T::sqrt(T::dot_slice(&w, &w));

            // Apply previous Givens rotations to column j
            for i in 0..j {
                let temp = cs[i] * h[i * rs + j] + sn[i] * h[(i + 1) * rs + j];
                h[(i + 1) * rs + j] = -sn[i] * h[i * rs + j] + cs[i] * h[(i + 1) * rs + j];
                h[i * rs + j] = temp;
            }

            // Lucky breakdown — Krylov subspace exhausted.
            let lucky = h_j1j == T::ZERO;

            // New Givens rotation to zero h[j+1, j]
            let h_jj = h[j * rs + j];
            let nu = T::sqrt(h_jj * h_jj + h_j1j * h_j1j);
            if nu != T::ZERO {
                cs[j] = h_jj / nu;
                sn[j] = h_j1j / nu;
                h[j * rs + j] = nu;
                h[(j + 1) * rs + j] = T::ZERO;

                let gj = g[j];
                g[j] = cs[j] * gj;
                g[j + 1] = -sn[j] * gj;
            } else {
                // Both h_jj and h_j1j are zero — exact solution.
                cs[j] = T::ONE;
                sn[j] = T::ZERO;
                g[j + 1] = T::ZERO;
            }

            let residual_j = T::abs(g[j + 1]);
            let last_inner = j + 1 == inner_limit;

            if residual_j < tol_b || lucky || last_inner {
                // Solve upper triangular R·y = g (first m = j+1 entries)
                let m = j + 1;
                let mut y = vec![T::ZERO; m];
                for i in (0..m).rev() {
                    let mut s = T::ZERO;
                    for k in (i + 1)..m {
                        s += h[i * rs + k] * y[k];
                    }
                    y[i] = (g[i] - s) / h[i * rs + i];
                }

                // x = x₀ + V·y
                for (i, yi) in y.iter().enumerate() {
                    for (xj, v_ij) in x.iter_mut().zip(v[i].iter()) {
                        *xj += *yi * *v_ij;
                    }
                }

                if lucky || residual_j < tol_b {
                    // After lucky breakdown recompute and return actual residual.
                    let final_r = if lucky {
                        let mut rr = vec![T::ZERO; n];
                        spmv_into(a, &x, &mut rr);
                        for (rri, bi) in rr.iter_mut().zip(b.iter()) {
                            *rri = *bi - *rri;
                        }
                        T::sqrt(T::dot_slice(&rr, &rr))
                    } else {
                        residual_j
                    };
                    return Ok(GmresResult {
                        x,
                        iterations: total_iters,
                        residual: final_r,
                        converged: true,
                    });
                }

                break; // restart
            }

            // Normalize w to form v[j+1]
            let inv_h = T::ONE / h_j1j;
            v.push(w.iter().map(|&wi| wi * inv_h).collect());
        }

        if total_iters >= max_iters {
            break;
        }
    }

    // Compute final residual for the error report.
    let mut r = vec![T::ZERO; n];
    spmv_into(a, &x, &mut r);
    for (ri, bi) in r.iter_mut().zip(b.iter()) {
        *ri = *bi - *ri;
    }
    let final_res = T::sqrt(T::dot_slice(&r, &r));

    Err(LetoError::ConvergenceError {
        max_iters,
        residual: final_res.to_f64(),
        tol: tol.to_f64(),
    })
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
    fn gmres_2x2_spd() {
        // A = [[2, 1], [1, 2]],  b = [3, 3]  →  x = [1, 1]
        let values = vec![2.0_f64, 1.0, 1.0, 2.0];
        let col_indices = vec![0usize, 1, 0, 1];
        let row_ptr = vec![0usize, 2, 4];
        let a = CsrMatrix::from_parts(values, col_indices, row_ptr, 2, 2).unwrap();
        let b = vec![3.0_f64, 3.0];

        let result = gmres(&a, &b, 2, 100, 1e-12).unwrap();
        assert!(result.converged, "GMRES should converge for SPD 2×2");
        assert!((result.x[0] - 1.0).abs() < 1e-10, "x[0] = {}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 1e-10, "x[1] = {}", result.x[1]);
    }

    #[test]
    fn gmres_5x5_tridiag() {
        let a = laplacian_1d(5);
        let b = vec![1.0_f64; 5];

        let result = gmres(&a, &b, 5, 200, 1e-10).unwrap();
        assert!(
            result.converged,
            "GMRES should converge for 5×5 SPD tridiagonal"
        );
        assert!(result.residual < 1e-10, "residual = {:e}", result.residual);

        let mut ax = vec![0.0_f64; 5];
        spmv_into(&a, &result.x, &mut ax);
        for (axi, bi) in ax.iter().zip(&b) {
            assert!((axi - bi).abs() < 1e-8, "|Ax - b| = {}", (axi - bi).abs());
        }
    }

    #[test]
    fn gmres_nonsymmetric_4x4() {
        // A = [[2, 1, 0, 0],
        //      [1, 3, 1, 0],
        //      [0, 2, 4, 1],
        //      [0, 0, 1, 2]]
        // Non-symmetric due to row 2 having 2 instead of 1 on the sub-diagonal.
        let values = vec![2.0_f64, 1.0, 1.0, 3.0, 1.0, 2.0, 4.0, 1.0, 1.0, 2.0];
        let col_indices = vec![0usize, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let row_ptr = vec![0usize, 2, 5, 8, 10];
        let a = CsrMatrix::from_parts(values, col_indices, row_ptr, 4, 4).unwrap();
        let b = vec![3.0_f64, 5.0, 7.0, 3.0];

        let result = gmres(&a, &b, 4, 200, 1e-10).unwrap();
        assert!(
            result.converged,
            "GMRES should converge for non-symmetric 4×4"
        );
        assert!(result.residual < 1e-10, "residual = {:e}", result.residual);

        let mut ax = vec![0.0_f64; 4];
        spmv_into(&a, &result.x, &mut ax);
        for (axi, bi) in ax.iter().zip(&b) {
            assert!((axi - bi).abs() < 1e-8, "|Ax - b| = {}", (axi - bi).abs());
        }
    }

    #[test]
    fn gmres_5x5_laplacian() {
        let a = laplacian_1d(5);
        let b = vec![1.0_f64; 5];
        let x_true = [2.5, 4.0, 4.5, 4.0, 2.5];
        let result = gmres(&a, &b, 5, 200, 1e-10).unwrap();
        assert!(result.converged);
        for (xi, xti) in result.x.iter().zip(x_true.iter()) {
            assert!((xi - xti).abs() < 1e-10)
        }
    }

    #[test]
    fn gmres_zero_rhs() {
        let a = laplacian_1d(5);
        let b = vec![0.0_f64; 5];

        let result = gmres(&a, &b, 5, 100, 1e-12).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        for xi in &result.x {
            assert_eq!(*xi, 0.0);
        }
    }

    #[test]
    fn gmres_shape_mismatch_errors() {
        let a = laplacian_1d(5);
        let b = vec![1.0_f64; 3];
        let err = gmres(&a, &b, 5, 100, 1e-12).unwrap_err();
        assert!(matches!(err, LetoError::ShapeMismatch { .. }));
    }
}
