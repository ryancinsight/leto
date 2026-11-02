//! Eigenvalue decomposition for complex Hermitian matrices.
//!
//! Provides two algorithms:
//! - [`hermitian_eigen_jacobi`] — classical Jacobi rotations, O(n³–n⁴), best for
//!   small (n ≤ 200) or ill-conditioned matrices.
//! - [`hermitian_eigen_qr`] — unitary Householder reduction and shared implicit
//!   QL with Wilkinson shifts, O(n³), for larger matrices (Jacobi for n ≤ 32).
//!
//! Both algorithms assume the input is Hermitian (`A[i,j] = conj(A[j,i])`).
//! They complement the [real symmetric solver](crate::application::linalg::eigen).
//!
//! ## References
//! - Golub & Van Loan (2013). *Matrix Computations*, §8.4 (Jacobi), §8.3 (QR).
//! - Parlett (1998). *The Symmetric Eigenvalue Problem*.

use eunomia::Complex;
use leto::{Array1, Array2, LetoError, Result};

mod kernel;
mod qr;
mod reduce;
#[cfg(test)]
mod tests;

type C64 = Complex<f64>;

/// Result of a Hermitian eigendecomposition.
#[derive(Debug, Clone)]
pub struct HermitianEigenResult {
    /// Real eigenvalues, sorted descending (largest first) by default.
    pub eigenvalues: Array1<f64>,
    /// Eigenvectors as columns (column `k` corresponds to `eigenvalues[k]`).
    pub eigenvectors: Array2<C64>,
    /// Number of sweeps / iterations performed.
    pub iterations: usize,
    /// Final off-diagonal Frobenius norm for Jacobi. For QR above order 32,
    /// a bound on the accumulated discarded tridiagonal norm; floating-point
    /// reduction/rotation error is separate from this convergence indicator.
    pub off_diagonal_norm: f64,
    /// Condition number estimate κ = |λ_max| / |λ_min|, or `None` if λ_min ≈ 0.
    pub condition_number: Option<f64>,
}

/// Configuration for Hermitian eigensolvers.
#[derive(Debug, Clone, Copy)]
pub struct HermitianEigenConfig {
    /// Convergence tolerance on the off-diagonal Frobenius norm.
    pub tolerance: f64,
    /// Maximum number of sweeps (Jacobi) or QR iterations.
    pub max_iterations: usize,
    /// Sort eigenvalues in descending order (largest first).
    pub sort_descending: bool,
    /// Compute condition number estimate.
    pub estimate_condition: bool,
}

impl Default for HermitianEigenConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            max_iterations: 1000,
            sort_descending: true,
            estimate_condition: true,
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Verify that `a` is Hermitian up to `tolerance`.
fn verify_hermitian(a: &Array2<C64>) -> Result<()> {
    let n = a.shape()[0];
    for i in 0..n {
        for j in i + 1..n {
            let diff = (a[[i, j]] - a[[j, i]].conj()).norm();
            if diff > 1e-10 {
                return Err(LetoError::InvalidInput(format!(
                    "hermitian_eigen: matrix is not Hermitian: \
                     |A[{i},{j}] - conj(A[{j},{i}])| = {diff:.2e}"
                )));
            }
        }
    }
    Ok(())
}

/// Off-diagonal Frobenius norm.
fn off_diag_norm(a: &Array2<C64>, n: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                s += a[[i, j]].norm_sqr();
            }
        }
    }
    s.sqrt()
}

fn sort_eig(ev: Array1<f64>, vecs: Array2<C64>, descending: bool) -> (Array1<f64>, Array2<C64>) {
    let n = ev.len();
    let mut idx: Vec<usize> = (0..n).collect();
    if descending {
        idx.sort_by(|&i, &j| ev[j].total_cmp(&ev[i]));
    } else {
        idx.sort_by(|&i, &j| ev[i].total_cmp(&ev[j]));
    }
    let mut sev = Array1::zeros([n]);
    let mut svec = Array2::from_elem([n, n], C64::new(0.0, 0.0));
    for (ni, &oi) in idx.iter().enumerate() {
        sev[ni] = ev[oi];
        for r in 0..n {
            svec[[r, ni]] = vecs[[r, oi]];
        }
    }
    (sev, svec)
}

fn complex_eye(n: usize) -> Array2<C64> {
    let mut eye = Array2::from_elem([n, n], C64::new(0.0, 0.0));
    for i in 0..n {
        eye[[i, i]] = C64::new(1.0, 0.0);
    }
    eye
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute the eigendecomposition of a complex Hermitian matrix using the
/// Jacobi algorithm.
///
/// Best for small matrices (n ≤ 200) or ill-conditioned problems where
/// stability is paramount.
///
/// # Errors
/// - [`LetoError::InvalidInput`] if `a` is not square.
/// - [`LetoError::InvalidInput`] if `a` is not Hermitian.
pub fn hermitian_eigen_jacobi(
    a: &Array2<C64>,
    config: HermitianEigenConfig,
) -> Result<HermitianEigenResult> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(LetoError::InvalidInput(format!(
            "hermitian_eigen_jacobi: A must be square, got {}×{}",
            a.shape()[0],
            a.shape()[1]
        )));
    }
    verify_hermitian(a)?;

    let mut h = a.clone();
    let mut v = complex_eye(n);
    let mut eigenvalues = Array1::<f64>::zeros([n]);
    let mut iters = 0usize;

    'outer: for sweep in 0..config.max_iterations {
        iters = sweep;
        let mut max_od = 0.0_f64;

        for p in 0..n {
            for q in p + 1..n {
                let h_pp = h[[p, p]].re;
                let h_qq = h[[q, q]].re;
                let h_pq = h[[p, q]];
                let r = h_pq.norm();
                max_od = max_od.max(r);
                if r <= 1e-15 {
                    continue;
                }
                let e_bar = h_pq.conj() / r;
                let tau = (h_qq - h_pp) / (2.0 * r);
                let t = if tau >= 0.0 {
                    1.0 / (tau + tau.mul_add(tau, 1.0).sqrt())
                } else {
                    -1.0 / (-tau + tau.mul_add(tau, 1.0).sqrt())
                };
                let c = 1.0 / t.mul_add(t, 1.0).sqrt();
                let s = t * c;

                for i in 0..n {
                    if i != p && i != q {
                        let hip = h[[i, p]];
                        let hiq = h[[i, q]];
                        h[[i, p]] = c * hip - e_bar * (s * hiq);
                        h[[i, q]] = s * hip + e_bar * (c * hiq);
                        h[[p, i]] = h[[i, p]].conj();
                        h[[q, i]] = h[[i, q]].conj();
                    }
                }
                h[[p, p]] = C64::new(t.mul_add(-r, h_pp), 0.0);
                h[[q, q]] = C64::new(t.mul_add(r, h_qq), 0.0);
                h[[p, q]] = C64::new(0.0, 0.0);
                h[[q, p]] = C64::new(0.0, 0.0);

                for i in 0..n {
                    let vip = v[[i, p]];
                    let viq = v[[i, q]];
                    v[[i, p]] = c * vip - e_bar * (s * viq);
                    v[[i, q]] = s * vip + e_bar * (c * viq);
                }
            }
        }

        if max_od < config.tolerance {
            break 'outer;
        }
    }

    for i in 0..n {
        eigenvalues[i] = h[[i, i]].re;
    }

    let od_norm = off_diag_norm(&h, n);
    let (eigenvalues, eigenvectors) = if config.sort_descending {
        sort_eig(eigenvalues, v, true)
    } else {
        (eigenvalues, v)
    };

    let condition_number = if config.estimate_condition && n > 0 {
        let lo = eigenvalues[n - 1].abs();
        if lo > 1e-14 {
            Some(eigenvalues[0].abs() / lo)
        } else {
            None
        }
    } else {
        None
    };

    Ok(HermitianEigenResult {
        eigenvalues,
        eigenvectors,
        iterations: iters,
        off_diagonal_norm: od_norm,
        condition_number,
    })
}

/// Compute eigendecomposition of a complex Hermitian matrix via unitary
/// Householder reduction and implicit-shift tridiagonal QL (the QR family).
///
/// Delegates to [`hermitian_eigen_jacobi`] for `n ≤ 32`.
/// Above that boundary the accepted lower triangle defines the Hermitian
/// matrix, the diagonal must be real, and every entry must be finite.
/// `max_iterations` bounds completed shifts across the entire matrix.
/// `tolerance` is the absolute Frobenius budget for discarded off-diagonal
/// entries; the real-symmetric solver retains its own rounding-based policy.
/// Exhaustion diagnostics use that global bound and the original input units.
/// A zero budget succeeds only if no shifts are needed. With sorting disabled,
/// ordering is algorithm-dependent (ascending after tridiagonal iteration;
/// input order for an already-converged diagonal shortcut).
/// The Jacobi entry and its small-matrix delegation retain their existing
/// convergence/ordering behavior.
///
/// # Errors
/// Same input shape/symmetry errors as [`hermitian_eigen_jacobi`]. Above order
/// 32, also rejects invalid tolerances, non-finite entries and non-real
/// diagonals, reports [`LetoError::ConvergenceError`] on exhausted budgets,
/// and reports numerical breakdown or overflow instead of non-finite results.
pub fn hermitian_eigen_qr(
    a: &Array2<C64>,
    config: HermitianEigenConfig,
) -> Result<HermitianEigenResult> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(LetoError::InvalidInput(format!(
            "hermitian_eigen_qr: A must be square, got {}×{}",
            a.shape()[0],
            a.shape()[1]
        )));
    }
    verify_hermitian(a)?;

    if n <= 32 {
        return hermitian_eigen_jacobi(a, config);
    }

    qr::decompose(a, config)
}
