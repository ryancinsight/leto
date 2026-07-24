//! Eigenvalue decomposition for complex Hermitian matrices.
//!
//! Provides two algorithms:
//! - [`hermitian_eigen_jacobi`] — classical Jacobi rotations, O(n³–n⁴), best for
//!   small (n ≤ 200) or ill-conditioned matrices.
//! - [`hermitian_eigen_qr`] — implicit QR with Wilkinson shift, O(n³), better for
//!   larger matrices (delegates to Jacobi for n ≤ 32).
//!
//! Both algorithms assume the input is Hermitian (`A[i,j] = conj(A[j,i])`).
//! They complement the real symmetric solver in [`eigen.rs`].
//!
//! ## References
//! - Golub & Van Loan (2013). *Matrix Computations*, §8.4 (Jacobi), §8.3 (QR).
//! - Parlett (1998). *The Symmetric Eigenvalue Problem*.

use eunomia::Complex;
use leto::{Array1, Array2, LetoError, Result};

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
    /// Final off-diagonal Frobenius norm (convergence indicator).
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

/// Wilkinson shift for the trailing 2×2 sub-block.
fn wilkinson_shift(a: &Array2<C64>, n: usize) -> f64 {
    let a_val = a[[n - 2, n - 2]].re;
    let b = a[[n - 2, n - 1]].norm();
    let d = a[[n - 1, n - 1]].re;
    let tr = a_val + d;
    let det = a_val.mul_add(d, -(b * b));
    let disc = (tr * tr / 4.0 - det).sqrt();
    let l1 = tr / 2.0 + disc;
    let l2 = tr / 2.0 - disc;
    if (l1 - d).abs() < (l2 - d).abs() { l1 } else { l2 }
}

fn sort_eig(
    ev: Array1<f64>,
    vecs: Array2<C64>,
    descending: bool,
) -> (Array1<f64>, Array2<C64>) {
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

fn complex_matmul(lhs: &Array2<C64>, rhs: &Array2<C64>) -> Array2<C64> {
    let [r, k] = lhs.shape();
    let [_, c] = rhs.shape();
    let mut out = Array2::from_elem([r, c], C64::new(0.0, 0.0));
    for i in 0..r {
        for j in 0..c {
            let mut s = C64::new(0.0, 0.0);
            for l in 0..k {
                s += lhs[[i, l]] * rhs[[l, j]];
            }
            out[[i, j]] = s;
        }
    }
    out
}

fn complex_eye(n: usize) -> Array2<C64> {
    let mut eye = Array2::from_elem([n, n], C64::new(0.0, 0.0));
    for i in 0..n {
        eye[[i, i]] = C64::new(1.0, 0.0);
    }
    eye
}

/// Complex QR factorisation via Householder reflectors, returning `(Q, R)`.
fn complex_qr(a: &Array2<C64>, n: usize) -> (Array2<C64>, Array2<C64>) {
    let mut h = a.clone();
    let mut q = complex_eye(n);
    for k in 0..n.saturating_sub(1) {
        let x: Vec<C64> = (k..n).map(|i| h[[i, k]]).collect();
        let sigma = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let sigma = if h[[k, k]].re >= 0.0 { sigma } else { -sigma };
        if sigma.abs() < 1e-14 {
            continue;
        }
        let mut u = x.clone();
        u[0] += C64::new(sigma, 0.0);
        let u_norm_sq: f64 = u.iter().map(|z| z.norm_sqr()).sum();
        if u_norm_sq < 1e-28 {
            continue;
        }
        // Apply reflector to H from the left.
        for j in k..n {
            let mut dot = C64::new(0.0, 0.0);
            for i in k..n {
                dot += u[i - k].conj() * h[[i, j]];
            }
            let fac = 2.0 * dot / u_norm_sq;
            for i in k..n {
                h[[i, j]] -= fac * u[i - k];
            }
        }
        // Accumulate Q.
        for i in 0..n {
            let mut dot = C64::new(0.0, 0.0);
            for j in k..n {
                dot += u[j - k].conj() * q[[j, i]];
            }
            let fac = 2.0 * dot / u_norm_sq;
            for j in k..n {
                q[[j, i]] -= fac * u[j - k];
            }
        }
    }
    (q.mapv(|z| z.conj()), h)
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
            a.shape()[0], a.shape()[1]
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
    let (eigenvalues, eigenvectors) =
        if config.sort_descending { sort_eig(eigenvalues, v, true) } else { (eigenvalues, v) };

    let condition_number = if config.estimate_condition && n > 0 {
        let lo = eigenvalues[n - 1].abs();
        if lo > 1e-14 { Some(eigenvalues[0].abs() / lo) } else { None }
    } else {
        None
    };

    Ok(HermitianEigenResult { eigenvalues, eigenvectors, iterations: iters, off_diagonal_norm: od_norm, condition_number })
}

/// Compute eigendecomposition of a complex Hermitian matrix via implicit QR
/// with Wilkinson shift.
///
/// Delegates to [`hermitian_eigen_jacobi`] for `n ≤ 32`.
///
/// # Errors
/// Same as [`hermitian_eigen_jacobi`].
pub fn hermitian_eigen_qr(
    a: &Array2<C64>,
    config: HermitianEigenConfig,
) -> Result<HermitianEigenResult> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(LetoError::InvalidInput(format!(
            "hermitian_eigen_qr: A must be square, got {}×{}",
            a.shape()[0], a.shape()[1]
        )));
    }
    verify_hermitian(a)?;

    if n <= 32 {
        return hermitian_eigen_jacobi(a, config);
    }

    let mut h = a.clone();
    let mut q = complex_eye(n);
    let mut eigenvalues = Array1::<f64>::zeros([n]);
    let mut iters = 0usize;

    for iter in 0..config.max_iterations {
        iters = iter;
        let shift = if iter % 10 == 0 { h[[n - 1, n - 1]].re } else { wilkinson_shift(&h, n) };

        for i in 0..n {
            h[[i, i]] -= C64::new(shift, 0.0);
        }
        let (q_iter, r) = complex_qr(&h, n);
        h = complex_matmul(&r, &q_iter);
        for i in 0..n {
            h[[i, i]] += C64::new(shift, 0.0);
        }
        q = complex_matmul(&q, &q_iter);

        if off_diag_norm(&h, n) < config.tolerance {
            break;
        }
    }

    for i in 0..n {
        eigenvalues[i] = h[[i, i]].re;
    }

    let od_norm = off_diag_norm(&h, n);
    let (eigenvalues, eigenvectors) =
        if config.sort_descending { sort_eig(eigenvalues, q, true) } else { (eigenvalues, q) };

    let condition_number = if config.estimate_condition && n > 0 {
        let lo = eigenvalues[n - 1].abs();
        if lo > 1e-14 { Some(eigenvalues[0].abs() / lo) } else { None }
    } else {
        None
    };

    Ok(HermitianEigenResult { eigenvalues, eigenvectors, iterations: iters, off_diagonal_norm: od_norm, condition_number })
}
