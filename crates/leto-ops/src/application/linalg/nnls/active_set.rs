//! Lawson & Hanson (1974) active-set solve: the outer Lagrange-multiplier
//! selection, the passive-set QR resolve, and the negative-coefficient
//! interpolation step. See the module-level docs (`super`) for the
//! algorithm overview and references.

use crate::application::linalg::qr::qr_decompose;
use leto::{Array1, Array2, ArrayView1, ArrayView2, LetoError, Result as LetoResult};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the NNLS solver.
#[derive(Debug, Clone, Copy)]
pub struct NnlsConfig {
    /// Maximum number of outer (active-set) iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on the maximum Lagrange multiplier.
    ///
    /// The solver stops when `max(w_j for j not in P) ≤ tolerance` where
    /// `w = Aᵀ(b − Ax)` is the negative gradient.
    pub tolerance: f64,
}

impl Default for NnlsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            tolerance: 1e-8,
        }
    }
}

// ── Result ────────────────────────────────────────────────────────────────────

/// Result returned by [`nnls`].
#[derive(Debug, Clone)]
pub struct NnlsResult {
    /// Non-negative solution vector `x`.
    pub solution: Array1<f64>,
    /// ‖Ax − b‖₂ at exit.
    pub residual_norm: f64,
    /// Number of active-set iterations performed.
    pub iterations: usize,
    /// `true` if the convergence tolerance was satisfied.
    pub converged: bool,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Solve `min ‖A·x − b‖₂  subject to  x ≥ 0`.
///
/// Uses the Lawson–Hanson active-set algorithm with `qr_decompose` for the
/// inner unconstrained solves.  The matrix `A` must be `m × n` with `m ≥ n`
/// (overdetermined or square).
///
/// # Errors
///
/// Returns [`LetoError::ShapeMismatch`] if `A` and `b` are incompatible.
/// Returns [`LetoError::StorageError`] if an inner QR factorisation fails
/// due to rank deficiency.
///
/// # Example
///
/// ```
/// use leto::Array1;
/// use leto::Array2;
/// use leto_ops::{nnls, NnlsConfig};
///
/// let a = Array2::from_vec([2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
/// let b = Array1::from_vec(2, vec![3.0, 4.0]).unwrap();
/// let result = nnls(&a.view(), &b.view(), NnlsConfig::default()).unwrap();
/// assert!(result.converged);
/// assert!((result.solution[0] - 3.0).abs() < 1e-10);
/// assert!((result.solution[1] - 4.0).abs() < 1e-10);
/// ```
pub fn nnls(
    a: &ArrayView2<'_, f64>,
    b: &ArrayView1<'_, f64>,
    config: NnlsConfig,
) -> LetoResult<NnlsResult> {
    let [m, n] = a.shape();
    if b.shape()[0] != m {
        return Err(LetoError::ShapeMismatch {
            lhs: vec![m],
            rhs: vec![b.shape()[0]],
        });
    }
    if n == 0 {
        return Ok(NnlsResult {
            solution: Array1::zeros([0]),
            residual_norm: norm_l2(b),
            iterations: 0,
            converged: true,
        });
    }

    // ── Initialisation ────────────────────────────────────────────────────
    let mut x = Array1::from_elem([n], 0.0_f64);
    // w = Aᵀ(b − Ax) — negative gradient at x=0.
    let mut w = at_times(a, b);
    let mut passive: Vec<usize> = Vec::with_capacity(n);
    // Boolean mask for O(1) membership tests.
    let mut is_passive = vec![false; n];
    // Workspace: residual r = b − Ax, reused each outer iteration.
    let mut residual = Array1::from_elem([m], 0.0_f64);
    // Workspace reused across inner solves.
    let mut a_passive = Array2::zeros([m, n]);

    let mut iterations = 0;
    let mut converged = false;

    loop {
        let (max_idx, max_val) = max_among_inactive(&w, &is_passive);
        if max_val <= config.tolerance {
            converged = true;
            break;
        }
        if iterations >= config.max_iterations {
            break;
        }

        passive.push(max_idx);
        is_passive[max_idx] = true;
        iterations += 1;

        // ── Inner loop: solve restricted to P, remove negative variables ──
        loop {
            if passive.is_empty() {
                break;
            }
            // Build A[:, P].
            build_passive_matrix(a, &passive, &mut a_passive);
            let k = passive.len();
            let a_p_view = a_passive
                .view()
                .slice(&[(0, m, 1), (0, k, 1)])
                .map_err(|e| LetoError::StorageError {
                    reason: e.to_string(),
                })?;

            // Solve unconstrained LS on the passive set.
            let z = qr_decompose(&a_p_view)?.solve_least_squares(b)?;

            // Check for negative components (allow small negative for fp error).
            let (_neg_idx, neg_val) = min_among(&z, passive.len());
            if neg_val > -1e-12 {
                // All non-negative — accept and break inner loop.
                for (local_idx, &global_idx) in passive.iter().enumerate() {
                    x[global_idx] = z[local_idx];
                }
                break;
            }

            // At least one z_j < 0 — find interpolation parameter α.
            let mut alpha = 1.0_f64;
            let mut remove_mask: Vec<usize> = Vec::new();

            for (local_idx, &global_idx) in passive.iter().enumerate() {
                if z[local_idx] <= 0.0 {
                    let denom = x[global_idx] - z[local_idx];
                    if denom.abs() > 1e-15 {
                        let ratio = x[global_idx] / denom;
                        if ratio < alpha {
                            alpha = ratio;
                        }
                    }
                    remove_mask.push(local_idx);
                }
            }

            // Interpolate: x ← x + α(z − x).
            for (local_idx, &global_idx) in passive.iter().enumerate() {
                x[global_idx] += alpha * (z[local_idx] - x[global_idx]);
            }

            // Remove variables that hit the bound from the passive set.
            for &local_idx in remove_mask.iter().rev() {
                let global_idx = passive[local_idx];
                if x[global_idx].abs() < 1e-15 {
                    x[global_idx] = 0.0;
                }
                passive.remove(local_idx);
                is_passive[global_idx] = false;
            }
        }

        // Update the negative gradient: w = Aᵀ(b − Ax) in O(m·n).
        update_gradient(a, b, &x, &mut w, &mut residual);
    }

    let res_norm = residual_norm(a, b, &x);
    Ok(NnlsResult {
        solution: x,
        residual_norm: res_norm,
        iterations,
        converged,
    })
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Compute `Aᵀ·v` (or `Aᵀ·b` when `v = b`).
fn at_times(a: &ArrayView2<'_, f64>, v: &ArrayView1<'_, f64>) -> Array1<f64> {
    let [m, n] = a.shape();
    let mut result = Array1::zeros([n]);
    for j in 0..n {
        let mut dot = 0.0;
        for i in 0..m {
            dot += a[[i, j]] * v[i];
        }
        result[j] = dot;
    }
    result
}

/// Find the maximum value and its index among inactive indices (those where
/// `is_passive[j] == false`).  Returns `(index, value)`.  Uses an O(1)
/// boolean mask instead of `Vec::contains`.
fn max_among_inactive(w: &Array1<f64>, is_passive: &[bool]) -> (usize, f64) {
    let n = w.shape()[0];
    let mut best_idx = 0;
    let mut best_val = f64::NEG_INFINITY;
    for j in 0..n {
        if is_passive[j] {
            continue;
        }
        if w[j] > best_val {
            best_val = w[j];
            best_idx = j;
        }
    }
    (best_idx, best_val)
}

/// Find the minimum value and its index among the first `k` entries.
fn min_among(z: &Array1<f64>, k: usize) -> (usize, f64) {
    let mut best_idx = 0;
    let mut best_val = z[0];
    for j in 1..k {
        if z[j] < best_val {
            best_val = z[j];
            best_idx = j;
        }
    }
    (best_idx, best_val)
}

/// Copy columns of `a` indexed by `passive` into `dest`.
fn build_passive_matrix(a: &ArrayView2<'_, f64>, passive: &[usize], dest: &mut Array2<f64>) {
    let m = a.shape()[0];
    for (local_j, &global_j) in passive.iter().enumerate() {
        for i in 0..m {
            dest[[i, local_j]] = a[[i, global_j]];
        }
    }
}

/// Update the negative gradient `w ← Aᵀ(b − Ax)` in O(m·n).
///
/// Computes the residual `r = b − Ax` once (O(m·n)), then
/// `w = Aᵀ r` (also O(m·n)).  The `residual` buffer is reused across calls.
fn update_gradient(
    a: &ArrayView2<'_, f64>,
    b: &ArrayView1<'_, f64>,
    x: &Array1<f64>,
    w: &mut Array1<f64>,
    residual: &mut Array1<f64>,
) {
    let [m, n] = a.shape();
    // r = b − Ax.
    for i in 0..m {
        residual[i] = b[i] - dot_row(a, x, i);
    }
    // w = Aᵀ r.
    for j in 0..n {
        let mut dot = 0.0;
        for i in 0..m {
            dot += a[[i, j]] * residual[i];
        }
        w[j] = dot;
    }
}

/// Compute `(Ax)[i] = Σ_j a[i,j] · x[j]`.
fn dot_row(a: &ArrayView2<'_, f64>, x: &Array1<f64>, i: usize) -> f64 {
    let n = a.shape()[1];
    let mut sum = 0.0;
    for j in 0..n {
        sum += a[[i, j]] * x[j];
    }
    sum
}

/// Compute `‖Ax − b‖₂`.
fn residual_norm(a: &ArrayView2<'_, f64>, b: &ArrayView1<'_, f64>, x: &Array1<f64>) -> f64 {
    let m = a.shape()[0];
    let mut sum_sq = 0.0;
    for i in 0..m {
        let diff = dot_row(a, x, i) - b[i];
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

/// L2 norm of a vector.
fn norm_l2(v: &ArrayView1<'_, f64>) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}
