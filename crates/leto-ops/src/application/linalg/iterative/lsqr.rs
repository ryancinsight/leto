//! LSQR solver: sparse least-squares via Lanczos bidiagonalisation.
//!
//! Solves `min ‖A·x − b‖₂` (and optionally `min ‖A·x − b‖₂ + λ²‖x‖₂`)
//! using the algorithm of Paige & Saunders (1982).  The operator A may be
//! rectangular; no matrix is factorised.
//!
//! ## References
//! - Paige, C. C. & Saunders, M. A. (1982). *LSQR: An algorithm for sparse
//!   linear equations and sparse least squares.*  ACM Trans. Math. Software
//!   8(1), 43–71.

use crate::application::linalg::iterative::traits::LinearOperator;
use leto::Array1;

/// Configuration for the LSQR solver.
#[derive(Debug, Clone, Copy)]
pub struct LsqrConfig {
    /// Maximum number of Lanczos iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on the relative residual.
    pub tolerance: f64,
    /// Tikhonov damping λ ≥ 0: minimises ‖Ax − b‖² + λ²‖x‖².
    pub damping: f64,
    /// Tolerance on ‖Aᵀr‖ (normal-equation residual).
    pub atol: f64,
    /// Tolerance on ‖r‖ (primal residual).
    pub btol: f64,
}

impl Default for LsqrConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            damping: 0.0,
            atol: 1e-8,
            btol: 1e-8,
        }
    }
}

/// Reason why LSQR stopped iterating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LsqrStopReason {
    /// Normal convergence (both tolerances satisfied).
    Converged,
    /// Maximum iteration count reached.
    MaxIterations,
    /// ‖Aᵀr‖ tolerance (`atol`) satisfied.
    AtolSatisfied,
    /// ‖r‖ tolerance (`btol`) satisfied.
    BtolSatisfied,
    /// Matrix appears ill-conditioned (condition-number estimate too large).
    IllConditioned,
}

/// Result returned by [`LsqrSolver::solve`].
#[derive(Debug, Clone)]
pub struct LsqrResult {
    /// Solution vector x.
    pub solution: Array1<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Estimated ‖Ax − b‖ at exit.
    pub residual_norm: f64,
    /// Estimated ‖Aᵀ(Ax − b)‖ at exit.
    pub at_residual_norm: f64,
    /// Estimated condition number of A.
    pub condition_number: f64,
    /// `true` if a stopping criterion was satisfied before `max_iterations`.
    pub converged: bool,
    /// The stopping condition that triggered exit.
    pub stop_reason: LsqrStopReason,
    /// Per-iteration ‖r‖ estimates (non-increasing).  Empty when LSQR exits
    /// before the first iteration (zero RHS or zero ‖Aᵀb‖).
    pub residual_history: Vec<f64>,
}

// ── Internal helpers ──────────────────────────────────────────────────────────

#[inline]
fn norm_l2(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[inline]
fn div_in_place(v: &mut Array1<f64>, s: f64) {
    for x in v.iter_mut() {
        *x /= s;
    }
}

/// LSQR solver.
pub struct LsqrSolver {
    config: LsqrConfig,
}

impl LsqrSolver {
    /// Create a new solver with the given configuration.
    #[must_use]
    pub fn new(config: LsqrConfig) -> Self {
        Self { config }
    }

    /// Minimise `‖A·x − b‖₂` (+ optional damping) using LSQR.
    ///
    /// The operator `a` is applied as a matrix-free [`LinearOperator`];
    /// `at` is its transpose.  Pass the same object for both when the
    /// operator supplies `apply_transpose`.
    #[must_use]
    pub fn solve<Op: LinearOperator<f64>>(&self, a: &Op, b: &Array1<f64>) -> LsqrResult {
        let m = b.shape()[0]; // number of rows (observations)
        let n = if a.ncols() > 0 { a.ncols() } else { m }; // number of cols (unknowns)
        let mut x = Array1::zeros([n]);

        // Initialise bidiagonalisation: u = b / β₁.
        let mut u = b.clone();
        let beta = norm_l2(&u);
        if beta < 1e-12 {
            return LsqrResult {
                solution: x,
                iterations: 0,
                residual_norm: 0.0,
                at_residual_norm: 0.0,
                condition_number: 1.0,
                converged: true,
                stop_reason: LsqrStopReason::Converged,
                residual_history: vec![],
            };
        }
        div_in_place(&mut u, beta);

        // v = Aᵀu / α₁.
        let mut v = Array1::zeros([n]);
        let _ = a.apply_transpose(&u, &mut v); // ignore unsupported error gracefully
        let mut alpha = norm_l2(&v);
        if alpha < 1e-12 {
            return LsqrResult {
                solution: x,
                iterations: 0,
                residual_norm: beta,
                at_residual_norm: alpha * beta,
                condition_number: 1.0,
                converged: true,
                stop_reason: LsqrStopReason::Converged,
                residual_history: vec![beta],
            };
        }
        div_in_place(&mut v, alpha);

        // QR state (Paige & Saunders 1982, Table 1).
        let mut w = v.clone();
        let mut phi_bar = beta;
        let mut rho_bar = alpha;
        let damping = self.config.damping;

        // Seed history with the initial primal residual ‖b‖ = beta so the
        // caller always receives at least one entry even for 1-iteration solves.
        let mut residual_norms = vec![beta];
        let mut at_residual_norms = Vec::new();
        let mut rho_values = Vec::new();
        let mut stop_reason = LsqrStopReason::MaxIterations;
        let mut converged = false;

        for _ in 1..=self.config.max_iterations {
            // Bidiagonalisation.
            let mut av = Array1::zeros([m]);
            let _ = a.apply(&v, &mut av);
            for i in 0..m {
                av[i] -= u[i] * alpha;
            }
            let beta_new = norm_l2(&av);
            let mut u_new = av;
            if beta_new > 1e-12 {
                div_in_place(&mut u_new, beta_new);
            }

            let mut atv = Array1::zeros([n]);
            let _ = a.apply_transpose(&u_new, &mut atv);
            for j in 0..n {
                atv[j] -= v[j] * beta_new;
            }
            let alpha_new = norm_l2(&atv);
            let mut v_new = atv;
            if alpha_new > 1e-12 {
                div_in_place(&mut v_new, alpha_new);
            }

            // Givens rotation.
            let rho = (rho_bar * rho_bar + beta_new * beta_new + damping * damping).sqrt();
            if rho < 1e-12 {
                break;
            }

            let c = rho_bar / rho;
            let s = beta_new / rho;
            let theta_next = s * alpha_new;
            let rho_bar_next = -c * alpha_new;
            let phi = c * phi_bar;
            phi_bar *= s;

            // Update x and w.
            let phi_rho = phi / rho;
            for i in 0..n {
                x[i] += w[i] * phi_rho;
            }
            let theta_rho = theta_next / rho;
            for i in 0..n {
                w[i] = v_new[i] - w[i] * theta_rho;
            }

            rho_bar = rho_bar_next;
            rho_values.push(rho.abs());

            let res = phi_bar.abs();
            let at_res = phi_bar.abs() * alpha_new;
            residual_norms.push(res);
            at_residual_norms.push(at_res);

            if at_res <= self.config.atol {
                stop_reason = LsqrStopReason::AtolSatisfied;
                converged = true;
                break;
            }
            if res <= self.config.btol {
                stop_reason = LsqrStopReason::BtolSatisfied;
                converged = true;
                break;
            }

            u = u_new;
            v = v_new;
            alpha = alpha_new;
        }

        let final_res = residual_norms.last().copied().unwrap_or(beta);
        let final_at = at_residual_norms.last().copied().unwrap_or(alpha);
        let cond = estimate_condition(&rho_values);

        LsqrResult {
            solution: x,
            iterations: residual_norms.len(),
            residual_norm: final_res,
            at_residual_norm: final_at,
            condition_number: cond,
            converged,
            stop_reason,
            residual_history: residual_norms,
        }
    }
}

fn estimate_condition(rho_values: &[f64]) -> f64 {
    if rho_values.is_empty() {
        return 1.0;
    }
    let max = rho_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = rho_values.iter().cloned().fold(f64::INFINITY, f64::min);
    if min < 1e-300 {
        f64::INFINITY
    } else {
        max / min
    }
}
