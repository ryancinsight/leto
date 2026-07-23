//! GMRES(m) solver with Arnoldi iteration and Givens rotations.
//!
//! Solves general non-symmetric linear systems `A·x = b` using the
//! Generalized Minimal Residual method with restarts.
//!
//! ## Algorithm sketch
//!
//! 1. Build orthonormal Krylov basis `V_m` via Arnoldi / modified Gram-Schmidt.
//! 2. Maintain the least-squares problem `min ‖β e₁ − H̄_m y‖` incrementally
//!    through Givens rotations applied to the growing Hessenberg matrix.
//! 3. When the restart dimension `m` is reached (or convergence), update `x`.
//! 4. Restart from the new residual.
//!
//! ## Optimality theorem (Saad & Schultz 1986)
//!
//! GMRES minimises ‖r_k‖₂ over `x₀ + K_k(A, r₀)`, i.e. it is the best
//! Krylov method without restarts.
//!
//! ## References
//! - Saad & Schultz (1986). *GMRES: A generalized minimal residual algorithm.*
//!   SIAM J. Sci. Stat. Comput. 7(3), 856–869.

use super::super::config::IterativeSolverConfig;
use super::super::convergence::ConvergenceMonitor;
use super::super::preconditioners::IdentityPreconditioner;
use super::super::traits::{
    Configurable, IterativeLinearSolver, LinearOperator, LinearSolver, Preconditioner,
};
use super::{arnoldi, givens};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, Array2, LetoError, Result};
use std::fmt::Debug;
use std::sync::Mutex;

struct Workspace<T: RealField + Copy> {
    v: Array2<T>,
    h: Array2<T>,
    g: Array1<T>,
    c: Array1<T>,
    s: Array1<T>,
    basis_work: Array1<T>,
    work: Array1<T>,
    precond_work: Array1<T>,
    ax: Array1<T>,
}

/// GMRES(m) solver with optional restart and left preconditioning.
///
/// `restart_dim` (default: 30) controls the maximum Krylov subspace size
/// between restarts.  Larger values reduce restarts at the cost of O(n·m)
/// memory.
pub struct GMRES<T: RealField + Copy> {
    config: IterativeSolverConfig<T>,
    restart_dim: usize,
    workspace: Mutex<Option<Workspace<T>>>,
}

#[inline]
fn fill_vec<T: Copy>(v: &mut Array1<T>, val: T) {
    for i in 0..v.shape()[0] {
        v[i] = val;
    }
}

#[inline]
fn fill_mat<T: Copy>(m: &mut Array2<T>, val: T) {
    let [r, c] = m.shape();
    for i in 0..r {
        for j in 0..c {
            m[[i, j]] = val;
        }
    }
}

#[inline]
fn vec_norm<T: NumericElement>(v: &Array1<T>) -> T {
    let mut s = T::ZERO;
    for i in 0..v.shape()[0] {
        s += v[i] * v[i];
    }
    s.sqrt()
}

#[inline]
fn sub_assign<T: NumericElement>(tgt: &mut Array1<T>, rhs: &Array1<T>) {
    for i in 0..tgt.shape()[0] {
        tgt[i] -= rhs[i];
    }
}

impl<T: RealField + Copy + FloatElement + Debug> GMRES<T> {
    /// Create with explicit configuration and restart dimension.
    ///
    /// # Panics
    /// Panics if `restart_dim == 0`.
    pub fn new(config: IterativeSolverConfig<T>, restart_dim: usize) -> Self {
        assert!(restart_dim > 0, "GMRES restart dimension must be positive");
        Self { config, restart_dim, workspace: Mutex::new(None) }
    }

    /// Create with default configuration and `restart_dim = 30`.
    #[must_use]
    pub fn default() -> Self {
        Self::new(IterativeSolverConfig::default(), 30)
    }

    /// Solve with an explicit preconditioner.
    pub fn solve_preconditioned<Op: LinearOperator<T> + ?Sized, P: Preconditioner<T>>(
        &self,
        a: &Op,
        b: &Array1<T>,
        precond: &P,
        x: &mut Array1<T>,
    ) -> Result<ConvergenceMonitor<T>> {
        let n = b.shape()[0];
        let a_size = a.size();
        if a_size != 0 && a_size != n {
            return Err(LetoError::InvalidInput(format!(
                "Operator size ({a_size}) doesn't match RHS ({n})"
            )));
        }

        let m = self.restart_dim;
        let mut guard = self.workspace.lock().unwrap();
        if guard.as_ref().is_none_or(|ws| ws.v.shape() != [n, m + 1]) {
            *guard = Some(Workspace {
                v: Array2::zeros([n, m + 1]),
                h: Array2::zeros([m + 1, m]),
                g: Array1::zeros([m + 1]),
                c: Array1::zeros([m]),
                s: Array1::zeros([m]),
                basis_work: Array1::zeros([n]),
                work: Array1::zeros([n]),
                precond_work: Array1::zeros([n]),
                ax: Array1::zeros([n]),
            });
        }
        let ws = guard.as_mut().unwrap();

        // Initial residual.
        a.apply(x, &mut ws.ax)?;
        let mut r0 = b.clone();
        sub_assign(&mut r0, &ws.ax);

        // Preconditioned initial residual.
        precond.apply_to(&r0, &mut ws.work)?;
        let beta = vec_norm(&ws.work);

        let r0_norm = vec_norm(&r0);
        if r0_norm < self.config.tolerance {
            return Ok(ConvergenceMonitor::new(r0_norm));
        }
        if beta <= <T as RealField>::EPSILON {
            return Err(LetoError::NumericalBreakdown(
                "GMRES: preconditioned initial residual is numerically zero".into(),
            ));
        }

        let mut monitor = ConvergenceMonitor::new(beta);
        let mut iters_used = 0usize;
        let mut first_restart = true;

        while iters_used < self.config.max_iterations {
            let beta_cur = if first_restart {
                beta
            } else {
                a.apply(x, &mut ws.ax)?;
                let mut r_restart = b.clone();
                sub_assign(&mut r_restart, &ws.ax);
                precond.apply_to(&r_restart, &mut ws.work)?;
                vec_norm(&ws.work)
            };

            if beta_cur <= <T as RealField>::EPSILON {
                return Err(LetoError::NumericalBreakdown(
                    "GMRES: restart residual is numerically zero".into(),
                ));
            }

            // Normalize first basis vector.
            let inv_beta = <T as NumericElement>::ONE / beta_cur;
            for row in 0..n {
                ws.v[[row, 0]] = ws.work[row] * inv_beta;
            }

            fill_mat(&mut ws.h, <T as NumericElement>::ZERO);
            fill_vec(&mut ws.g, <T as NumericElement>::ZERO);
            fill_vec(&mut ws.c, <T as NumericElement>::ZERO);
            fill_vec(&mut ws.s, <T as NumericElement>::ZERO);
            ws.g[0] = beta_cur;

            let remaining = self.config.max_iterations - iters_used;
            let inner = m.min(remaining);
            let mut converged_at = None;

            for k in 0..inner {
                // Split mutable borrows: pass only what Arnoldi needs.
                let Workspace {
                    v, h, basis_work, work, precond_work, ..
                } = ws;
                arnoldi::arnoldi_step(
                    a,
                    v,
                    h,
                    k,
                    n,
                    basis_work,
                    work,
                    Some(precond),
                    Some(precond_work),
                )?;
                iters_used += 1;

                givens::apply_previous_rotations(&mut ws.h, &ws.c, &ws.s, k);
                let (ck, sk) = givens::compute_rotation(ws.h[[k, k]], ws.h[[k + 1, k]]);
                ws.c[k] = ck;
                ws.s[k] = sk;
                givens::apply_new_rotation(&mut ws.h, &mut ws.g, ck, sk, k);

                let res_est = NumericElement::abs(ws.g[k + 1]);
                monitor.record_residual(res_est);
                if res_est < self.config.tolerance {
                    converged_at = Some(k + 1);
                    break;
                }
            }

            let k_final = converged_at.unwrap_or(inner);
            if k_final == 0 {
                break;
            }
            let y = givens::solve_upper_triangular(&ws.h, &ws.g, k_final)?;
            for i in 0..k_final {
                for row in 0..n {
                    x[row] += y[i] * ws.v[[row, i]];
                }
            }

            // Check true residual.
            a.apply(x, &mut ws.ax)?;
            let mut r_check = b.clone();
            sub_assign(&mut r_check, &ws.ax);
            if vec_norm(&r_check) < self.config.tolerance {
                return Ok(monitor);
            }

            if converged_at.is_some() {
                return Ok(monitor);
            }
            first_restart = false;
        }

        Err(LetoError::ConvergenceError {
            max_iters: self.config.max_iterations,
            residual: NumericElement::to_f64(
                *monitor.residual_history.last().unwrap_or(&<T as NumericElement>::ZERO),
            ),
            tol: NumericElement::to_f64(self.config.tolerance),
        })
    }

    /// Solve without preconditioning.
    pub fn solve_unpreconditioned<Op: LinearOperator<T> + ?Sized>(
        &self,
        a: &Op,
        b: &Array1<T>,
        x: &mut Array1<T>,
    ) -> Result<ConvergenceMonitor<T>> {
        self.solve_preconditioned(a, b, &IdentityPreconditioner, x)
    }
}

impl<T: RealField + Copy + FloatElement + Debug> Configurable<T> for GMRES<T> {
    type Config = IterativeSolverConfig<T>;
    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<T: RealField + Debug + Copy + FloatElement> IterativeLinearSolver<T> for GMRES<T> {
    fn solve<Op: LinearOperator<T> + ?Sized, P: Preconditioner<T>>(
        &self,
        a: &Op,
        b: &Array1<T>,
        x: &mut Array1<T>,
        preconditioner: Option<&P>,
    ) -> Result<ConvergenceMonitor<T>> {
        if let Some(p) = preconditioner {
            self.solve_preconditioned(a, b, p, x)
        } else {
            self.solve_unpreconditioned(a, b, x)
        }
    }
}

impl<T: RealField + Copy + FloatElement + Debug> LinearSolver<T> for GMRES<T> {
    fn solve_system(
        &self,
        a: &dyn LinearOperator<T>,
        b: &Array1<T>,
        x0: Option<&Array1<T>>,
    ) -> Result<Array1<T>> {
        let mut x = x0.cloned().unwrap_or_else(|| Array1::zeros(b.shape()));
        self.solve(a, b, &mut x, None::<&IdentityPreconditioner>)?;
        Ok(x)
    }
}
