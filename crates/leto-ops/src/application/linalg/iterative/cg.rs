//! Preconditioned Conjugate Gradient (PCG) solver.
//!
//! Solves symmetric positive-definite (SPD) linear systems `A·x = b` using
//! the Hestenes–Stiefel Conjugate Gradient algorithm with optional
//! left preconditioning.
//!
//! ## Convergence theorem (A-norm error bound)
//!
//! For an SPD matrix A ∈ ℝⁿˣⁿ with condition number κ = λ_max / λ_min the
//! CG iterates satisfy:
//!
//! ```text
//! ‖x_k − x*‖_A  ≤  2 ((√κ − 1) / (√κ + 1))^k ‖x_0 − x*‖_A
//! ```
//!
//! ## References
//! - Hestenes & Stiefel (1952). *Methods of conjugate gradients for solving linear equations.*
//! - Saad (2003). *Iterative Methods for Sparse Linear Systems*, §6.7.

use super::config::IterativeSolverConfig;
use super::convergence::ConvergenceMonitor;
use super::ops::{assign_residual, axpy, copy_vec, dot, norm, scale_add, validate_len};
use super::preconditioners::IdentityPreconditioner;
use super::traits::{
    Configurable, IterativeLinearSolver, LinearOperator, LinearSolver, Preconditioner,
};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, LetoError, Result};
use std::fmt::Debug;
use std::sync::Mutex;

/// Per-solve scratch allocation.
struct Workspace<T> {
    r: Array1<T>,
    p: Array1<T>,
    z: Array1<T>,
    ap: Array1<T>,
    ax: Array1<T>,
}

/// Preconditioned Conjugate Gradient solver.
pub struct ConjugateGradient<T: RealField + Copy> {
    config: IterativeSolverConfig<T>,
    workspace: Mutex<Option<Workspace<T>>>,
}

impl<T: RealField + Copy + NumericElement> ConjugateGradient<T> {
    /// Create with explicit configuration.
    pub const fn new(config: IterativeSolverConfig<T>) -> Self {
        Self { config, workspace: Mutex::new(None) }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn default() -> Self
    where
        T: FloatElement,
    {
        Self::new(IterativeSolverConfig::default())
    }

    /// Solve `A·x = b` with an explicit preconditioner.
    pub fn solve_preconditioned<Op: LinearOperator<T> + ?Sized, P: Preconditioner<T>>(
        &self,
        a: &Op,
        b: &Array1<T>,
        preconditioner: &P,
        x: &mut Array1<T>,
    ) -> Result<ConvergenceMonitor<T>> {
        let n = b.shape()[0];
        validate_len("CG solution", x, n)?;
        let a_size = a.size();
        if a_size != 0 && a_size != n {
            return Err(LetoError::InvalidInput(format!(
                "Operator size ({a_size}) doesn't match RHS vector ({n})"
            )));
        }

        // Allocate or resize workspace.
        {
            let mut guard = self.workspace.lock().unwrap();
            if guard.as_ref().is_none_or(|ws| ws.r.shape()[0] != n) {
                *guard = Some(Workspace {
                    r: Array1::zeros([n]),
                    p: Array1::zeros([n]),
                    z: Array1::zeros([n]),
                    ap: Array1::zeros([n]),
                    ax: Array1::zeros([n]),
                });
            }
        }

        // Initial residual r = b − A·x
        let (r0_norm, mut rz) = {
            let mut guard = self.workspace.lock().unwrap();
            let ws = guard.as_mut().unwrap();
            a.apply(x, &mut ws.ax)?;
            assign_residual(&mut ws.r, b, &ws.ax);
            let r0 = norm(&ws.r);
            if r0 < self.config.tolerance {
                return Ok(ConvergenceMonitor::new(r0));
            }
            preconditioner.apply_to(&ws.r, &mut ws.z)?;
            copy_vec(&ws.z, &mut ws.p);
            let rz = dot(&ws.r, &ws.z);
            (r0, rz)
        };

        let eps = <T as RealField>::EPSILON;
        let bd_tol = eps * eps;

        // Validate initial r·z.
        {
            let guard = self.workspace.lock().unwrap();
            let ws = guard.as_ref().unwrap();
            let rz_scale = norm(&ws.r) * norm(&ws.z);
            if NumericElement::abs(rz) < bd_tol * (<T as NumericElement>::ONE + rz_scale) {
                return Err(LetoError::NumericalBreakdown("CG: initial r·z ≈ 0".into()));
            }
            if rz < <T as NumericElement>::ZERO {
                return Err(LetoError::NotPositiveDefinite {
                    detail: "CG: initial r·z < 0 — operator is not SPD".into(),
                });
            }
        }

        let mut monitor = ConvergenceMonitor::new(r0_norm);

        for _iter in 0..self.config.max_iterations {
            // Clone p so we can pass it immutably to `a.apply` while mutating `ws.ap`.
            let p_buf = {
                let guard = self.workspace.lock().unwrap();
                guard.as_ref().unwrap().p.clone()
            };

            let mut guard = self.workspace.lock().unwrap();
            let ws = guard.as_mut().unwrap();
            a.apply(&p_buf, &mut ws.ap)?;

            let pap = dot(&p_buf, &ws.ap);
            let pap_scale = norm(&p_buf) * norm(&ws.ap);
            if NumericElement::abs(pap) < bd_tol * (<T as NumericElement>::ONE + pap_scale) {
                return Err(LetoError::NumericalBreakdown("CG: p·Ap ≈ 0 — breakdown".into()));
            }
            if pap < <T as NumericElement>::ZERO {
                return Err(LetoError::NotPositiveDefinite {
                    detail: "CG: p·Ap < 0 — operator is not SPD".into(),
                });
            }

            let alpha = rz / pap;
            axpy(x, alpha, &p_buf);
            axpy(&mut ws.r, -alpha, &ws.ap);

            let res = norm(&ws.r);
            monitor.record_residual(res);
            if res < self.config.tolerance {
                return Ok(monitor);
            }

            preconditioner.apply_to(&ws.r, &mut ws.z)?;

            let rz_new = dot(&ws.r, &ws.z);
            if NumericElement::to_f64(rz_new).is_nan() {
                return Err(LetoError::NumericalBreakdown("CG: rz_new is NaN".into()));
            }
            if rz_new < <T as NumericElement>::ZERO {
                return Err(LetoError::NotPositiveDefinite {
                    detail: "CG: rz_new < 0 — operator lost SPD".into(),
                });
            }

            let beta = rz_new / rz;
            scale_add(&mut ws.p, beta, &ws.z);
            rz = rz_new;
        }

        let final_res = {
            let guard = self.workspace.lock().unwrap();
            let ws = guard.as_ref().unwrap();
            NumericElement::to_f64(norm(&ws.r))
        };

        Err(LetoError::ConvergenceError {
            max_iters: self.config.max_iterations,
            residual: final_res,
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

impl<T: RealField + Debug + Copy + NumericElement> Configurable<T> for ConjugateGradient<T> {
    type Config = IterativeSolverConfig<T>;
    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<T: RealField + Debug + Copy + NumericElement> IterativeLinearSolver<T>
    for ConjugateGradient<T>
{
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

impl<T: RealField + Copy + FloatElement + NumericElement + Debug> LinearSolver<T>
    for ConjugateGradient<T>
{
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
