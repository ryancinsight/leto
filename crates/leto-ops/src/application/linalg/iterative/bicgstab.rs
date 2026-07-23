//! BiCGSTAB (BiConjugate Gradient Stabilized) solver.
//!
//! Solves non-symmetric linear systems `A·x = b`.  BiCGSTAB stabilises
//! BiCG by combining it with a GMRES-like smoothing step, giving smoother
//! convergence and better robustness against erratic behaviour.
//!
//! ## Convergence
//!
//! BiCGSTAB converges for matrices whose field of values lies in the right
//! half-plane.  Two breakdown conditions are handled:
//! 1. **ρ-breakdown**: ρ_new = 0 (biorthogonality lost).
//! 2. **ω-breakdown**: ω = 0 (smoothing step stagnates).
//!
//! ## References
//! - Van der Vorst (1992). *Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG.*
//!   SIAM J. Sci. Stat. Comput. 13(2), 631–644.

use super::config::IterativeSolverConfig;
use super::convergence::ConvergenceMonitor;
use super::ops::{assign_residual, axpy, dot, norm, validate_len};
use super::preconditioners::IdentityPreconditioner;
use super::traits::{
    Configurable, IterativeLinearSolver, LinearOperator, LinearSolver, Preconditioner,
};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, LetoError, Result};
use std::fmt::Debug;

/// BiCGSTAB solver.
pub struct BiCGSTAB<T: RealField + Copy> {
    config: IterativeSolverConfig<T>,
}

#[inline]
fn is_finite<T: NumericElement>(x: T) -> bool {
    NumericElement::to_f64(x).is_finite()
}

impl<T: RealField + Copy + NumericElement> BiCGSTAB<T> {
    /// Create with explicit configuration.
    pub const fn new(config: IterativeSolverConfig<T>) -> Self {
        Self { config }
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
        precond: &P,
        x: &mut Array1<T>,
    ) -> Result<ConvergenceMonitor<T>> {
        let n = b.shape()[0];
        validate_len("BiCGSTAB solution", x, n)?;
        let a_size = a.size();
        if a_size != 0 && a_size != n {
            return Err(LetoError::InvalidInput(format!(
                "Operator size ({a_size}) doesn't match RHS ({n})"
            )));
        }

        let zero = <T as NumericElement>::ZERO;
        let one = <T as NumericElement>::ONE;

        // r = b − A·x₀
        let mut ax = Array1::zeros([n]);
        a.apply(x, &mut ax)?;
        let mut r = Array1::zeros([n]);
        assign_residual(&mut r, b, &ax);

        let r0 = norm(&r);
        let mut monitor = ConvergenceMonitor::new(r0);
        if r0 < self.config.tolerance {
            return Ok(monitor);
        }

        // r_hat = r  (shadow residual, never updated)
        let r_hat = r.clone();

        let mut rho_prev = one;
        let mut alpha = one;
        let mut omega = one;

        let mut v = Array1::zeros([n]);
        let mut p = Array1::zeros([n]);
        let mut y = Array1::zeros([n]);
        let mut z = Array1::zeros([n]);
        let mut s = Array1::zeros([n]);
        let mut t = Array1::zeros([n]);
        let mut kt = Array1::zeros([n]);

        for _iter in 0..self.config.max_iterations {
            let rho = dot(&r_hat, &r);
            if !is_finite(rho) || rho == zero {
                return Err(LetoError::NumericalBreakdown(
                    "BiCGSTAB: rho breakdown (shadow residual orthogonal to residual)".into(),
                ));
            }

            let beta = (rho / rho_prev) * (alpha / omega);
            if !is_finite(beta) {
                return Err(LetoError::NumericalBreakdown(
                    "BiCGSTAB: beta is non-finite".into(),
                ));
            }

            // p = r + beta * (p − omega * v)
            for i in 0..n {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }

            // y = M⁻¹ · p
            precond.apply_to(&p, &mut y)?;
            // v = A · y
            a.apply(&y, &mut v)?;

            let r_hat_v = dot(&r_hat, &v);
            if !is_finite(r_hat_v) || r_hat_v == zero {
                return Err(LetoError::NumericalBreakdown(
                    "BiCGSTAB: r_hat · v ≈ 0 (alpha breakdown)".into(),
                ));
            }
            alpha = rho / r_hat_v;
            if !is_finite(alpha) {
                return Err(LetoError::NumericalBreakdown(
                    "BiCGSTAB: alpha is non-finite".into(),
                ));
            }

            // s = r − alpha * v
            for i in 0..n {
                s[i] = r[i] - alpha * v[i];
            }

            let s_norm = norm(&s);
            monitor.record_residual(s_norm);
            if s_norm < self.config.tolerance {
                axpy(x, alpha, &y);
                return Ok(monitor);
            }

            // z = M⁻¹ · s
            precond.apply_to(&s, &mut z)?;
            // t = A · z
            a.apply(&z, &mut t)?;
            // kt = M⁻¹ · t
            precond.apply_to(&t, &mut kt)?;

            let kt_s = dot(&kt, &z);
            let kt_kt = dot(&kt, &kt);
            if !is_finite(kt_kt) || kt_kt == zero {
                return Err(LetoError::NumericalBreakdown(
                    "BiCGSTAB: omega breakdown (‖t‖ = 0)".into(),
                ));
            }
            omega = kt_s / kt_kt;
            if !is_finite(omega) {
                return Err(LetoError::NumericalBreakdown(
                    "BiCGSTAB: omega is non-finite".into(),
                ));
            }

            // x = x + alpha * y + omega * z
            axpy(x, alpha, &y);
            axpy(x, omega, &z);

            // r = s − omega * t
            for i in 0..n {
                r[i] = s[i] - omega * t[i];
            }

            let res = norm(&r);
            monitor.record_residual(res);
            if res < self.config.tolerance {
                return Ok(monitor);
            }

            rho_prev = rho;
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

impl<T: RealField + Debug + Copy + NumericElement> Configurable<T> for BiCGSTAB<T> {
    type Config = IterativeSolverConfig<T>;
    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<T: RealField + Debug + Copy + NumericElement> IterativeLinearSolver<T> for BiCGSTAB<T> {
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

impl<T: RealField + Copy + FloatElement + NumericElement + Debug> LinearSolver<T> for BiCGSTAB<T> {
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
