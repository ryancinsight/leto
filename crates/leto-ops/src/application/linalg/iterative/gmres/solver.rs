//! GMRES(m) solver with Arnoldi iteration and Givens rotations.
//!
//! Solves general non-symmetric linear systems `A·x = b` using the
//! Generalized Minimal Residual method with restarts and left preconditioning.
//!
//! ## Algorithm sketch
//!
//! 1. Build an orthonormal Krylov basis `V_m` of `K_m(M⁻¹A, M⁻¹r₀)` via
//!    Arnoldi / modified Gram-Schmidt.
//! 2. Maintain the least-squares problem `min ‖β e₁ − H̄_m y‖` incrementally
//!    through Givens rotations applied to the growing Hessenberg matrix, so
//!    `|g[k+1]|` is the minimised residual norm at every step.
//! 3. When the restart dimension `m` is reached, or the estimate meets the
//!    threshold, form `x ← x + V_k y` and recompute the true residual.
//! 4. Restart from that residual.
//!
//! ## Convergence criterion
//!
//! Termination is decided on the **true, unpreconditioned** residual
//! `‖b − A·x‖₂` against [`IterativeSolverConfig::threshold`]. The Arnoldi
//! recurrence only yields `‖M⁻¹(b − A·x)‖₂`, which differs from the true
//! residual by up to `κ(M)`; accepting the preconditioned estimate as proof of
//! convergence would report success on an unconverged solution whenever `M` is
//! ill-conditioned. The estimate ends a cycle early, it never declares
//! convergence.
//!
//! ## Optimality theorem (Saad and Schultz 1986)
//!
//! GMRES minimises ‖r_k‖₂ over `x₀ + K_k(A, r₀)`, i.e. it is the best
//! Krylov method without restarts.
//!
//! ## References
//! - Saad and Schultz (1986). *GMRES: A generalized minimal residual algorithm.*
//!   SIAM J. Sci. Stat. Comput. 7(3), 856–869.
//! - Saad (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed., §6.5.

use super::super::config::IterativeSolverConfig;
use super::super::convergence::ConvergenceMonitor;
use super::super::preconditioners::IdentityPreconditioner;
use super::super::traits::{
    Configurable, IterativeLinearSolver, LinearOperator, LinearSolver, Preconditioner,
};
use super::arnoldi::{self, ArnoldiOutcome};
use super::{flat, flat_mut, flat2, flat2_mut, givens};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, Array2, LetoError, Result};
use std::fmt::Debug;
use std::sync::{Mutex, PoisonError};

/// Solver scratch space, retained across solves so that a steady time-stepping
/// loop performs no allocation after the first call.
struct Workspace<T: RealField + Copy> {
    /// `m + 1` Krylov vectors, packed row-contiguously (see [`arnoldi`]).
    basis: Array2<T>,
    /// Transposed Hessenberg store: row `k` is column `k` of `H̄`.
    hessenberg: Array2<T>,
    /// Rotated right-hand side `Qᵀ β e₁`.
    g: Array1<T>,
    cosines: Array1<T>,
    sines: Array1<T>,
    /// Least-squares coefficients `y`.
    coefficients: Array1<T>,
    basis_work: Array1<T>,
    work: Array1<T>,
    precond_work: Array1<T>,
    ax: Array1<T>,
    residual: Array1<T>,
    update: Array1<T>,
}

impl<T: RealField + Copy> Workspace<T> {
    fn new(n: usize, m: usize) -> Self {
        Self {
            basis: Array2::zeros([m + 1, n]),
            hessenberg: Array2::zeros([m, m + 1]),
            g: Array1::zeros([m + 1]),
            cosines: Array1::zeros([m]),
            sines: Array1::zeros([m]),
            coefficients: Array1::zeros([m]),
            basis_work: Array1::zeros([n]),
            work: Array1::zeros([n]),
            precond_work: Array1::zeros([n]),
            ax: Array1::zeros([n]),
            residual: Array1::zeros([n]),
            update: Array1::zeros([n]),
        }
    }

    fn matches(&self, n: usize, m: usize) -> bool {
        self.basis.shape() == [m + 1, n]
    }
}

/// GMRES(m) solver with restarts and optional left preconditioning.
///
/// `restart_dim` (default: 30) bounds the Krylov subspace size between
/// restarts; larger values reduce restarts at the cost of `O(n·m)` memory and
/// `O(n·m²)` orthogonalisation work per cycle.
pub struct GMRES<T: RealField + Copy> {
    config: IterativeSolverConfig<T>,
    restart_dim: usize,
    workspace: Mutex<Option<Workspace<T>>>,
}

impl<T: RealField + Copy + FloatElement + Debug> GMRES<T> {
    /// Create with explicit configuration and restart dimension.
    ///
    /// # Panics
    /// Panics if `restart_dim == 0`, which is a construction-time programmer
    /// error: GMRES(0) builds no Krylov subspace and can never make progress.
    pub fn new(config: IterativeSolverConfig<T>, restart_dim: usize) -> Self {
        assert!(restart_dim > 0, "GMRES restart dimension must be positive");
        Self {
            config,
            restart_dim,
            workspace: Mutex::new(None),
        }
    }

    /// Create with default configuration and `restart_dim = 30`.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::new(IterativeSolverConfig::default(), 30)
    }

    /// Solve with an explicit preconditioner.
    ///
    /// # Errors
    /// Returns [`LetoError::InvalidInput`] on a dimension mismatch,
    /// [`LetoError::NumericalBreakdown`] when the recurrence produces a
    /// non-finite or singular state, and [`LetoError::ConvergenceError`] when
    /// the iteration budget is exhausted before the residual threshold is met.
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
                "Operator size ({a_size}) does not match RHS ({n})"
            )));
        }
        if x.shape()[0] != n {
            return Err(LetoError::InvalidInput(format!(
                "Solution length ({}) does not match RHS ({n})",
                x.shape()[0]
            )));
        }

        let m = self.restart_dim;
        // Workspace poisoning carries no meaning here: the buffers are pure
        // scratch, fully overwritten before every read, so a panic in an
        // earlier solve leaves nothing to salvage or corrupt.
        let mut guard = self
            .workspace
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if guard.as_ref().is_none_or(|ws| !ws.matches(n, m)) {
            *guard = Some(Workspace::new(n, m));
        }
        let Some(ws) = guard.as_mut() else {
            unreachable!("invariant: the workspace was just installed")
        };

        let threshold = self.config.threshold(norm_of(b));
        let mut residual_norm = recompute_residual(a, b, x, &mut ws.ax, &mut ws.residual)?;
        if !residual_norm.is_finite() {
            return Err(LetoError::NumericalBreakdown(
                "GMRES: initial residual is not finite".into(),
            ));
        }
        let mut monitor = ConvergenceMonitor::new(residual_norm);
        if residual_norm <= threshold {
            return Ok(monitor);
        }

        let mut iterations = 0usize;
        while iterations < self.config.max_iterations {
            // The preconditioned residual seeds the Krylov basis.
            precond.apply_to(&ws.residual, &mut ws.work)?;
            let beta = norm_of(&ws.work);
            if !beta.is_finite() {
                return Err(LetoError::NumericalBreakdown(
                    "GMRES: preconditioned residual is not finite".into(),
                ));
            }
            if beta == <T as NumericElement>::ZERO {
                return Err(LetoError::NumericalBreakdown(
                    "GMRES: preconditioner maps a non-zero residual to zero (singular M)".into(),
                ));
            }

            {
                let inverse_beta = <T as NumericElement>::ONE / beta;
                let source = flat("work vector", &ws.work);
                let basis = flat2_mut("Krylov basis", &mut ws.basis);
                for (target, &value) in basis[..n].iter_mut().zip(source.iter()) {
                    *target = value * inverse_beta;
                }
            }
            fill(flat2_mut("Hessenberg", &mut ws.hessenberg));
            fill(flat_mut("cosines", &mut ws.cosines));
            fill(flat_mut("sines", &mut ws.sines));
            let g = flat_mut("rotated RHS", &mut ws.g);
            fill(g);
            g[0] = beta;

            let cycle_limit = m.min(self.config.max_iterations - iterations);
            let mut vectors_used = 0usize;
            let mut happy_breakdown = false;

            for k in 0..cycle_limit {
                let stride = m + 1;
                let Workspace {
                    basis,
                    hessenberg,
                    g,
                    cosines,
                    sines,
                    basis_work,
                    work,
                    precond_work,
                    ..
                } = ws;
                let basis = flat2_mut("Krylov basis", basis);
                let hessenberg = flat2_mut("Hessenberg", hessenberg);
                let (before, column) = hessenberg.split_at_mut(k * stride);
                debug_assert!(
                    before.len() == k * stride,
                    "invariant: Hessenberg is packed"
                );
                let column = &mut column[..stride];

                let outcome = arnoldi::arnoldi_step(
                    a,
                    precond,
                    basis,
                    column,
                    k,
                    n,
                    basis_work,
                    work,
                    precond_work,
                )?;
                iterations += 1;
                vectors_used = k + 1;
                match outcome {
                    ArnoldiOutcome::NonFinite => {
                        return Err(LetoError::NumericalBreakdown(
                            "GMRES: Arnoldi recurrence produced a non-finite value".into(),
                        ));
                    }
                    ArnoldiOutcome::HappyBreakdown => happy_breakdown = true,
                    ArnoldiOutcome::Extended(_) => {}
                }

                let cosines = flat_mut("cosines", cosines);
                let sines = flat_mut("sines", sines);
                let g = flat_mut("rotated RHS", g);
                givens::apply_previous_rotations(column, cosines, sines, k);
                let (cosine, sine) = givens::compute_rotation(column[k], column[k + 1])?;
                cosines[k] = cosine;
                sines[k] = sine;
                givens::apply_new_rotation(column, g, cosine, sine, k);

                let estimate = NumericElement::abs(g[k + 1]);
                if !estimate.is_finite() {
                    return Err(LetoError::NumericalBreakdown(
                        "GMRES: residual estimate is not finite".into(),
                    ));
                }
                monitor.record_residual(estimate);
                if estimate <= threshold || happy_breakdown {
                    break;
                }
            }

            if vectors_used == 0 {
                break;
            }

            {
                let stride = m + 1;
                let Workspace {
                    hessenberg,
                    g,
                    coefficients,
                    basis,
                    update,
                    ..
                } = ws;
                let hessenberg = flat2("Hessenberg", hessenberg);
                let g = flat("rotated RHS", g);
                let coefficients = flat_mut("coefficients", coefficients);
                givens::solve_upper_triangular(hessenberg, stride, g, coefficients, vectors_used)?;

                // Accumulate `V_k · y` in a contiguous buffer, then fold it into
                // the caller-owned `x` in one pass: the `O(n·k)` work stays
                // unit-stride while `x` keeps whatever layout the caller chose.
                let basis = flat2("Krylov basis", basis);
                let update = flat_mut("solution update", update);
                fill(update);
                for (i, &coefficient) in coefficients[..vectors_used].iter().enumerate() {
                    let basis_i = &basis[i * n..(i + 1) * n];
                    for (target, &value) in update.iter_mut().zip(basis_i.iter()) {
                        *target += coefficient * value;
                    }
                }
            }
            for i in 0..n {
                x[i] += ws.update[i];
            }

            // One operator application per cycle serves both the convergence
            // test and the next cycle Krylov seed.
            residual_norm = recompute_residual(a, b, x, &mut ws.ax, &mut ws.residual)?;
            if !residual_norm.is_finite() {
                return Err(LetoError::NumericalBreakdown(
                    "GMRES: residual is not finite".into(),
                ));
            }
            if residual_norm <= threshold {
                return Ok(monitor);
            }
            if happy_breakdown {
                // The subspace is invariant under M⁻¹A, so no further Krylov
                // vector exists and restarting would reproduce this cycle.
                return Err(LetoError::NumericalBreakdown(format!(
                    "GMRES: Krylov subspace exhausted at residual {} above threshold {}",
                    NumericElement::to_f64(residual_norm),
                    NumericElement::to_f64(threshold)
                )));
            }
        }

        Err(LetoError::ConvergenceError {
            max_iters: self.config.max_iterations,
            residual: NumericElement::to_f64(residual_norm),
            tol: NumericElement::to_f64(threshold),
        })
    }

    /// Solve without preconditioning.
    ///
    /// # Errors
    /// See [`Self::solve_preconditioned`].
    pub fn solve_unpreconditioned<Op: LinearOperator<T> + ?Sized>(
        &self,
        a: &Op,
        b: &Array1<T>,
        x: &mut Array1<T>,
    ) -> Result<ConvergenceMonitor<T>> {
        self.solve_preconditioned(a, b, &IdentityPreconditioner, x)
    }
}

#[inline]
fn fill<T: NumericElement>(target: &mut [T]) {
    target.fill(T::ZERO);
}

#[inline]
fn norm_of<T: NumericElement>(v: &Array1<T>) -> T {
    let mut sum = T::ZERO;
    for i in 0..v.shape()[0] {
        sum += v[i] * v[i];
    }
    sum.sqrt()
}

/// `residual ← b − A·x`, returning `‖residual‖₂`.
fn recompute_residual<T, Op>(
    a: &Op,
    b: &Array1<T>,
    x: &Array1<T>,
    ax: &mut Array1<T>,
    residual: &mut Array1<T>,
) -> Result<T>
where
    T: RealField + Copy,
    Op: LinearOperator<T> + ?Sized,
{
    a.apply(x, ax)?;
    let mut sum = <T as NumericElement>::ZERO;
    for i in 0..b.shape()[0] {
        let value = b[i] - ax[i];
        residual[i] = value;
        sum += value * value;
    }
    Ok(sum.sqrt())
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
