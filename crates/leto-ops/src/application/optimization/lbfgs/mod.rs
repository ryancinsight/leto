//! Limited-memory BFGS (L-BFGS) quasi-Newton optimiser.
//!
//! L-BFGS approximates the inverse Hessian from the last `m` gradient/step pairs
//! `(s_k, y_k)` and computes a descent direction by the Nocedal two-loop
//! recursion, giving super-linear convergence without storing or inverting a
//! dense Hessian. It is the standard refinement step for full-waveform inversion
//! and PINN training (Inverse Problems §9.1).
//!
//! The driver loop is host-side by design: line search, history updates, and
//! convergence tests are sequential. A GPU path composes the inner vector
//! operations from existing device operators rather than a dedicated solver
//! trait, so L-BFGS has no GPU counterpart of its own.
//!
//! # References
//! - Nocedal, J. (1980). "Updating quasi-Newton matrices with limited storage."
//!   *Math. Comp.*, 35(151), 773–782.
//! - Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.), Alg. 7.4–7.5.

mod solver;

pub use solver::{minimize, LbfgsConfig, LbfgsMemory, LbfgsResult};

#[cfg(test)]
mod tests;
