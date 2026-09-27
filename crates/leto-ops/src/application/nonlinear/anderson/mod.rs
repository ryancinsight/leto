//! Anderson Acceleration (Extrapolation) for Fixed-Point Iterations
//!
//! # Theorem — Anderson Acceleration Convergence (Anderson, 1965)
//!
//! Let $G: \mathbb{R}^n \to \mathbb{R}^n$ be a contractive mapping with fixed point
//! $x^* = G(x^*)$. Anderson Acceleration computes an accelerated update:
//!
//! $$ \mathbf{x}_{k+1} = \mathbf{x}_k + \beta \mathbf{f}_k - (\Delta \mathbf{X}_k + \beta \Delta \mathbf{F}_k) \gamma $$
//!
//! where $\mathbf{f}_k = G(\mathbf{x}_k) - \mathbf{x}_k$ is the residual, $\Delta \mathbf{X}_k$ and
//! $\Delta \mathbf{F}_k$ are matrices of the last $m$ step differences, and $\gamma$ solves:
//!
//! $$ \min_\gamma \|\mathbf{f}_k - \Delta \mathbf{F}_k \gamma\|_2 $$
//!
//! Locally achieves superlinear convergence without an explicit Jacobian.
//!
//! # Theorem — MGS-QR Anderson vs Normal Equations (Walker & Ni 2011, Thm 2.1)
//!
//! When solving $\min_\gamma \|f - \Delta F \gamma\|_2$, two approaches are possible:
//!
//! **Normal equations** (Type-I): $({\Delta F}^T \Delta F)\gamma = {\Delta F}^T f$.
//! - Condition number: $\kappa(\Delta F^T \Delta F) = \kappa(\Delta F)^2$
//! - Numerically unstable when $\Delta F$ columns are nearly linearly dependent.
//!
//! **QR factorization** (Type-II): $\Delta F = QR \Rightarrow R\gamma = Q^T f$.
//! - Condition number: $\kappa(R) = \kappa(\Delta F)$
//! - Stable: MGS-QR halves the sensitivity to near-linear-dependence in history.
//!
//! **Proof sketch**: For $\Delta F = QR$ (thin QR), the unique least-squares solution
//! is $\gamma^* = R^{-1} Q^T f$ whenever $\Delta F$ has full column rank. The
//! backward-stable MGS process produces $\|Q^T Q - I\| = O(\epsilon_{\rm mach} \kappa(\Delta F))$.
//! The normal equations approach amplifies this error by $\kappa(\Delta F)$, giving
//! $O(\epsilon_{\rm mach} \kappa(\Delta F)^2)$ rounding error in $\gamma^*$.
//!
//! **Reference**: Walker, H.F. & Ni, P. (2011). Anderson acceleration for fixed-point
//! iterations. *SIAM J. Numer. Anal.* 49(4):1715–1735.
//!
//! # Theorem — VecDeque O(1) history eviction (GAP-PERF-004)
//!
//! History eviction via `Vec::remove(0)` is O(m) (memory shift of m vectors).
//! `VecDeque::pop_front()` is O(1) (pointer rotation on ring buffer).
//! For history depth m=5 and 10³ outer iterations, total shift cost drops from
//! O(5 × 10³) = 5000 ops to O(10³) = 1000 ops in pointer increments.

mod solver;

pub use solver::{AndersonAccelerator, AndersonConfig, AndersonMethod};

#[cfg(test)]
mod tests;
