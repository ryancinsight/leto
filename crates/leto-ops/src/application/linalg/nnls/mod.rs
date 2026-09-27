//! Non-Negative Least Squares (NNLS) — Lawson & Hanson active-set algorithm.
//!
//! Solves `min ‖A·x − b‖₂  subject to  x ≥ 0` for dense matrices.
//! This is the canonical solver for constrained spherical deconvolution (CSD)
//! in diffusion MRI, where the fibre orientation distribution must be
//! non-negative at every direction.
//!
//! # Algorithm
//!
//! Lawson & Hanson (1974), Chapter 23.  The active-set method maintains a
//! partition of variable indices into the *passive* set `P` (indices where
//! `x_j` is allowed to be non-zero) and the active set (indices forced to
//! zero).  At each outer iteration the index with the largest Lagrange
//! multiplier (negative gradient) joins `P`, an unconstrained least-squares
//! problem is solved on `P`, and any variables that become negative are
//! removed from `P` via linear interpolation until all `x_j ≥ 0`.
//!
//! # References
//!
//! - Lawson, C. L. & Hanson, R. J. (1974). *Solving Least Squares Problems.*
//!   Prentice-Hall.  Chapter 23.
//! - Bro, R. & de Jong, S. (1997). A fast non-negativity-constrained least
//!   squares algorithm.  *J. Chemometrics* 11(5), 393–401.

mod active_set;

pub use active_set::{nnls, NnlsConfig, NnlsResult};

#[cfg(test)]
mod tests;
