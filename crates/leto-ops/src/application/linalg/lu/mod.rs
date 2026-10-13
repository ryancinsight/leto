//! LU decomposition with partial pivoting: `P · A = L · U`.
//!
//! Vertical structure (SoC, mirroring `qr/{decompose, solve}`): the `decompose`
//! leaf owns the factorization driver ([`lu_decompose`]) and the blocked-path
//! gate; the `eliminate` leaf owns the elimination kernels; the `solvers`
//! leaf owns substitution and the one-call conveniences. This module owns the
//! [`LuDecomposition`] type and its materialization accessors.

pub mod decompose;
pub mod eliminate;
pub mod solvers;

pub use decompose::lu_decompose;
pub use solvers::{det, inv, solve};

#[cfg(test)]
mod tests;

use crate::domain::real::RealScalar;
use leto::Array2;

/// LU decomposition with partial pivoting: `P · A = L · U`.
///
/// `L` (unit lower-triangular) and `U` (upper-triangular) are packed into one
/// row-major `n × n` factor matrix (`L` strictly below the diagonal, `U` on
/// and above it); `pivots` records the row permutation; `sign` is the
/// permutation parity (`+1`/`-1`), used by [`det`](Self::det).
///
/// Generic over `T: RealScalar`; elimination runs in the native precision of
/// `T` per the `Scalar` contract (no hidden widening — a caller needing a
/// wider working precision converts the input explicitly). Driver: CFDrs
/// `cfd-math` dense solver paths (leto replacement Stage A1).
#[derive(Debug, Clone)]
pub struct LuDecomposition<T> {
    pub(super) factors: Array2<T>,
    pub(super) pivots: Vec<usize>,
    pub(super) sign: i8,
}

impl<T: RealScalar> LuDecomposition<T> {
    /// Construct an LU decomposition directly from its raw components.
    #[must_use]
    #[inline]
    pub fn from_raw_parts(factors: Array2<T>, pivots: Vec<usize>, sign: i8) -> Self {
        Self {
            factors,
            pivots,
            sign,
        }
    }

    /// Matrix dimension `n`.
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.pivots.len()
    }

    /// The packed `L`/`U` factor matrix (unit `L` strictly below the
    /// diagonal, `U` on and above it), in row-major order.
    #[must_use]
    #[inline]
    pub fn factors(&self) -> &Array2<T> {
        &self.factors
    }

    /// The permutation pivots vector.
    #[must_use]
    #[inline]
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }

    /// Determinant of the original matrix: parity × product of `U`'s diagonal.
    #[must_use]
    pub fn det(&self) -> T {
        let n = self.dim();
        let mut det = if self.sign >= 0 {
            T::ONE
        } else {
            T::ZERO.sub(T::ONE)
        };
        for k in 0..n {
            det = det.mul(*self.factors.get([k, k]).expect("diagonal in bounds"));
        }
        det
    }
}
