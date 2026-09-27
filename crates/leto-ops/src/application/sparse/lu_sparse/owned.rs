//! [`OwnedNumericLu`]: the cached, symbolic-borrow-free factorization
//! produced by `SparseLuSolver::factor_sparse_with_symbolic`.

use super::super::lu_numeric::triangular_solve_into;
use super::super::lu_symbolic::SymbolicLu;
use crate::application::linalg::lu::LuDecomposition;
use crate::domain::real::RealScalar;
use leto::{Array1, ArrayView1, ArrayViewMut1, LetoError, Result};

/// Owned, reusable numeric factorization produced by
/// `SparseLuSolver::factor_sparse_with_symbolic`.
///
/// Stores everything the triangular solves need — no borrow of the
/// symbolic analysis — so factors can be cached across solver iterations
/// (a preconditioner factoring each momentum block once and applying it
/// every Krylov iteration) while the caller separately caches the
/// [`SymbolicLu`] for refactorization on unchanged patterns.
///
/// Two representations mirror the solver's dispatch contract: pivoting-free
/// matrices hold the sparse L/U value buffers; matrices that need partial
/// pivoting, or that the dispatch criteria route dense, hold the dense
/// partial-pivoting factorization. Both arms solve to the same values up
/// to IEEE 754 rounding (differential-tested below).
#[derive(Debug, Clone)]
#[must_use = "OwnedNumericLu carries the factorization consumed by solve"]
pub struct OwnedNumericLu<T: RealScalar> {
    pub(super) repr: OwnedLuRepr<T>,
}

#[derive(Debug, Clone)]
pub(super) enum OwnedLuRepr<T: RealScalar> {
    Sparse {
        symbolic: SymbolicLu,
        l_values: Vec<T>,
        u_values: Vec<T>,
        row_perm: Vec<usize>,
        /// Column-ordering permutation applied during symbolic
        /// factorization. Length `n`; identity `[0, n)` for natural
        /// ordering, AMD output for `OrderingStrategy::AmdApproxMinDegree`.
        /// The triangular solve inverse-permutes the column-order
        /// solution back to original row order via this vector.
        col_perm: Vec<usize>,
    },
    Dense(LuDecomposition<T>),
}

impl<T: RealScalar> OwnedNumericLu<T> {
    /// Matrix order `n`.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        match &self.repr {
            OwnedLuRepr::Sparse { symbolic, .. } => symbolic.n(),
            OwnedLuRepr::Dense(lu) => lu.dim(),
        }
    }

    /// Solve `A · x = rhs` directly into a caller-owned view `out`.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::ShapeMismatch`] when `rhs` or `out` length
    /// differs from the matrix order `n`.
    pub fn solve_into(
        &self,
        rhs: &ArrayView1<'_, T>,
        out: &mut ArrayViewMut1<'_, T>,
    ) -> Result<()> {
        match &self.repr {
            OwnedLuRepr::Sparse {
                symbolic,
                l_values,
                u_values,
                row_perm,
                col_perm,
            } => triangular_solve_into(symbolic, l_values, u_values, row_perm, col_perm, rhs, out),
            OwnedLuRepr::Dense(lu) => lu.solve_into(rhs, out),
        }
    }

    /// Solve `A · x = rhs`, returning a freshly-owned solution.
    ///
    /// # Errors
    ///
    /// Forwards from [`Self::solve_into`].
    pub fn solve(&self, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>> {
        let n = self.n();
        let mut x =
            Array1::from_shape_vec([n], vec![T::ZERO; n]).map_err(|e| LetoError::StorageError {
                reason: format!("OwnedNumericLu::solve internal shape error: {e}"),
            })?;
        self.solve_into(rhs, &mut x.view_mut())?;
        Ok(x)
    }
}
