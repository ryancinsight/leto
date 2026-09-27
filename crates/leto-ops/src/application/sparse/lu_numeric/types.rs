//! [`NumericLu`] storage and its shape accessors.

use super::super::lu_symbolic::SymbolicLu;
use crate::domain::real::RealScalar;

/// Numeric LU factorization: `P · A = L · U` with partial pivoting.
///
/// The factorization is single-use; the symbolic L/U patterns are
/// consumed *by reference* (so a caller may reuse the same symbolic
/// analysis across multiple right-hand-side solves if the matrix
/// structure is preserved and only values change).
#[derive(Debug, Clone)]
#[must_use = "NumericLu carries the P*A=L*U factorization consumed by solve"]
pub struct NumericLu<'a, T: RealScalar> {
    /// Borrowed symbolic pattern (lifetied to the caller).
    pub(super) symbolic: &'a SymbolicLu,
    /// Numeric values of L column-by-column, parallel to
    /// [`SymbolicLu::l_row_indices`] / [`SymbolicLu::l_col_ptr`].
    /// `L[j, j] = 1` implicitly; entries below the diagonal are stored.
    pub(super) l_values: Vec<T>,
    /// Numeric values of U column-by-column, parallel to
    /// [`SymbolicLu::u_row_indices`] / [`SymbolicLu::u_col_ptr`],
    /// including the diagonal.
    pub(super) u_values: Vec<T>,
    /// Row permutation produced by partial pivoting. `row_perm[i]` is the
    /// original row index that ended up in position `i` after pivoting
    /// (P·b is `b[row_perm[0]], b[row_perm[1]], …`).
    pub(super) row_perm: Vec<usize>,
}

impl<'a, T: RealScalar> NumericLu<'a, T> {
    /// Matrix order `n`.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.symbolic.n
    }

    /// Row permutation: `row_perm[i]` is the original row index that ends
    /// up in slot `i` after pivoting. Used by `solve`/`solve_into` to
    /// permute the RHS.
    #[must_use]
    #[inline]
    pub fn row_perm(&self) -> &[usize] {
        &self.row_perm
    }

    /// Decompose into the owned value/permutation buffers, releasing the
    /// symbolic borrow. The caller pairs them with a clone of the pattern
    /// (see `lu_sparse::OwnedNumericLu`).
    pub(in crate::application::sparse) fn into_parts(self) -> (Vec<T>, Vec<T>, Vec<usize>) {
        (self.l_values, self.u_values, self.row_perm)
    }
}
