//! Symbolic factorization phase of sparse LU over CSC.
//!
//! The symbolic phase computes the static sparsity patterns of `L` (unit
//! lower triangular) and `U` (upper triangular) without considering matrix
//! values. The pattern is encoded as column-compressed index lists
//! (`l_row_indices`, `u_row_indices`) plus their `col_ptr` offset arrays.
//!
//! # Algorithm (sequential left-looking symbolic reach — Gilbert/Peierls)
//!
//! For each column `j = 0 .. n-1`, the structural pattern `R(j)` of
//! `(L ∖ I) ∪ U` at column `j` is computed by left-looking transitive
//! reachability against the *already-computed* L patterns of columns
//! `0 .. j-1`:
//!
//! 1. Seed the reach with the row indices of `A[:, j]`.
//! 2. Maintain a worklist of pattern rows; for each worklist row `k < j`,
//!    the structural rows of column `k` of `L` (the rows strictly below
//!    the natural pivot row `k`) are merged into the pattern. Merging is
//!    transitive: newly added rows are also walked, in increasing
//!    order, so the pattern upper-bounds fill under natural column
//!    ordering.
//! 3. Rows `≤ j` go to `U` column `j` (the diagonal `j` is always in `U`);
//!    rows `> j` go to `L` column `j` (strictly below the diagonal).
//!
//! This is the classical pivoting-free upper bound on fill (Davis, *Direct
//! Methods for Sparse Linear Systems*, SIAM 2006, §6.1). Partial pivoting in
//! the numeric phase permutes row *labels* inside this pattern but never
//! grows it: the pattern is structurally valid for any pivot permutation
//! allowed by partial pivoting.
//!
//! # Ordering strategies
//!
//! Natural column ordering is the default and shipped in v0.40.0. The
//! Approximate Minimum Degree (AMD) ordering of Amestoy-Davis-Duff 1996 is
//! available via [`factor_symbolic_with_ordering`] with
//! [`OrderingStrategy::AmdApproxMinDegree`]; see the `amd` submodule for the
//! algorithm reference and the bounded-increment simplification rationale.
//! AMD returns a permutation that is applied symmetrically to `A` before the
//! symbolic reach runs, so the resulting `SymbolicLu` references row/column
//! indices in the *permuted* matrix. The numeric phase is unchanged; the
//! `amd_col_perm: Option<Vec<usize>>` slot records the permutation so the
//! solve can inverse-permute the column-order solution back to original
//! row/column order.

use super::amd::amd_order;
use super::lu_sparse::OrderingStrategy;
use super::CscMatrix;
use crate::domain::scalar::Scalar;

/// Symbolic factorization result: the static sparsity pattern of `L ∪ U`
/// under natural column ordering, encoded as paired CSC index arrays.
///
/// This is the pivoting-free upper bound on fill. The numeric phase in
/// [`crate::application::sparse::lu_numeric`] consumes this pattern and
/// records the actual row permutation produced by partial pivoting.
///
/// # Storage convention
///
/// - `l_col_ptr`, `l_row_indices`: CSC storage for the strictly-lower part
///   of `L`. The unit diagonal is implicit (not stored). For column `j`,
///   the stored row indices are strictly greater than `j` and sorted
///   ascending. `l_row_indices[l_col_ptr[j]..l_col_ptr[j+1]]` is the set
///   of structural row indices `i > j` with a nonzero `L[i, j]`.
/// - `u_col_ptr`, `u_row_indices`: CSC storage for the upper-triangular
///   part of `U` (diagonal inclusive). For column `j`, the stored row
///   indices are `≤ j` and sorted ascending. `u_row_indices[u_col_ptr[j]..u_col_ptr[j+1]]`
///   is the set of structural row indices `i ≤ j` with a nonzero `U[i, j]`.
#[derive(Debug, Clone)]
#[must_use = "SymbolicLu carries the static L/U pattern consumed by factor_numeric"]
pub struct SymbolicLu {
    /// Matrix order `n`.
    pub n: usize,
    /// CSC column pointers for the strictly-lower part of `L`.
    pub l_col_ptr: Vec<usize>,
    /// CSC row indices for the strictly-lower part of `L`.
    pub l_row_indices: Vec<usize>,
    /// CSC column pointers for the upper-triangular part of `U`.
    pub u_col_ptr: Vec<usize>,
    /// CSC row indices for the upper-triangular part of `U`.
    pub u_row_indices: Vec<usize>,
    /// Optional column permutation produced by a fill-reducing strategy
    /// (currently [`OrderingStrategy::AmdApproxMinDegree`]).
    ///
    /// `None` means natural ordering was used. `Some(perm)` means the
    /// symbolic pattern refers to `A_perm = A[perm, perm]`: row/column
    /// indices in `l_row_indices`/`u_row_indices` are indices into
    /// `A_perm`, not `A`. The numeric/solve phases consume the same
    /// permuted matrix; the triangular solve inverse-permutes `x_perm`
    /// back to original order via `x[perm[i]] = x_perm[i]`.
    ///
    /// Length `n` when `Some`; never populated for the natural-ordering
    /// `factor_symbolic` entry point. The string convention is identical
    /// to [`amd_order`](super::amd::amd_order)'s output.
    pub amd_col_perm: Option<Vec<usize>>,
}

impl SymbolicLu {
    /// Returns the matrix order `n`.
    #[must_use]
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of structurally-nonzero entries in the strictly
    /// lower part of `L`.
    #[must_use]
    #[inline]
    pub fn l_nnz(&self) -> usize {
        self.l_row_indices.len()
    }

    /// Returns the number of structurally-nonzero entries in `U` (including
    /// the diagonal).
    #[must_use]
    #[inline]
    pub fn u_nnz(&self) -> usize {
        self.u_row_indices.len()
    }
}

/// Compute the symbolic LU pattern of `A` (encoded as a [`CscMatrix`])
/// under natural column ordering.
///
/// The result captures the pivoting-free upper bound on `L`/`U` fill; the
/// numeric phase reshuffles row orderings inside this pattern.
///
/// # Panics
///
/// Panics if `csc` is not square (by construction this is asserted in the
/// caller — the public `SparseLuSolver` validates square shape before
/// reaching the symbolic phase).
///
/// # Examples
///
/// ```
/// use leto_ops::application::sparse::{CooMatrix, factor_symbolic};
///
/// let mut coo = CooMatrix::new(3, 3);
/// // Identity matrix: U == I, L is empty.
/// coo.push(0, 0, 1.0_f64);
/// coo.push(1, 1, 1.0_f64);
/// coo.push(2, 2, 1.0_f64);
/// let csc = coo.to_csc();
/// let symbolic = factor_symbolic(&csc);
/// assert_eq!(symbolic.n(), 3);
/// assert_eq!(symbolic.u_nnz(), 3); // diagonal only
/// assert_eq!(symbolic.l_nnz(), 0);
/// ```
pub fn factor_symbolic<T: Scalar>(csc: &CscMatrix<T>) -> SymbolicLu {
    let (nrows, ncols) = csc.shape();
    debug_assert_eq!(nrows, ncols, "sparse LU requires a square matrix");
    let n = nrows;

    let col_ptr = csc.col_ptr();
    let row_indices = csc.row_indices();

    // Per-column scratch arrays for the symbolic reachability walk.
    let mut marked: Vec<bool> = vec![false; n]; // tracks membership in reach[j]
    let mut reach: Vec<usize> = Vec::with_capacity(n); // worklist + final reach
    let mut sweep: Vec<usize> = Vec::with_capacity(n); // sorted iteration set

    let mut u_col_ptr = Vec::with_capacity(n + 1);
    let mut u_row_indices: Vec<usize> = Vec::new();
    // L columns computed so far — used to fan out the reach of later columns.
    // L column k is at l_row_indices[l_col_ptr[k]..l_col_ptr[k+1]].
    let mut l_col_ptr = Vec::with_capacity(n + 1);
    let mut l_row_indices: Vec<usize> = Vec::new();

    u_col_ptr.push(0);
    l_col_ptr.push(0);

    for j in 0..n {
        reach.clear();
        sweep.clear();

        // 1. Seed reach with column j of A's direct row indices: every
        //    `A[i, j]` is structurally nonzero, so row `i` is in the column-j
        //    pattern. (Merging also marks these rows.)
        for &r in row_indices.iter().take(col_ptr[j + 1]).skip(col_ptr[j]) {
            if !marked[r] {
                marked[r] = true;
                reach.push(r);
            }
        }
        // The diagonal row j is always in U column j (a value is always read
        // there by the numeric phase, even if A[j, j] happens to be zero —
        // pivoting will permute a candidate in). Ensure it's marked.
        if !marked[j] {
            marked[j] = true;
            reach.push(j);
        }

        // 2. Transitive left-looking reach: for each worklist row k that is
        //    `< j`, the L column k's rows get merged (they are pivot-fan-out
        //    rows of column k under natural ordering, all strictly below k).
        //    Newly added rows are also walked, in increasing order; rows ≥ j
        //    are leaf entries in the reach (no further fan-out to merge — no
        //    L column exists at index ≥ j yet).
        let mut head = 0usize;
        while head < reach.len() {
            let k = reach[head];
            head += 1;
            if k >= j {
                continue; // leaf — no L column k yet (computed only for k < j)
            }
            // Merge L column k's structurally-nonzero rows into the pattern.
            for &r in l_row_indices
                .iter()
                .take(l_col_ptr[k + 1])
                .skip(l_col_ptr[k])
            {
                if !marked[r] {
                    marked[r] = true;
                    reach.push(r);
                }
            }
        }

        // 3. Sort reach by row index and split into U-column-j (rows ≤ j)
        //    and L-column-j (rows strictly > j) per the storage convention.
        sweep.extend_from_slice(&reach);
        sweep.sort_unstable();
        for &r in &sweep {
            if r <= j {
                u_row_indices.push(r);
            } else {
                l_row_indices.push(r);
            }
        }
        u_col_ptr.push(u_row_indices.len());
        l_col_ptr.push(l_row_indices.len());

        // Clear marks for the next column.
        for &r in &reach {
            marked[r] = false;
        }
        debug_assert!(
            marked.iter().all(|&m| !m),
            "mark array not fully cleared at column {j}"
        );
    }

    SymbolicLu {
        n,
        l_col_ptr,
        l_row_indices,
        u_col_ptr,
        u_row_indices,
        amd_col_perm: None,
    }
}

/// Compute the symbolic LU pattern of `A` under a chosen column-ordering
/// strategy.
///
/// [`OrderingStrategy::Natural`] runs the canonical
/// [`factor_symbolic`] on `csc` directly; the result references the
/// original row/column indices and carries `amd_col_perm: None`.
///
/// [`OrderingStrategy::AmdApproxMinDegree`] computes the AMD permutation
/// via [`crate::application::sparse::amd::amd_order`], applies it
/// symmetrically to produce `A_perm = A[perm, perm]`, runs `factor_symbolic`
/// on the permuted matrix, and tags the result with `amd_col_perm: Some(perm)`.
/// The numeric and triangular-solve phases consume the same permuted CSC;
/// the inverse-permutation to original row/column order is performed by the
/// solve path.
///
/// # Panics
///
/// Panics if `csc` is not square (same precondition as [`factor_symbolic`]).
///
/// # Examples
///
/// ```
/// use leto_ops::application::sparse::{
///     CooMatrix, factor_symbolic_with_ordering, OrderingStrategy, CscMatrix,
/// };
///
/// // 3×3 identity under AMD: pattern is diagonal, no fill regardless of
/// // ordering, and the returned permutation is a 3-permutation.
/// let mut coo = CooMatrix::new(3, 3);
/// for i in 0..3 {
///     coo.push(i, i, 1.0_f64);
/// }
/// let csc = coo.to_csc();
/// let symbolic = factor_symbolic_with_ordering(&csc, OrderingStrategy::Natural);
/// assert_eq!(symbolic.n(), 3);
/// assert_eq!(symbolic.u_nnz(), 3);
/// assert_eq!(symbolic.l_nnz(), 0);
/// assert!(symbolic.amd_col_perm.is_none(), "Natural yields no col perm");
///
/// let symbolic_amd = factor_symbolic_with_ordering(&csc, OrderingStrategy::AmdApproxMinDegree);
/// assert_eq!(symbolic_amd.n(), 3);
/// assert!(symbolic_amd.amd_col_perm.is_some(), "AMD yields a col perm");
/// let perm = symbolic_amd.amd_col_perm.as_ref().expect("AMD yields a perm");
/// assert_eq!(perm.len(), 3);
/// ```
#[must_use = "factor_symbolic_with_ordering carries the L/U pattern consumed by factor_numeric"]
pub fn factor_symbolic_with_ordering<T: Scalar>(
    csc: &CscMatrix<T>,
    strategy: OrderingStrategy,
) -> SymbolicLu {
    match strategy {
        OrderingStrategy::Natural => factor_symbolic(csc),
        OrderingStrategy::AmdApproxMinDegree => {
            let perm = amd_order(csc);
            let permuted_csc = apply_symmetric_perm(csc, &perm);
            let mut symbolic = factor_symbolic(&permuted_csc);
            symbolic.amd_col_perm = Some(perm);
            symbolic
        }
    }
}

/// Apply a symmetric permutation `A_perm[i, j] = A[perm[i], perm[j]]` and
/// return the permuted CSC.
///
/// Internal helper for [`factor_symbolic_with_ordering`]; also exercised by
/// the AMD tests. The implementation walks the CSC nonzeros and scatters to
/// the inverse-permuted `(i, j)` slots through a COO intermediary, letting
/// `CscMatrix::to_csc` handle the sort/dedupe.
//
// Note: applying the permutation via a COO intermediary is intentionally
// explicit here rather than threaded into `factor_symbolic`'s body. Keeping
// the permutation step out of the canonical symbolic walk preserves
// `factor_symbolic` as the natural-ordering SSOT (the existing back-compat
// invariant for consumers that already rely on it).
pub(super) fn apply_symmetric_perm<T: Scalar>(csc: &CscMatrix<T>, perm: &[usize]) -> CscMatrix<T> {
    let n = csc.nrows();
    debug_assert_eq!(csc.ncols(), n);
    debug_assert_eq!(perm.len(), n);
    let mut inv = vec![0usize; n];
    for (slot, &orig) in perm.iter().enumerate() {
        inv[orig] = slot;
    }

    let col_ptr = csc.col_ptr();
    let row_indices = csc.row_indices();
    let values = csc.values();

    let mut coo = super::CooMatrix::new(n, n);
    for j in 0..n {
        for p in col_ptr[j]..col_ptr[j + 1] {
            let i = row_indices[p];
            coo.push(inv[i], inv[j], values[p]);
        }
    }
    coo.to_csc()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::sparse::{CooMatrix, OrderingStrategy};

    fn csc2(entries: &[(usize, usize, f64)]) -> CscMatrix<f64> {
        let n = 2;
        let mut coo = CooMatrix::new(n, n);
        for &(i, j, v) in entries {
            coo.push(i, j, v);
        }
        coo.to_csc()
    }

    fn csc3(entries: &[(usize, usize, f64)]) -> CscMatrix<f64> {
        let mut coo = CooMatrix::new(3, 3);
        for &(i, j, v) in entries {
            coo.push(i, j, v);
        }
        coo.to_csc()
    }

    /// U column `j` stores rows `≤ j` ascending; L column `j` stores rows
    /// `> j` ascending; the diagonal `j` is always structurally present in U.
    #[test]
    fn storage_convention_holds_for_a_full_column() {
        // A[2,0], A[1,0], A[0,0] all nonzero -> column 0 of U carries row 0,
        // column 0 of L carries rows {1, 2}.
        let csc = csc3(&[(0, 0, 2.0), (1, 0, 3.0), (2, 0, 5.0)]);
        let sym = factor_symbolic(&csc);
        assert_eq!(sym.n(), 3);
        // Whole-vector patterns across all columns: col0 U={0}, col1 U={1},
        // col2 U={2}; L col0={1,2}, L col1/2 empty.
        assert_eq!(sym.u_row_indices, vec![0, 1, 2]);
        assert_eq!(sym.l_row_indices, vec![1, 2]);
        assert_eq!(sym.u_col_ptr, vec![0, 1, 2, 3]);
        assert_eq!(sym.l_col_ptr, vec![0, 2, 2, 2]);
    }

    /// The diagonal is always in U even when A[j, j] is zero.
    #[test]
    fn diagonal_is_always_structurally_present_in_u() {
        // Off-diagonal only: A[1,0] nonzero.
        let csc = csc2(&[(1, 0, 4.0)]);
        let sym = factor_symbolic(&csc);
        // Column 0: seed {1}; diagonal 0 forced into U. Rows ≤ 0 -> {0} to U,
        // row 1 > 0 -> L column 0.
        assert_eq!(sym.u_row_indices, vec![0, 1]);
        assert_eq!(sym.l_row_indices, vec![1]);
        assert_eq!(sym.u_col_ptr, vec![0, 1, 2]);
        assert_eq!(sym.l_col_ptr, vec![0, 1, 1]);
        assert_eq!(sym.u_nnz(), 2);
        assert_eq!(sym.l_nnz(), 1);
    }

    /// Identity: U is the diagonal, L is empty.
    #[test]
    fn identity_has_no_l_fill() {
        let csc = csc3(&[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)]);
        let sym = factor_symbolic(&csc);
        assert_eq!(sym.u_nnz(), 3);
        assert_eq!(sym.l_nnz(), 0);
        // U row indices per column: [0], [1], [2].
        assert_eq!(sym.u_row_indices, vec![0, 1, 2]);
    }

    /// Structural fill is an upper bound: a 2×2 with A[0,1] and A[1,0]
    /// produces a full L∪U pattern (2×2 dense), matching partial-pivot fill.
    #[test]
    fn fill_upper_bound_on_dense_2x2() {
        let csc = csc2(&[(0, 1, 1.0), (1, 0, 1.0)]);
        let sym = factor_symbolic(&csc);
        // Column 0: seed {1}; diagonal 0. U col0 = {0}, L col0 = {1}.
        // Column 1: seed {0}; diagonal 1; fan-out from L col0 row 1 → merge
        //   row 1; U col1 = {0, 1}.
        assert_eq!(sym.u_row_indices, vec![0, 0, 1]);
        assert_eq!(sym.l_row_indices, vec![1]);
        assert_eq!(sym.u_nnz(), 3);
        assert_eq!(sym.l_nnz(), 1);
    }

    /// The symmetric permutation reorders rows and columns together; applying
    /// it to the identity yields the identity (pattern preserved).
    #[test]
    fn symmetric_perm_preserves_identity_pattern() {
        let csc = csc3(&[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)]);
        let perm = vec![2usize, 0, 1];
        let permuted = apply_symmetric_perm(&csc, &perm);
        let sym = factor_symbolic(&permuted);
        assert_eq!(sym.u_nnz(), 3);
        assert_eq!(sym.l_nnz(), 0);
    }

    /// AMD ordering returns a valid permutation and the symbolic pattern
    /// still satisfies the storage convention on the permuted matrix.
    #[test]
    fn amd_ordering_produces_valid_permuted_pattern() {
        let csc = csc3(&[(0, 2, 1.0), (2, 0, 1.0), (1, 1, 1.0)]);
        let sym = factor_symbolic_with_ordering(&csc, OrderingStrategy::AmdApproxMinDegree);
        let perm = sym.amd_col_perm.as_ref().expect("AMD yields a perm");
        assert_eq!(perm.len(), 3);
        // Validate the permutation: it is a bijection on 0..3.
        let mut seen = [false; 3];
        for &p in perm {
            assert!(!seen[p], "AMD permutation must be a bijection");
            seen[p] = true;
        }
        // Storage convention holds for the permuted pattern.
        for j in 0..sym.n() {
            let u = &sym.u_row_indices[sym.u_col_ptr[j]..sym.u_col_ptr[j + 1]];
            let l = &sym.l_row_indices[sym.l_col_ptr[j]..sym.l_col_ptr[j + 1]];
            assert!(u.iter().all(|&r| r <= j), "U col {j} rows must be ≤ {j}");
            assert!(l.iter().all(|&r| r > j), "L col {j} rows must be > {j}");
            assert!(u.iter().is_sorted(), "U col {j} rows must be ascending");
            assert!(l.iter().is_sorted(), "L col {j} rows must be ascending");
        }
    }

    /// Natural ordering never sets the AMD column permutation.
    #[test]
    fn natural_ordering_has_no_col_perm() {
        let csc = csc2(&[(0, 1, 1.0), (1, 0, 1.0)]);
        let sym = factor_symbolic_with_ordering(&csc, OrderingStrategy::Natural);
        assert!(sym.amd_col_perm.is_none());
    }
}
