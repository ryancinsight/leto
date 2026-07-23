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
//! # Why natural ordering for v0.40.0?
//!
//! Approximate Minimum Degree (AMD) ordering (Amestoy-Davis-Duff 1996) is
//! the long-term path for fill-reducing column reordering, but its ~300-line
//! implementation surface (with aggressive absorption + external degree
//! updates) exceeds the bounded-increment budget for v0.40.0; a buggy AMD
//! produces numerically-broken factorizations. For CFDrs's banded
//! saddle-point systems natural ordering is already near-optimal — the
//! bandwidth is small and AMD gains are modest. The AMD upgrade is tracked
//! as a follow-up board item (see ADR 0031 Consequences).

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
/// use leto_ops::application::sparse::{CooMatrix, lu_symbolic::factor_symbolic};
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
        for &r in row_indices
            .iter()
            .take(col_ptr[j + 1])
            .skip(col_ptr[j])
        {
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
    }
}
