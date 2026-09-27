//! Numeric phase of sparse LU factorization over CSC with partial pivoting.
//!
//! Given a [`CscMatrix`](super::CscMatrix) `A` and a symbolic distribution
//! [`super::lu_symbolic::SymbolicLu`], this module computes the numerical
//! factorization `P · A = L · U`, with `P` the row permutation produced
//! by partial-pivoting selection at each elimination step. The result is
//! a [`NumericLu`] exposing a triangular-solve path
//! `L · y = P · b` then `U · x = y`.
//!
//! # Algorithm (left-looking, partial-pivoting)
//!
//! For each column `j = 0 .. n-1`:
//!
//! 1. Initialize a dense work column `w[0..n]` with the entries of `A[:, j]`.
//! 2. For each prior column `k < j` whose pivot row `r_k` falls in column
//!    `j`'s structural row set, eliminate its contribution:
//!    `w[i] -= L[i, k] · U[k, j]` for `i > r_k` in column k's L-pattern that
//!    intersect column j's pattern. Equivalently under the Gilbert/Peierls
//!    left-looking form: walk column `j`'s reachability; for each row
//!    index `i ≥ j` in the reach, perform the elimination using the already
//!    factored columns of L/U and the known pivot permutation.
//! 3. After all contributions are absorbed, the column-evaluation
//!    `w[0..n]` is the unreduced column `(L·U)[:, j]`. The pivot row is
//!    chosen as the largest-magnitude row index in `w[j..n]` (the strict
//!    lower-triangular part of the unreduced column), subject to
//!    `pivot_tolerance` thresholding. Record the row swap into `P`.
//! 4. Slot the values into the preallocated `L`/`U` value buffers per the
//!    symbolic pattern.
//!
//! The left-looking iteration never grows the pattern — the symbolic phase
//! upper-bounds fill. Partial pivoting permutes row *labels* but not
//! pattern *counts*. See Davis, *Direct Methods for Sparse Linear Systems*
//! (SIAM, 2006), §8.7 (numeric factorization) for the formal theorem.
//!
//! # Complexity
//!
//! `O(flops(A) + n)` where `flops(A)` is the count of nontrivial multiply
//! accumulations performed by the elimination — bounded by the symbolic
//! pattern upper bound. For banded `k`-diagonal matrices `flops = O(n ·
//! k²)`; for general matrices the dense fallback path is dispatched before
//! the symbolic phase runs (see [`super::lu_sparse::SparseLuSolver`] for
//! the dispatch criteria).
//!
//! # Singularity detection
//!
//! If the largest-magnitude candidate pivot in column `j`'s surviving lower
//! part is below `pivot_tolerance · max(|w[j..n]|)` in absolute value,
//! the factorization fails with [`LetoError::StorageError`](leto::LetoError::StorageError) carrying a
//! "matrix singular to working precision at column {j}" reason. The dense
//! [`lu_decompose`](crate::application::linalg::lu) path uses the same
//! convention, so consumers' error-handling logic is unchanged.
//!
//! `types` holds [`NumericLu`]'s storage and shape accessors; `solve` is the
//! triangular-solve path (also the shared core `lu_sparse::OwnedNumericLu`
//! reuses); `factor` is [`factor_numeric`] itself.

mod factor;
mod solve;
mod types;

pub use factor::factor_numeric;
pub(in crate::application::sparse) use solve::triangular_solve_into;
pub use types::NumericLu;

#[cfg(test)]
mod tests;
