//! Sparse matrix–vector product `y = A · x` over CSC, in `O(nnz)` time.
//!
//! CSC SpMV accesses each nonzero exactly once: for each column `j`, every
//! stored entry `(i, value)` in that column contributes `value * x[j]` to
//! `y[i]`.  This is a gather-free scatter-add pattern (no indirection on `x`),
//! unlike CSR which needs `x[col_indices[p]]`.
//!
//! The module is split by responsibility: [`columns`] owns the product
//! entries, the serial/parallel dispatch, the column walk and the
//! scatter-add kernels; [`parallel`] the gated row-range driver. Rows are the
//! decomposition — two columns may write the same `y[i]`, but two row ranges
//! cannot — so no per-thread accumulator or reduction pass is involved.

mod columns;
#[cfg(feature = "parallel")]
mod parallel;

#[cfg(test)]
mod tests;

pub use columns::{csc_spmv, csc_spmv_into, csc_spmv_view_into};
