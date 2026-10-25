//! Sparse matrix–vector product `y = A · x` over CSR, in `O(nnz)` time.
//!
//! The module is split by responsibility: [`rows`] owns the product entries,
//! the serial/parallel dispatch, the row walk and the gather-dot kernel;
//! [`parallel`] the gated row driver. Rows are independent, so past the cache
//! gate the product spreads them over the runtime's workers without changing
//! any row's result.

#[cfg(feature = "parallel")]
mod parallel;
mod rows;

#[cfg(test)]
mod tests;

pub use rows::{spmv, spmv_into, spmv_view_into};
