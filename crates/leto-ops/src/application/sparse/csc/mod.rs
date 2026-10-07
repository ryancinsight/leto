//! Compressed Sparse Column (CSC) storage — the column-major sparse format.
//!
//! CSC is the column-major analogue of CSR: columns are stored as contiguous
//! runs in `(values, row_indices)`, with `col_ptr[j]` marking the first index
//! of column `j`. CSC is preferred when column-wise access patterns dominate
//! (column extraction, matrix constructed from column vectors, certain
//! element-assembly loops, and transposed SpMV where CSR would need a
//! gather).
//!
//! The canonical pipeline is *assemble in COO →
//! [`to_csc`](CooMatrix::to_csc) → CSC kernels*.  Conversion to CSR via
//! [`to_csr`](Self::to_csr) (which is
//! [`transpose`](Self::transpose) of the CSR ↔ CSC duality) bridges to the
//! existing CSR solver surface.

mod accessors;
mod construction;
mod conversion;
mod mutation;
mod properties;
mod types;

pub use types::{CscColumn, CscMatrix, CscView};

#[cfg(test)]
mod tests;
