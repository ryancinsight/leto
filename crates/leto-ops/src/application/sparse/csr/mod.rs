//! Compressed Sparse Row (CSR) storage — the solve/kernel-facing sparse format.
//!
//! See the [module theorem](super) for the representation identity and the
//! `O(nnz)` kernel-complexity argument. CSR is the format [`spmv`](super::spmv)
//! and [`spmm`](super::spmm) consume; assembly happens in
//! [`CooMatrix`](super::CooMatrix), which converts here via
//! [`CooMatrix::to_csr`](super::CooMatrix::to_csr).

mod matrix;
mod policy;
mod structure;
mod views;

pub use matrix::CsrMatrix;
pub(super) use policy::address;
pub use policy::{Canonical, CsrIndex, CsrOrdering, StoredOrder};
pub use views::{CsrRow, CsrView, CsrViewBuilder};
