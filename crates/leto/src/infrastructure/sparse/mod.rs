//! Sparse array storage formats and operations.
//!
//! This module provides sparse matrix formats (CSR, CSC, COO, block-sparse)
//! for efficient representation and manipulation of sparse data structures.
//! All formats implement a common trait surface for backend-agnostic operations.

pub mod convert;
pub mod coo;
pub mod csc;
pub mod csr;
pub mod ops;
pub mod traits;

// Re-export common types
pub use coo::CooArray;
pub use csc::CscArray;
pub use csr::CsrArray;
pub use traits::{SparseFormat, SparseStorage, SparseStorageMut};
