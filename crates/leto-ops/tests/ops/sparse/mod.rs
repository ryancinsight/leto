//! Sparse-container tests, split by concern.
//!
//! [`csr`] pins construction and structure invariants, [`products`] the
//! sparse kernels against closed forms, [`complex`] the canonical containers
//! at `Complex64`, and [`views`] the zero-copy view constructors.

mod complex;
mod csr;
mod products;
mod views;
