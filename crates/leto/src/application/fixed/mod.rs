//! Stack-backed fixed-size vector and matrix primitives.
//!
//! These types cover small linear-algebra values where heap-backed strided
//! arrays would add avoidable allocation and layout metadata. They are plain
//! row-major array wrappers, so indexing and arithmetic stay stack-local.
//!
//! The `vector` and `matrix` submodules hold the storage types and their
//! elementwise operators; `linalg` holds the analytic 2x2/3x3
//! eigendecompositions and inverses built on top of them.

mod linalg;
mod matrix;
mod vector;

pub use matrix::FixedMatrix;
pub use vector::FixedVector;
