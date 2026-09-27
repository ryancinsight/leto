//! Dense 2-D and batched 3-D matrix multiplication.
//!
//! `types` holds the shared layout/pointer structs and validation, `packing`
//! the output zero-fill and strided copy-back helpers, `dot_outer_kernels`
//! the two dense fast-path kernel families (`i-k-j` dot product and rank-1
//! outer product), `generic_kernel` the row-blocked fallback family for
//! strided/transposed operands, `dispatch` the layout-driven routing between
//! them, and `api` the public entry points.

mod api;
mod dispatch;
mod dot_outer_kernels;
mod generic_kernel;
mod packing;
mod types;

pub use api::{batched_matmul, matmul, matmul_accumulate, matmul_with_tile_policy};
