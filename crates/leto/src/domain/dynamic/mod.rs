//! Runtime-rank (dynamic) layout primitives — the boundary-carrier side of the
//! const-rank/dynamic-rank split (ADR 0007).
//!
//! # Submodules
//! - layout — [`LayoutDyn`]: a `Box<[_]>`-backed strided layout whose rank is
//!   a runtime value, sharing all arithmetic with `Layout<N>` via the shared
//!   layout `kernels` module (SSOT).

pub mod layout;

pub use layout::LayoutDyn;
