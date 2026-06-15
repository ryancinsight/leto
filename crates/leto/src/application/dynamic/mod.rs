//! Runtime-rank array boundary layer (ADR 0007).
//!
//! `ArrayD<T, S>` carries data whose rank is known only at run time and bridges,
//! zero-copy, to the const-rank [`Array`](crate::application::array::Array) for
//! all computation. It is a boundary carrier, not a parallel compute substrate.
//!
//! # Submodules
//! - array — [`ArrayD`]: the runtime-rank carrier (construct/inspect/index/
//!   reshape/materialize).
//! - bridge — `into_dyn` / `into_dimensionality`: the zero-copy rank bridge.

pub mod array;
pub mod bridge;

pub use array::ArrayD;
