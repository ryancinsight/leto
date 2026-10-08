//! Elementwise unary maps and the named-operation family.
//!
//! [`traversal`] holds the shared map kernels; [`operators`] the
//! [`UnaryOp`] contract and its operations.

mod operators;
mod traversal;

pub use operators::*;
pub use traversal::{map, map_inplace, map_into, map_into_with_cache_geometry, mapv};

// Path-preserving internal re-export: application::map routes through this
// entry, so the flat-module path it was written against keeps resolving.
pub(crate) use traversal::map_into_gated;
