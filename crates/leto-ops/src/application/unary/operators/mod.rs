//! The named real unary operations, by family.
//!
//! [`contract`] holds the `UnaryOp` trait and the `unary_map` entries,
//! [`elementary`] the macro-defined elementary ops, [`bessel`] the Bessel
//! family, and [`activation`] the activation set with its gradients.

mod activation;
mod bessel;
mod contract;
mod elementary;

pub use activation::*;
pub use bessel::*;
pub use contract::{unary_map, unary_map_into, UnaryOp};
pub use elementary::*;
