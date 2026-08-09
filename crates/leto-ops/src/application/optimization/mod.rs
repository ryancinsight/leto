//! Numerical optimisation utilities.
//!
//! General-purpose, problem-agnostic optimisers.
//! SSOT for the Atlas simulation stack.

pub mod lbfgs;

pub use lbfgs::{LbfgsConfig, LbfgsMemory, LbfgsResult, minimize};
