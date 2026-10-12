//! Stack-backed row-major fixed-size matrix primitive.

mod core;
mod matrix4;
mod ops;
#[cfg(test)]
mod tests;

pub use core::FixedMatrix;
