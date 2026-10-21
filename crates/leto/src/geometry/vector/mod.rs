//! Fixed-dimension geometric [`Vector`]: the type, constructors, and indexing.
//!
//! Core type definition and indexing live in `core`, operator overloads in
//! `ops`, Euclidean geometry in `euclidean`, and the const-generic serde impls
//! in `serde` — one concern per leaf module.

mod core;
mod euclidean;
mod ops;
#[cfg(feature = "serde")]
mod serde;

pub use self::core::{Vector, Vector2, Vector3};
