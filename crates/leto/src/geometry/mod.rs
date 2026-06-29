//! Fixed-dimension Euclidean geometry — vectors (and, incrementally, points,
//! unit vectors, and rigid transforms) over a real scalar field.
//!
//! The atlas-native replacement for nalgebra's geometry on the CPU; GPU paths
//! belong in hephaestus. Built on the eunomia scalar field SSOT
//! ([`eunomia::RealField`]).

pub mod vector;

pub use vector::{Vector, Vector2, Vector3};
