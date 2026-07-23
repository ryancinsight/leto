//! Fixed-dimension Euclidean geometry — vectors, points, unit vectors,
//! quaternion rotations, and rigid transforms over a real scalar field.
//!
//! The atlas-native replacement for nalgebra's geometry on the CPU; GPU paths
//! belong in hephaestus. Built on the eunomia scalar field SSOT
//! ([`eunomia::RealField`]).

pub mod isometry;
pub mod operators;
pub mod point;
pub mod quaternion;
pub mod rotation;
pub mod swizzle;
pub mod unit;
pub mod vector;

pub use isometry::{Isometry3, Translation3};
pub use point::{Point, Point1, Point2, Point3};
pub use quaternion::{Quaternion, UnitQuaternion};
pub use rotation::RotationBasisError;
pub use swizzle::{X, XY, XYZ, XYZW};
pub use unit::{Unit, UnitVector2, UnitVector3};
pub use vector::{Vector, Vector2, Vector3};
