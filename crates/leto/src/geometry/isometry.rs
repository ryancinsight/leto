//! Rigid-body transforms: [`Translation3`] and [`Isometry3`].

use super::{Point3, UnitQuaternion, Vector3};
use eunomia::RealField;

/// A pure translation in 3-space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(deserialize = "T: serde::Deserialize<'de> + Copy + Default")))]
#[repr(transparent)]
pub struct Translation3<T> {
    /// Translation vector.
    pub vector: Vector3<T>,
}

impl<T> Translation3<T> {
    /// Construct from components.
    #[inline(always)]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self {
            vector: Vector3::new([x, y, z]),
        }
    }

    /// Wrap a translation vector.
    #[inline(always)]
    pub const fn from_vector(vector: Vector3<T>) -> Self {
        Self { vector }
    }
}

/// A direct (orientation-preserving) rigid-body transform: a rotation followed
/// by a translation, `p ↦ R·p + t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(deserialize = "T: serde::Deserialize<'de> + Copy + Default")))]
#[repr(C)]
pub struct Isometry3<T> {
    /// Rotation component.
    pub rotation: UnitQuaternion<T>,
    /// Translation component.
    pub translation: Vector3<T>,
}

impl<T: RealField> Isometry3<T> {
    /// The identity transform.
    #[inline]
    pub fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
        }
    }

    /// Construct from a translation and a rotation (rotation applied first).
    #[inline]
    pub fn from_parts(translation: Translation3<T>, rotation: UnitQuaternion<T>) -> Self {
        Self {
            rotation,
            translation: translation.vector,
        }
    }

    /// Apply to a point: `R·p + t`.
    #[inline]
    pub fn transform_point(self, p: Point3<T>) -> Point3<T> {
        Point3::new(self.rotation.transform_vector(p.coords) + self.translation)
    }

    /// Apply to a vector: rotation only (vectors are translation-invariant).
    #[inline]
    pub fn transform_vector(self, v: Vector3<T>) -> Vector3<T> {
        self.rotation.transform_vector(v)
    }

    /// The inverse transform, `p ↦ Rᵀ·(p − t)`.
    #[inline]
    pub fn inverse(self) -> Self {
        let inv_rot = self.rotation.inverse();
        Self {
            rotation: inv_rot,
            translation: -inv_rot.transform_vector(self.translation),
        }
    }
}

impl<T: RealField> Default for Isometry3<T> {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Unit, Vector};

    #[test]
    fn transform_point_rotates_then_translates() {
        let z = Unit::new_normalize(Vector::new([0.0_f64, 0.0, 1.0]));
        let rot = UnitQuaternion::from_axis_angle(z, core::f64::consts::FRAC_PI_2);
        let iso = Isometry3::from_parts(Translation3::new(10.0, 0.0, 0.0), rot);
        // x=(1,0,0) rotates to (0,1,0), then +translation (10,0,0) → (10,1,0)
        let p = iso.transform_point(Point3::from_array([1.0, 0.0, 0.0]));
        assert!((p.coords.data[0] - 10.0).abs() < 1e-12);
        assert!((p.coords.data[1] - 1.0).abs() < 1e-12);
        assert!(p.coords.data[2].abs() < 1e-12);
    }

    #[test]
    fn inverse_round_trips() {
        let axis = Unit::new_normalize(Vector::new([1.0_f64, 1.0, 1.0]));
        let iso = Isometry3::from_parts(
            Translation3::new(3.0, -2.0, 5.0),
            UnitQuaternion::from_axis_angle(axis, 0.9),
        );
        let p = Point3::from_array([2.0_f64, -1.0, 4.0]);
        let round = iso.inverse().transform_point(iso.transform_point(p));
        for k in 0..3 {
            assert!((round.coords.data[k] - p.coords.data[k]).abs() < 1e-10);
        }
    }
}
