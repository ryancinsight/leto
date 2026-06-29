//! [`Quaternion`] and the rotation type [`UnitQuaternion`].

use super::{Unit, Vector3};
use eunomia::{NumericElement, RealField};
use core::ops::Mul;

/// A quaternion `w + xi + yj + zk`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(C)]
pub struct Quaternion<T> {
    /// Scalar (real) component.
    pub w: T,
    /// `i` component.
    pub x: T,
    /// `j` component.
    pub y: T,
    /// `k` component.
    pub z: T,
}

impl<T> Quaternion<T> {
    /// Construct from components `w + xi + yj + zk`.
    #[inline(always)]
    pub const fn new(w: T, x: T, y: T, z: T) -> Self {
        Self { w, x, y, z }
    }
}

impl<T: RealField> Quaternion<T> {
    /// The multiplicative identity `1 + 0i + 0j + 0k`.
    #[inline]
    pub fn identity() -> Self {
        let zero = <T as NumericElement>::ZERO;
        Self::new(<T as NumericElement>::ONE, zero, zero, zero)
    }

    /// Conjugate `w − xi − yj − zk`.
    #[inline]
    pub fn conjugate(self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Squared norm `w² + x² + y² + z²`.
    #[inline]
    pub fn norm_squared(self) -> T {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Norm `√(w² + x² + y² + z²)`.
    #[inline]
    pub fn norm(self) -> T {
        self.norm_squared().sqrt()
    }

    /// Unit-norm quaternion in the same direction.
    #[inline]
    pub fn normalize(self) -> Self {
        let s = self.norm().recip();
        Self::new(self.w * s, self.x * s, self.y * s, self.z * s)
    }
}

/// Hamilton product.
impl<T: RealField> Mul for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn mul(self, r: Self) -> Self {
        Self::new(
            self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z,
            self.w * r.x + self.x * r.w + self.y * r.z - self.z * r.y,
            self.w * r.y - self.x * r.z + self.y * r.w + self.z * r.x,
            self.w * r.z + self.x * r.y - self.y * r.x + self.z * r.w,
        )
    }
}

/// A unit quaternion — a rotation in 3-space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct UnitQuaternion<T> {
    q: Quaternion<T>,
}

impl<T: RealField> UnitQuaternion<T> {
    /// The identity rotation.
    #[inline]
    pub fn identity() -> Self {
        Self {
            q: Quaternion::identity(),
        }
    }

    /// Rotation by `angle` (radians) about a unit `axis`.
    ///
    /// `q = cos(θ/2) + sin(θ/2)·(axis)`.
    #[inline]
    pub fn from_axis_angle(axis: Unit<T, 3>, angle: T) -> Self {
        let half = angle * T::from_f64(0.5);
        let s = half.sin();
        let a = axis.as_vector();
        Self {
            q: Quaternion::new(half.cos(), a.data[0] * s, a.data[1] * s, a.data[2] * s),
        }
    }

    /// Wrap a quaternion asserted by the caller to be unit-norm.
    #[inline(always)]
    pub const fn new_unchecked(q: Quaternion<T>) -> Self {
        Self { q }
    }

    /// The underlying quaternion.
    #[inline(always)]
    pub fn into_inner(self) -> Quaternion<T> {
        self.q
    }

    /// Inverse rotation (the conjugate, since the quaternion is unit-norm).
    #[inline]
    pub fn inverse(self) -> Self {
        Self {
            q: self.q.conjugate(),
        }
    }

    /// Rotate a vector, `v' = q·v·q⁻¹`.
    #[inline]
    pub fn transform_vector(self, v: Vector3<T>) -> Vector3<T> {
        let zero = <T as NumericElement>::ZERO;
        let pure = Quaternion::new(zero, v.data[0], v.data[1], v.data[2]);
        let r = self.q * pure * self.q.conjugate();
        Vector3::from_array([r.x, r.y, r.z])
    }

    /// Compose two rotations (`self` applied after `rhs`).
    #[inline]
    pub fn mul_unit(self, rhs: Self) -> Self {
        Self {
            q: self.q * rhs.q,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vector;

    #[test]
    fn quaternion_product_and_conjugate() {
        let i = Quaternion::new(0.0_f64, 1.0, 0.0, 0.0);
        let j = Quaternion::new(0.0_f64, 0.0, 1.0, 0.0);
        // i·j = k
        assert_eq!(i * j, Quaternion::new(0.0, 0.0, 0.0, 1.0));
        // i·i = -1
        assert_eq!(i * i, Quaternion::new(-1.0, 0.0, 0.0, 0.0));
        let q = Quaternion::new(1.0_f64, 2.0, 3.0, 4.0);
        assert_eq!(q.norm_squared(), 30.0);
    }

    #[test]
    fn rotation_90_about_z_maps_x_to_y() {
        let z = Unit::new_normalize(Vector::from_array([0.0_f64, 0.0, 1.0]));
        let r = UnitQuaternion::from_axis_angle(z, core::f64::consts::FRAC_PI_2);
        let rotated = r.transform_vector(Vector::from_array([1.0_f64, 0.0, 0.0]));
        assert!((rotated.data[0]).abs() < 1e-12);
        assert!((rotated.data[1] - 1.0).abs() < 1e-12);
        assert!((rotated.data[2]).abs() < 1e-12);
        // inverse rotates back
        let back = r.inverse().transform_vector(rotated);
        assert!((back.data[0] - 1.0).abs() < 1e-12 && back.data[1].abs() < 1e-12);
    }
}
