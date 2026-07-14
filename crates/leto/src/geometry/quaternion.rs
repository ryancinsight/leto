//! [`Quaternion`] and the rotation type [`UnitQuaternion`].

use super::{Unit, Vector3};
use core::ops::{Add, Div, Mul, Neg, Sub};
use eunomia::{NumericElement, RealField};

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

impl<T: RealField> Add for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn add(self, r: Self) -> Self {
        Self::new(self.w + r.w, self.x + r.x, self.y + r.y, self.z + r.z)
    }
}

impl<T: RealField> Sub for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn sub(self, r: Self) -> Self {
        Self::new(self.w - r.w, self.x - r.x, self.y - r.y, self.z - r.z)
    }
}

impl<T: RealField> Neg for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.w, -self.x, -self.y, -self.z)
    }
}

impl<T: RealField> Mul<T> for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn mul(self, s: T) -> Self {
        Self::new(self.w * s, self.x * s, self.y * s, self.z * s)
    }
}

impl<T: RealField> Div<T> for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn div(self, s: T) -> Self {
        Self::new(self.w / s, self.x / s, self.y / s, self.z / s)
    }
}

impl<T: RealField> Quaternion<T> {
    /// Inverse `q⁻¹ = conj(q) / |q|²`.
    ///
    /// Returns `None` when the quaternion is zero (cannot be inverted).
    #[inline]
    pub fn try_inverse(self) -> Option<Self> {
        let n2 = self.norm_squared();
        if n2 == T::ZERO {
            return None;
        }
        Some(self.conjugate() / n2)
    }

    /// Convert to a 4×4 row-major rotation matrix.
    ///
    /// The matrix is the rotation part of a 4×4 homogeneous transform,
    /// suitable for use with column vectors: `v' = M·v`.
    #[inline]
    pub fn to_rotation_matrix(&self) -> crate::FixedMatrix<T, 4, 4> {
        let (x2, y2, z2) = (self.x * self.x, self.y * self.y, self.z * self.z);
        let (xy, xz, yz) = (self.x * self.y, self.x * self.z, self.y * self.z);
        let (wx, wy, wz) = (self.w * self.x, self.w * self.y, self.w * self.z);
        let one = T::ONE;
        let two = one + one;
        crate::FixedMatrix::from_rows([
            [
                one - two * (y2 + z2),
                two * (xy - wz),
                two * (xz + wy),
                T::ZERO,
            ],
            [
                two * (xy + wz),
                one - two * (x2 + z2),
                two * (yz - wx),
                T::ZERO,
            ],
            [
                two * (xz - wy),
                two * (yz + wx),
                one - two * (x2 + y2),
                T::ZERO,
            ],
            [T::ZERO, T::ZERO, T::ZERO, T::ONE],
        ])
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
    ///
    /// Evaluated in the Rodrigues form `t = 2·(q.xyz × v); v + q.w·t + q.xyz × t`
    /// rather than the literal two-quaternion sandwich: ~30 flops vs ~56, on a
    /// per-vertex-transform hot path. Differentially verified against the
    /// sandwich in tests.
    #[inline]
    pub fn transform_vector(self, v: Vector3<T>) -> Vector3<T> {
        let u = Vector3::new(self.q.x, self.q.y, self.q.z);
        let t = u.cross(v) * T::from_f64(2.0);
        v + t * self.q.w + u.cross(t)
    }

    /// Compose two rotations (`self` applied after `rhs`).
    #[inline]
    pub fn mul_unit(self, rhs: Self) -> Self {
        Self { q: self.q * rhs.q }
    }

    /// Interpolate rotations along the shortest spherical path.
    ///
    /// The interpolation is Shoemake's spherical linear interpolation:
    /// `sin((1-t)θ) / sin(θ) * q₀ + sin(tθ) / sin(θ) * q₁`, where `θ` is the
    /// angle between the unit quaternions. Nearly parallel endpoints use the
    /// normalized linear limit to avoid division by a small sine.
    #[inline]
    pub fn slerp(self, rhs: Self, t: T) -> Self {
        let mut end = rhs.q;
        let mut dot = self.q.w * end.w + self.q.x * end.x + self.q.y * end.y + self.q.z * end.z;
        if dot < T::ZERO {
            end = -end;
            dot = -dot;
        }
        dot = dot.clamp(-T::ONE, T::ONE);

        // The angular separation is below the square-root machine-epsilon
        // scale, so normalized linear interpolation is the stable limit of
        // the spherical formula for the active precision.
        if T::ONE - dot <= T::EPSILON.sqrt() {
            return Self::new_unchecked((self.q * (T::ONE - t) + end * t).normalize());
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let start_weight = ((T::ONE - t) * theta).sin() / sin_theta;
        let end_weight = (t * theta).sin() / sin_theta;
        Self::new_unchecked((self.q * start_weight + end * end_weight).normalize())
    }

    /// Convert to a 4×4 rotation matrix.
    #[inline]
    pub fn to_rotation_matrix(&self) -> crate::FixedMatrix<T, 4, 4> {
        self.q.to_rotation_matrix()
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

    #[test]
    fn transform_vector_matches_qvq_sandwich() {
        // The optimized Rodrigues form must equal the literal q·v·q⁻¹ sandwich
        // for an arbitrary rotation and vector.
        let axis = Unit::new_normalize(Vector::from_array([1.0_f64, 2.0, -3.0]));
        let q = UnitQuaternion::from_axis_angle(axis, 0.7);
        let v = Vector::from_array([4.0_f64, -1.0, 2.0]);
        let opt = q.transform_vector(v);
        let qq = q.into_inner();
        let pure = Quaternion::new(0.0, v.data[0], v.data[1], v.data[2]);
        let r = qq * pure * qq.conjugate();
        let sandwich = [r.x, r.y, r.z];
        for (k, &s) in sandwich.iter().enumerate() {
            assert!((opt.data[k] - s).abs() < 1e-12, "component {k}");
        }
    }

    #[test]
    fn slerp_midpoint_follows_shortest_rotation_path() {
        let axis = Unit::new_normalize(Vector::from_array([0.0_f64, 0.0, 1.0]));
        let start = UnitQuaternion::identity();
        let end = UnitQuaternion::from_axis_angle(axis, core::f64::consts::FRAC_PI_2);

        let midpoint = start.slerp(end, 0.5);
        let rotated = midpoint.transform_vector(Vector::from_array([1.0_f64, 0.0, 0.0]));
        let expected = 2.0_f64.sqrt() / 2.0;
        assert!((rotated.data[0] - expected).abs() < 1e-12);
        assert!((rotated.data[1] - expected).abs() < 1e-12);
        assert!(rotated.data[2].abs() < 1e-12);
    }

    #[test]
    fn slerp_handles_antipodal_representation_without_long_path() {
        let start = UnitQuaternion::identity();
        let end = UnitQuaternion::new_unchecked(-start.into_inner());

        let midpoint = start.slerp(end, 0.5);
        assert_eq!(
            midpoint.transform_vector(Vector::from_array([1.0_f64, 0.0, 0.0])),
            Vector::from_array([1.0_f64, 0.0, 0.0])
        );
    }

    #[test]
    fn rotation_matrix_matches_column_vector_convention() {
        let axis = Unit::new_normalize(Vector::from_array([0.0_f64, 0.0, 1.0]));
        let rotation = UnitQuaternion::from_axis_angle(axis, core::f64::consts::FRAC_PI_2);
        let matrix = rotation.to_rotation_matrix();
        let rotated = matrix * crate::FixedVector::new([1.0_f64, 0.0, 0.0, 1.0]);

        assert!(rotated[0].abs() < 1e-12);
        assert!((rotated[1] - 1.0).abs() < 1e-12);
        assert!(rotated[2].abs() < 1e-12);
        assert!((rotated[3] - 1.0).abs() < 1e-12);
    }
}
