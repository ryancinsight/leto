//! leto-compatible operator surface.
//!
//! `leto` provides by-reference affine-point operators (`&p - &q`) and
//! applies rigid transforms through `*` (`isometry * point`, `rotation *
//! vector`). The leto geometry types are `Copy`, so their canonical operators
//! are by-value (in [`point`](super::point) etc.); these impls add the
//! reference and `Mul`-application forms so generic code ported from leto
//! compiles unchanged. Every impl forwards to the existing by-value operator or
//! transform method and is `#[inline(always)]`, so it is identical machine code
//! to the value form — no allocation, no copy beyond the `Copy` the value form
//! already performs.

use super::{Isometry3, Point, Point3, UnitQuaternion, Vector, Vector3};
use core::ops::{Add, Mul, Sub};
use eunomia::RealField;

// ---- Point − Point = Vector (reference forms; value form in `point`) --------

impl<T: Sub<Output = T> + Copy, const N: usize> Sub<&Point<T, N>> for &Point<T, N> {
    type Output = Vector<T, N>;
    #[inline(always)]
    fn sub(self, rhs: &Point<T, N>) -> Vector<T, N> {
        *self - *rhs
    }
}

impl<T: Sub<Output = T> + Copy, const N: usize> Sub<Point<T, N>> for &Point<T, N> {
    type Output = Vector<T, N>;
    #[inline(always)]
    fn sub(self, rhs: Point<T, N>) -> Vector<T, N> {
        *self - rhs
    }
}

impl<T: Sub<Output = T> + Copy, const N: usize> Sub<&Point<T, N>> for Point<T, N> {
    type Output = Vector<T, N>;
    #[inline(always)]
    fn sub(self, rhs: &Point<T, N>) -> Vector<T, N> {
        self - *rhs
    }
}

// ---- Point ± Vector with a reference left operand ---------------------------

impl<T: Add<Output = T> + Copy, const N: usize> Add<Vector<T, N>> for &Point<T, N> {
    type Output = Point<T, N>;
    #[inline(always)]
    fn add(self, rhs: Vector<T, N>) -> Point<T, N> {
        *self + rhs
    }
}

impl<T: Sub<Output = T> + Copy, const N: usize> Sub<Vector<T, N>> for &Point<T, N> {
    type Output = Point<T, N>;
    #[inline(always)]
    fn sub(self, rhs: Vector<T, N>) -> Point<T, N> {
        *self - rhs
    }
}

// ---- Rigid-transform application via `*` (leto ergonomics) --------------

/// Rotate a vector: `rotation * v`.
impl<T: RealField> Mul<Vector3<T>> for UnitQuaternion<T> {
    type Output = Vector3<T>;
    #[inline(always)]
    fn mul(self, v: Vector3<T>) -> Vector3<T> {
        self.transform_vector(v)
    }
}

/// Apply an isometry to a point: `iso * p = R·p + t`.
impl<T: RealField> Mul<Point3<T>> for Isometry3<T> {
    type Output = Point3<T>;
    #[inline(always)]
    fn mul(self, p: Point3<T>) -> Point3<T> {
        self.transform_point(p)
    }
}

impl<T: RealField> Mul<Point3<T>> for &Isometry3<T> {
    type Output = Point3<T>;
    #[inline(always)]
    fn mul(self, p: Point3<T>) -> Point3<T> {
        self.transform_point(p)
    }
}

/// Apply an isometry to a vector (rotation only): `iso * v = R·v`.
impl<T: RealField> Mul<Vector3<T>> for Isometry3<T> {
    type Output = Vector3<T>;
    #[inline(always)]
    fn mul(self, v: Vector3<T>) -> Vector3<T> {
        self.transform_vector(v)
    }
}

#[cfg(test)]
// These tests deliberately use the by-reference operator forms (`&a - &b`,
// `&p + v`) that this module adds; `op_ref` would flag them as needless on
// `Copy` types, which is exactly the form under test.
#[allow(clippy::op_ref)]
mod tests {
    use crate::geometry::{Isometry3, Point3, Translation3, Unit, UnitQuaternion, Vector, Vector3};

    #[test]
    fn point_reference_subtraction_matches_value() {
        let a = Point3::new(3.0_f64, 5.0, -2.0);
        let b = Point3::new(1.0, 2.0, 4.0);
        let by_val: Vector3<f64> = a - b;
        let by_ref: Vector3<f64> = &a - &b;
        assert_eq!(by_val.data, by_ref.data);
        assert_eq!((&a - b).data, by_val.data);
        assert_eq!((a - &b).data, by_val.data);
    }

    #[test]
    fn reference_point_plus_vector() {
        let p = Point3::new(1.0_f64, 2.0, 3.0);
        let v = Vector3::from_array([10.0, 20.0, 30.0]);
        assert_eq!((&p + v).coords.data, (p + v).coords.data);
        assert_eq!((&p - v).coords.data, (p - v).coords.data);
    }

    #[test]
    fn mul_applies_transforms() {
        let z = Unit::new_normalize(Vector::from_array([0.0_f64, 0.0, 1.0]));
        let rot = UnitQuaternion::from_axis_angle(z, core::f64::consts::FRAC_PI_2);
        // rotation * vector == transform_vector
        let v = Vector3::from_array([1.0_f64, 0.0, 0.0]);
        assert_eq!((rot * v).data, rot.transform_vector(v).data);
        // isometry * point == transform_point
        let iso = Isometry3::from_parts(Translation3::new(10.0, 0.0, 0.0), rot);
        let p = Point3::new(1.0_f64, 0.0, 0.0);
        assert_eq!((iso * p).coords.data, iso.transform_point(p).coords.data);
        assert_eq!((&iso * p).coords.data, iso.transform_point(p).coords.data);
    }
}
