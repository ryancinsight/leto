//! Affine [`Point`] — a position, distinct from a [`Vector`] (displacement).
//!
//! The affine algebra is type-enforced: `Point − Point = Vector`,
//! `Point ± Vector = Point`. There is intentionally no `Point + Point`.

use super::Vector;
use core::ops::{Add, AddAssign, Index, Sub, SubAssign};
use eunomia::{NumericElement, RealField};

/// A point in `N`-dimensional affine space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(deserialize = "T: serde::Deserialize<'de> + Copy + Default"))
)]
#[repr(C)]
pub struct Point<T, const N: usize> {
    /// Position vector from the origin.
    pub coords: Vector<T, N>,
}

/// A 1-dimensional point.
pub type Point1<T> = Point<T, 1>;
/// A 2-dimensional point.
pub type Point2<T> = Point<T, 2>;
/// A 3-dimensional point.
pub type Point3<T> = Point<T, 3>;

impl<T, const N: usize> Point<T, N> {
    /// Construct from a position vector.
    #[inline(always)]
    pub const fn from_coords(coords: Vector<T, N>) -> Self {
        Self { coords }
    }

    /// Construct from a coordinate array.
    #[inline(always)]
    pub const fn from_array(data: [T; N]) -> Self {
        Self {
            coords: Vector::from_array(data),
        }
    }
}

impl<T> Point<T, 1> {
    /// Construct from a single `x` coordinate.
    #[inline(always)]
    pub const fn new(x: T) -> Self {
        Self {
            coords: Vector::from_array([x]),
        }
    }
}

impl<T> Point<T, 2> {
    /// Construct from `x`, `y` coordinates.
    #[inline(always)]
    pub const fn new(x: T, y: T) -> Self {
        Self {
            coords: Vector::from_array([x, y]),
        }
    }
}

impl<T> Point<T, 3> {
    /// Construct from `x`, `y`, `z` coordinates.
    #[inline(always)]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self {
            coords: Vector::from_array([x, y, z]),
        }
    }
}

impl<T, const N: usize> From<[T; N]> for Point<T, N> {
    #[inline(always)]
    fn from(data: [T; N]) -> Self {
        Self::from_array(data)
    }
}

impl<T, const N: usize> From<Vector<T, N>> for Point<T, N> {
    #[inline(always)]
    fn from(coords: Vector<T, N>) -> Self {
        Self { coords }
    }
}

impl<T: RealField, const N: usize> Point<T, N> {
    /// The origin (all-zero coordinates).
    #[inline]
    pub fn origin() -> Self {
        Self {
            coords: Vector::zeros(),
        }
    }

    /// Euclidean distance to `other`.
    #[inline]
    pub fn distance(self, other: Self) -> T {
        (self - other).norm()
    }

    /// Squared Euclidean distance to `other`.
    #[inline]
    pub fn distance_squared(self, other: Self) -> T {
        (self - other).norm_squared()
    }
}

/// Displacement between two points: `Point − Point = Vector`.
impl<T: Sub<Output = T> + Copy, const N: usize> Sub for Point<T, N> {
    type Output = Vector<T, N>;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Vector<T, N> {
        self.coords - rhs.coords
    }
}

/// Translate a point by a vector: `Point + Vector = Point`.
impl<T: Add<Output = T> + Copy, const N: usize> Add<Vector<T, N>> for Point<T, N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Vector<T, N>) -> Self {
        Self {
            coords: self.coords + rhs,
        }
    }
}

/// Translate a point by `−vector`: `Point − Vector = Point`.
impl<T: Sub<Output = T> + Copy, const N: usize> Sub<Vector<T, N>> for Point<T, N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Vector<T, N>) -> Self {
        Self {
            coords: self.coords - rhs,
        }
    }
}

/// Translate in place by a vector: `Point += Vector`.
impl<T: Add<Output = T> + Copy, const N: usize> AddAssign<Vector<T, N>> for Point<T, N> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Vector<T, N>) {
        self.coords += rhs;
    }
}

/// Translate in place by `−vector`: `Point −= Vector`.
impl<T: Sub<Output = T> + Copy, const N: usize> SubAssign<Vector<T, N>> for Point<T, N> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Vector<T, N>) {
        self.coords -= rhs;
    }
}

/// The origin (all coordinates zero), matching leto's `Point: Default`.
impl<T: NumericElement, const N: usize> Default for Point<T, N> {
    #[inline]
    fn default() -> Self {
        Self::from_coords(Vector::splat(<T as NumericElement>::ZERO))
    }
}

impl<T, const N: usize> Index<usize> for Point<T, N> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        &self.coords.data[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vector3;

    #[test]
    fn assign_ops_translate_in_place() {
        let mut p = Point3::new(1.0_f64, 2.0, 3.0);
        p += Vector3::from_array([10.0, 20.0, 30.0]);
        assert_eq!(p.coords.data, [11.0, 22.0, 33.0]);
        p -= Vector3::from_array([1.0, 2.0, 3.0]);
        assert_eq!(p.coords.data, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn default_is_origin() {
        assert_eq!(Point3::<f64>::default().coords.data, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn affine_algebra() {
        let a = Point3::from_array([1.0_f64, 2.0, 3.0]);
        let b = Point3::from_array([4.0_f64, 6.0, 3.0]);
        // Point − Point = Vector
        let d = b - a;
        assert_eq!(d.data, [3.0, 4.0, 0.0]);
        assert_eq!(a.distance(b), 5.0);
        // Point + Vector = Point
        assert_eq!((a + d), b);
        // Point − Vector = Point
        assert_eq!((b - d), a);
        assert_eq!(Point3::<f64>::origin().coords.data, [0.0, 0.0, 0.0]);
    }
}
