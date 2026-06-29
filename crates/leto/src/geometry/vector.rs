//! Fixed-dimension geometric [`Vector`] over a real scalar field.

use eunomia::{NumericElement, RealField};
use core::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

/// A fixed-dimension column vector in Euclidean space, generic over the scalar
/// field `T` and the dimension `N` (const generic — one definition per
/// dimension, monomorphized). [`Vector2`]/[`Vector3`] are dimension aliases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Vector<T, const N: usize> {
    /// Components.
    pub data: [T; N],
}

impl<T: Default + Copy, const N: usize> Default for Vector<T, N> {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(T::default())
    }
}

/// A 2-dimensional vector.
pub type Vector2<T> = Vector<T, 2>;
/// A 3-dimensional vector.
pub type Vector3<T> = Vector<T, 3>;

impl<T, const N: usize> Vector<T, N> {
    /// Construct from a component array.
    #[inline(always)]
    pub const fn new(data: [T; N]) -> Self {
        Self { data }
    }
}

impl<T: Copy, const N: usize> Vector<T, N> {
    /// A vector with every component equal to `value`.
    #[inline(always)]
    pub const fn splat(value: T) -> Self {
        Self { data: [value; N] }
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    #[inline(always)]
    fn from(data: [T; N]) -> Self {
        Self { data }
    }
}

impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

impl<T: Add<Output = T> + Copy, const N: usize> Add for Vector<T, N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            data: core::array::from_fn(|i| self.data[i] + rhs.data[i]),
        }
    }
}

impl<T: Sub<Output = T> + Copy, const N: usize> Sub for Vector<T, N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            data: core::array::from_fn(|i| self.data[i] - rhs.data[i]),
        }
    }
}

impl<T: Neg<Output = T> + Copy, const N: usize> Neg for Vector<T, N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            data: core::array::from_fn(|i| -self.data[i]),
        }
    }
}

impl<T: Mul<Output = T> + Copy, const N: usize> Mul<T> for Vector<T, N> {
    type Output = Self;
    /// Scalar multiplication.
    #[inline(always)]
    fn mul(self, scalar: T) -> Self {
        Self {
            data: core::array::from_fn(|i| self.data[i] * scalar),
        }
    }
}

/// Euclidean operations over a real scalar field.
impl<T: RealField, const N: usize> Vector<T, N> {
    /// Inner product `Σ aᵢ·bᵢ`. Requires `N ≥ 1`.
    #[inline]
    pub fn dot(self, rhs: Self) -> T {
        let mut acc = self.data[0] * rhs.data[0];
        let mut i = 1;
        while i < N {
            acc = acc + self.data[i] * rhs.data[i];
            i += 1;
        }
        acc
    }

    /// Squared Euclidean norm `‖self‖²`.
    #[inline]
    pub fn norm_squared(self) -> T {
        self.dot(self)
    }

    /// Euclidean norm (length) `‖self‖`.
    #[inline]
    pub fn norm(self) -> T {
        self.norm_squared().sqrt()
    }

    /// Unit vector in the same direction, `self / ‖self‖`.
    ///
    /// Matches `nalgebra::normalize`: a zero-length input yields a non-finite
    /// result rather than panicking.
    #[inline]
    pub fn normalize(self) -> Self {
        self * self.norm().recip()
    }

    /// Distance to `other`.
    #[inline]
    pub fn distance(self, other: Self) -> T {
        (self - other).norm()
    }

    /// Squared distance to `other`.
    #[inline]
    pub fn distance_squared(self, other: Self) -> T {
        (self - other).norm_squared()
    }
}

impl<T: RealField, const N: usize> Vector<T, N> {
    /// The zero vector.
    #[inline]
    pub fn zeros() -> Self {
        Self::splat(<T as NumericElement>::ZERO)
    }
}

/// 3-vector cross product.
impl<T: RealField> Vector<T, 3> {
    /// Cross product `self × rhs`.
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        let [ax, ay, az] = self.data;
        let [bx, by, bz] = rhs.data;
        Self {
            data: [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_norm_normalize() {
        let v = Vector3::new([3.0_f64, 4.0, 0.0]);
        assert_eq!(v.norm_squared(), 25.0);
        assert_eq!(v.norm(), 5.0);
        assert!((v.normalize().norm() - 1.0).abs() < 1e-12);
        let a = Vector3::new([1.0_f64, 2.0, 3.0]);
        let b = Vector3::new([4.0_f64, 5.0, 6.0]);
        assert_eq!(a.dot(b), 32.0);
    }

    #[test]
    fn cross_is_right_handed() {
        let x = Vector3::new([1.0_f64, 0.0, 0.0]);
        let y = Vector3::new([0.0_f64, 1.0, 0.0]);
        assert_eq!(x.cross(y).data, [0.0, 0.0, 1.0]);
        // anti-commutative
        assert_eq!(y.cross(x).data, [0.0, 0.0, -1.0]);
    }

    #[test]
    fn distance_and_arithmetic() {
        let a = Vector3::new([0.0_f64, 0.0, 0.0]);
        let b = Vector3::new([1.0_f64, 2.0, 2.0]);
        assert_eq!(a.distance(b), 3.0);
        assert_eq!((b * 2.0).data, [2.0, 4.0, 4.0]);
        assert_eq!((a - b).data, [-1.0, -2.0, -2.0]);
    }
}
