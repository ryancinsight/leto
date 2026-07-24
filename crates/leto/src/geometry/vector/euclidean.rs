//! Euclidean geometry on [`Vector`] over a real scalar field: inner product,
//! norms, normalization, distance, the zero vector, the 3-vector cross product,
//! and the canonical axes.

use super::Vector;
use crate::geometry::Unit;
use eunomia::{NumericElement, RealField};

impl<T: RealField, const N: usize> Vector<T, N> {
    /// The zero vector.
    #[inline]
    pub fn zeros() -> Self {
        Self::splat(<T as NumericElement>::ZERO)
    }

    /// Inner product `Σ aᵢ·bᵢ`. Requires `N ≥ 1`.
    #[inline]
    pub fn dot(self, rhs: Self) -> T {
        let mut acc = self.data[0] * rhs.data[0];
        let mut i = 1;
        while i < N {
            acc += self.data[i] * rhs.data[i];
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

/// 3-vector cross product and canonical axes.
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

    /// The `+x` unit axis `(1, 0, 0)`.
    #[inline]
    pub fn x_axis() -> Unit<T, 3> {
        Unit::new_unchecked(Self::new(
            <T as NumericElement>::ONE,
            <T as NumericElement>::ZERO,
            <T as NumericElement>::ZERO,
        ))
    }

    /// The `+y` unit axis `(0, 1, 0)`.
    #[inline]
    pub fn y_axis() -> Unit<T, 3> {
        Unit::new_unchecked(Self::new(
            <T as NumericElement>::ZERO,
            <T as NumericElement>::ONE,
            <T as NumericElement>::ZERO,
        ))
    }

    /// The `+z` unit axis `(0, 0, 1)`.
    #[inline]
    pub fn z_axis() -> Unit<T, 3> {
        Unit::new_unchecked(Self::new(
            <T as NumericElement>::ZERO,
            <T as NumericElement>::ZERO,
            <T as NumericElement>::ONE,
        ))
    }

    /// The basis vector `(1, 0, 0)` (the [`x_axis`](Self::x_axis) direction as a
    /// plain `Vector`, not wrapped in [`Unit`]).
    #[inline]
    pub fn x() -> Self {
        Self::new(
            <T as NumericElement>::ONE,
            <T as NumericElement>::ZERO,
            <T as NumericElement>::ZERO,
        )
    }

    /// The basis vector `(0, 1, 0)`.
    #[inline]
    pub fn y() -> Self {
        Self::new(
            <T as NumericElement>::ZERO,
            <T as NumericElement>::ONE,
            <T as NumericElement>::ZERO,
        )
    }

    /// The basis vector `(0, 0, 1)`.
    #[inline]
    pub fn z() -> Self {
        Self::new(
            <T as NumericElement>::ZERO,
            <T as NumericElement>::ZERO,
            <T as NumericElement>::ONE,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::Vector3;

    #[test]
    fn dot_norm_normalize() {
        let v = Vector3::new(3.0_f64, 4.0, 0.0);
        assert_eq!(v.norm_squared(), 25.0);
        assert_eq!(v.norm(), 5.0);
        assert!((v.normalize().norm() - 1.0).abs() < 1e-12);
        let a = Vector3::new(1.0_f64, 2.0, 3.0);
        let b = Vector3::new(4.0_f64, 5.0, 6.0);
        assert_eq!(a.dot(b), 32.0);
    }

    #[test]
    fn cross_is_right_handed() {
        let x = Vector3::new(1.0_f64, 0.0, 0.0);
        let y = Vector3::new(0.0_f64, 1.0, 0.0);
        assert_eq!(x.cross(y).data, [0.0, 0.0, 1.0]);
        assert_eq!(y.cross(x).data, [0.0, 0.0, -1.0]);
    }

    #[test]
    fn basis_vectors_are_the_axes() {
        assert_eq!(Vector3::<f64>::x().data, [1.0, 0.0, 0.0]);
        assert_eq!(Vector3::<f64>::y().data, [0.0, 1.0, 0.0]);
        assert_eq!(Vector3::<f64>::z().data, [0.0, 0.0, 1.0]);
        // basis `k()` equals the corresponding `k_axis()` direction
        assert_eq!(
            Vector3::<f64>::z().data,
            Vector3::<f64>::z_axis().into_inner().data
        );
    }

    #[test]
    fn distance_and_arithmetic() {
        let a = Vector3::new(0.0_f64, 0.0, 0.0);
        let b = Vector3::new(1.0_f64, 2.0, 2.0);
        assert_eq!(a.distance(b), 3.0);
        assert_eq!((b * 2.0).data, [2.0, 4.0, 4.0]);
        assert_eq!((a - b).data, [-1.0, -2.0, -2.0]);
    }
}
