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
    ///
    /// Components are scaled by their largest magnitude before squaring. This
    /// preserves finite representable lengths when the unscaled sum of squares
    /// would overflow or underflow, while all arithmetic remains in `T`.
    /// A NaN component produces NaN, including when another component is
    /// infinite; otherwise any infinite component produces positive infinity.
    #[inline]
    pub fn norm(self) -> T {
        let mut scale = <T as NumericElement>::ZERO;
        for value in self.data {
            let magnitude = value.abs();
            if magnitude.is_nan() {
                return <T as NumericElement>::NAN;
            }
            if magnitude > scale {
                scale = magnitude;
            }
        }

        if scale == <T as NumericElement>::ZERO || scale == <T as NumericElement>::INFINITY {
            return scale;
        }

        let mut scaled_sum = <T as NumericElement>::ZERO;
        for value in self.data {
            let normalized = value / scale;
            scaled_sum = normalized.scalar_fmadd(normalized, scaled_sum);
        }
        scale * scaled_sum.sqrt()
    }

    /// Unit vector in the same direction, `self / ‖self‖`.
    ///
    /// Matches `leto::normalize`: a zero-length input yields a non-finite
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
    use eunomia::{NumericElement, RealField};

    fn assert_relative_eq<T: RealField>(actual: T, expected: T) {
        let relative_error = (actual - expected).abs() / expected.abs();
        // Scaling, two squared terms, one addition, sqrt, and rescaling each
        // contribute at most a small multiple of epsilon for this conditioned
        // reference. Eight epsilons bounds those six rounded operations.
        let bound = T::EPSILON * T::from_f64(8.0);
        assert!(
            relative_error <= bound,
            "relative error {relative_error:?} exceeds {bound:?}: actual={actual:?}, expected={expected:?}"
        );
    }

    fn assert_range_stable<T: RealField>(large: T, small: T) {
        let zero = <T as NumericElement>::ZERO;
        let root_two = T::SQRT_2;

        let large_vector = Vector3::new(large, large, zero);
        assert_relative_eq(large_vector.norm(), large * root_two);
        assert_relative_eq(Vector3::zeros().distance(large_vector), large * root_two);

        let small_vector = Vector3::new(small, small, zero);
        assert_relative_eq(small_vector.norm(), small * root_two);
        assert_relative_eq(Vector3::zeros().distance(small_vector), small * root_two);
    }

    fn assert_boundary_norms<T: RealField>(smallest_subnormal: T, maximum_finite: T) {
        let zero = <T as NumericElement>::ZERO;

        assert_eq!(
            Vector3::new(smallest_subnormal, zero, zero).norm(),
            smallest_subnormal
        );
        assert_eq!(
            Vector3::new(maximum_finite, zero, zero).norm(),
            maximum_finite
        );

        let half_maximum = maximum_finite / T::from_f64(2.0);
        assert_relative_eq(
            Vector3::new(half_maximum, half_maximum, zero).norm(),
            half_maximum * T::SQRT_2,
        );
    }

    fn assert_ieee_behavior<T: RealField>() {
        let zero = <T as NumericElement>::ZERO;
        let one = <T as NumericElement>::ONE;
        let infinity = <T as NumericElement>::INFINITY;
        let nan = <T as NumericElement>::NAN;
        let origin = Vector3::new(zero, zero, zero);

        assert_eq!(origin.norm(), zero);
        assert_eq!(Vector3::new(infinity, one, zero).norm(), infinity);
        assert!(Vector3::new(nan, infinity, zero).norm().is_nan());
        assert!(origin.distance(Vector3::new(nan, one, zero)).is_nan());
        assert_eq!(origin.distance(Vector3::new(infinity, one, zero)), infinity);
    }

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

    #[test]
    fn norm_is_range_stable_for_supported_fields() {
        assert_range_stable(1.0e20_f32, 1.0e-30_f32);
        assert_range_stable(1.0e200_f64, 1.0e-200_f64);
        assert_boundary_norms(f32::from_bits(1), f32::MAX);
        assert_boundary_norms(f64::from_bits(1), f64::MAX);
    }

    #[test]
    fn norm_preserves_ieee_non_finite_behavior() {
        assert_ieee_behavior::<f32>();
        assert_ieee_behavior::<f64>();
    }
}
