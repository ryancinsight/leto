//! [`Unit`] — a vector carrying the type-level invariant that its norm is 1.

use super::Vector;
use eunomia::RealField;

/// A wrapper guaranteeing its inner [`Vector`] has unit length.
///
/// Construct through [`Unit::new_normalize`] (normalizes) or [`Unit::try_new`]
/// (rejects near-zero input); [`Unit::new_unchecked`] trusts the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound(deserialize = "T: serde::Deserialize<'de> + Copy + Default")))]
#[repr(transparent)]
pub struct Unit<T, const N: usize> {
    value: Vector<T, N>,
}

/// A 2-dimensional unit vector.
pub type UnitVector2<T> = Unit<T, 2>;
/// A 3-dimensional unit vector.
pub type UnitVector3<T> = Unit<T, 3>;

impl<T, const N: usize> Unit<T, N> {
    /// Wrap a vector asserted by the caller to already be unit length.
    #[inline(always)]
    pub const fn new_unchecked(value: Vector<T, N>) -> Self {
        Self { value }
    }

    /// The inner vector (guaranteed unit length).
    #[inline(always)]
    pub fn into_inner(self) -> Vector<T, N> {
        self.value
    }

    /// Borrow the inner unit vector.
    #[inline(always)]
    pub fn as_vector(&self) -> &Vector<T, N> {
        &self.value
    }
}

impl<T: RealField, const N: usize> Unit<T, N> {
    /// Normalize `v` to unit length.
    ///
    /// As with [`Vector::normalize`], a zero-length input yields a non-finite
    /// result; use [`Unit::try_new`] to reject it.
    #[inline]
    pub fn new_normalize(v: Vector<T, N>) -> Self {
        Self {
            value: v.normalize(),
        }
    }

    /// Normalize `v`, returning `None` if `‖v‖ ≤ min_norm`.
    #[inline]
    pub fn try_new(v: Vector<T, N>, min_norm: T) -> Option<Self> {
        if v.norm() > min_norm {
            Some(Self::new_normalize(v))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_and_invariant() {
        let u = Unit::new_normalize(Vector::from_array([3.0_f64, 4.0, 0.0]));
        assert!((u.as_vector().norm() - 1.0).abs() < 1e-12);
        assert!((u.as_vector().data[0] - 0.6).abs() < 1e-12);
        assert!((u.as_vector().data[1] - 0.8).abs() < 1e-12);
        // try_new rejects near-zero
        assert!(Unit::try_new(Vector::from_array([0.0_f64, 0.0, 0.0]), 1e-9).is_none());
        assert!(Unit::try_new(Vector::from_array([1.0_f64, 0.0, 0.0]), 1e-9).is_some());
    }
}
