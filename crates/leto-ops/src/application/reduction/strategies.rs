//! [`AxisReduction`] strategy marker types: `Sum`/`Product`/`Mean`/`Min`/`Max`.

use crate::domain::scalar::{fold_max, fold_min, Scalar};
use eunomia::FloatElement;

/// Zero-sized axis-reduction contract.
pub trait AxisReduction<T: Scalar>: crate::private::OpMarker {
    /// Initial accumulator for non-empty reductions.
    fn initial(first: T) -> T;
    /// Fold one value into the accumulator.
    fn fold(acc: T, value: T) -> T;
    /// Finalize the accumulator after `axis_len` elements.
    fn finalize(acc: T, axis_len: usize) -> T;
    /// Whether an empty reduction has a defined value.
    const ALLOW_EMPTY: bool;
    /// Empty reduction value when `ALLOW_EMPTY` is true.
    const EMPTY: T;
    /// Try a fast-path slice-based reduction.
    fn reduce_slice(slice: &[T]) -> Option<T>;
}

/// Sum axis-reduction marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SumAxis;

/// Product axis-reduction marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProductAxis;

/// Mean axis-reduction marker.
///
/// Defined for float elements: the divisor is the axis length converted to `T`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeanAxis;

/// Minimum axis-reduction marker.
///
/// Folds with `fold_min` from `T::MAX_VALUE`; the contiguous route inherits
/// the same contract from [`Scalar::min_slice`]. The result is the least
/// non-NaN element of the axis: a real NaN and a `Complex` with a NaN
/// component are ignored wherever they sit, and an all-NaN axis reduces to
/// `T::MAX_VALUE`, the identity. A `Complex` orders by real part, then
/// imaginary part. The sign of a zero tie (`+0` against `-0`) is unspecified.
#[derive(Clone, Copy, Debug, Default)]
pub struct MinAxis;

/// Maximum axis-reduction marker.
///
/// The mirror of [`MinAxis`] under `fold_max`: the greatest non-NaN
/// element, `T::MIN_VALUE` for an all-NaN axis.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaxAxis;

impl crate::private::Sealed for SumAxis {}
impl crate::private::Sealed for ProductAxis {}
impl crate::private::Sealed for MeanAxis {}
impl crate::private::Sealed for MinAxis {}
impl crate::private::Sealed for MaxAxis {}

impl<T: Scalar> AxisReduction<T> for SumAxis {
    #[inline(always)]
    fn initial(first: T) -> T {
        first
    }

    #[inline(always)]
    fn fold(acc: T, value: T) -> T {
        acc.add(value)
    }

    #[inline(always)]
    fn finalize(acc: T, _axis_len: usize) -> T {
        acc
    }

    const ALLOW_EMPTY: bool = true;
    const EMPTY: T = T::ZERO;

    #[inline(always)]
    fn reduce_slice(slice: &[T]) -> Option<T> {
        Some(T::sum_slice(slice))
    }
}

impl<T: Scalar> AxisReduction<T> for ProductAxis {
    #[inline(always)]
    fn initial(first: T) -> T {
        first
    }

    #[inline(always)]
    fn fold(acc: T, value: T) -> T {
        acc.mul(value)
    }

    #[inline(always)]
    fn finalize(acc: T, _axis_len: usize) -> T {
        acc
    }

    const ALLOW_EMPTY: bool = true;
    const EMPTY: T = T::ONE;

    #[inline(always)]
    fn reduce_slice(slice: &[T]) -> Option<T> {
        Some(
            slice
                .iter()
                .copied()
                .fold(T::ONE, |acc, value| acc.mul(value)),
        )
    }
}

impl<T: Scalar + FloatElement> AxisReduction<T> for MeanAxis {
    #[inline(always)]
    fn initial(first: T) -> T {
        first
    }

    #[inline(always)]
    fn fold(acc: T, value: T) -> T {
        acc.add(value)
    }

    #[inline(always)]
    fn finalize(acc: T, axis_len: usize) -> T {
        acc.div(T::from_count(axis_len))
    }

    const ALLOW_EMPTY: bool = false;
    const EMPTY: T = T::ZERO;

    #[inline(always)]
    fn reduce_slice(slice: &[T]) -> Option<T> {
        Some(T::sum_slice(slice))
    }
}

impl<T: Scalar> AxisReduction<T> for MinAxis {
    // Seeding from the identity sends the first element through the same
    // NaN-rejecting comparison as every later one; a raw seed would carry a
    // leading NaN through the whole fold.
    #[inline(always)]
    fn initial(first: T) -> T {
        Self::fold(T::MAX_VALUE, first)
    }

    #[inline(always)]
    fn fold(acc: T, value: T) -> T {
        fold_min(acc, value)
    }

    #[inline(always)]
    fn finalize(acc: T, _axis_len: usize) -> T {
        acc
    }

    const ALLOW_EMPTY: bool = false;
    const EMPTY: T = T::ZERO;

    #[inline(always)]
    fn reduce_slice(slice: &[T]) -> Option<T> {
        Some(T::min_slice(slice))
    }
}

impl<T: Scalar> AxisReduction<T> for MaxAxis {
    #[inline(always)]
    fn initial(first: T) -> T {
        Self::fold(T::MIN_VALUE, first)
    }

    #[inline(always)]
    fn fold(acc: T, value: T) -> T {
        fold_max(acc, value)
    }

    #[inline(always)]
    fn finalize(acc: T, _axis_len: usize) -> T {
        acc
    }

    const ALLOW_EMPTY: bool = false;
    const EMPTY: T = T::ZERO;

    #[inline(always)]
    fn reduce_slice(slice: &[T]) -> Option<T> {
        Some(T::max_slice(slice))
    }
}
