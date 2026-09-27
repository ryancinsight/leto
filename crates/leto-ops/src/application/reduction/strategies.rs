//! [`AxisReduction`] strategy marker types: `Sum`/`Product`/`Mean`/`Min`/`Max`.

use crate::domain::scalar::Scalar;

mod sealed {
    pub trait Sealed {}
}

/// Zero-sized axis-reduction contract.
pub trait AxisReduction<T: Scalar>: sealed::Sealed + Copy + Send + Sync + 'static {
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
#[derive(Clone, Copy, Debug, Default)]
pub struct MeanAxis;

/// Minimum axis-reduction marker.
///
/// NaN lanes are ignored on every route: the fold starts from
/// `T::MAX_VALUE` and only a value that compares below the accumulator
/// replaces it, and the contiguous route inherits the same contract from
/// [`Scalar::min_slice`]. An all-NaN axis therefore reduces to
/// `T::MAX_VALUE`, the identity.
#[derive(Clone, Copy, Debug, Default)]
pub struct MinAxis;

/// Maximum axis-reduction marker.
///
/// The mirror of [`MinAxis`]: NaN lanes are ignored, the fold starts from
/// `T::MIN_VALUE`, and an all-NaN axis reduces to that identity.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaxAxis;

impl sealed::Sealed for SumAxis {}
impl sealed::Sealed for ProductAxis {}
impl sealed::Sealed for MeanAxis {}
impl sealed::Sealed for MinAxis {}
impl sealed::Sealed for MaxAxis {}

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

impl<T: Scalar> AxisReduction<T> for MeanAxis {
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
        acc.div(T::from_usize(axis_len))
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
        if value < acc {
            value
        } else {
            acc
        }
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
        if value > acc {
            value
        } else {
            acc
        }
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
