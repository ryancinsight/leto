//! Checked RHS width metadata, independent of arithmetic and scheduling.

use core::fmt::Debug;
use leto::{LetoError, Result};

mod sealed {
    pub trait Width {}
}

/// Sealed representation of a product's RHS column count.
///
/// Checked construction proves `get(checked(n)?) == n`. The row kernel relies
/// on this law for its address bounds, so implementations remain sealed.
/// Dynamic metadata stores usize; fixed metadata stores unit and monomorphizes.
pub trait ProductWidth: sealed::Width + Debug + Send + Sync + 'static {
    /// Runtime representation: usize for dynamic width, unit for fixed width.
    type Value: Copy + Debug + Send + Sync + 'static;

    /// Validate a supplied width before accepting its metadata.
    ///
    /// # Errors
    /// [`LetoError::ShapeMismatch`] when a fixed width differs from the input.
    fn checked(columns: usize) -> Result<Self::Value>;

    /// Recover exactly the supplied and checked width.
    #[must_use]
    fn get(value: Self::Value) -> usize;
}

/// Runtime RHS width (the existing product default).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DynamicWidth;
impl sealed::Width for DynamicWidth {}
impl ProductWidth for DynamicWidth {
    type Value = usize;

    #[inline]
    fn checked(columns: usize) -> Result<Self::Value> {
        // Every usize is a valid dimension; product counts are checked next.
        Ok(columns)
    }

    #[inline]
    fn get(value: Self::Value) -> usize {
        value
    }
}

/// Statically known RHS width, with no runtime metadata storage.
///
/// For example, a vector product selects `FixedWidth<1>`. Preparation still
/// checks the supplied dimension; this policy never bypasses shape validation.
///
/// ```
/// use leto_ops::{CsrProduct, CsrView, FixedWidth, ScalarStrategy};
/// let view = CsrView::from_slices(&[3i64], &[0], &[0, 1], 1, 1)?;
/// let mut output = [-1];
/// CsrProduct::<_, _, _, ScalarStrategy, FixedWidth<1>>::prepare(view, &[7], 1)?
///     .execute(&mut output)?;
/// assert_eq!(output, [21]);
/// # Ok::<(), leto::LetoError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FixedWidth<const N: usize>;
impl<const N: usize> sealed::Width for FixedWidth<N> {}
impl<const N: usize> ProductWidth for FixedWidth<N> {
    type Value = ();

    #[inline]
    fn checked(columns: usize) -> Result<Self::Value> {
        if columns != N {
            return Err(LetoError::ShapeMismatch {
                lhs: vec![columns],
                rhs: vec![N],
            });
        }
        Ok(())
    }

    #[inline]
    fn get((): Self::Value) -> usize {
        N
    }
}
