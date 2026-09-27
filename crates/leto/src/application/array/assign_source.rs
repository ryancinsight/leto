use super::types::Array;
use crate::application::view::ArrayView;
use crate::domain::error::Result;
use crate::infrastructure::storage::Storage;

/// Read-only source for assigning values into an array.
pub trait AssignSource<T, const N: usize> {
    /// Shape of the assignment source.
    fn assign_shape(&self) -> [usize; N];

    /// Value at a logical index.
    ///
    /// # Errors
    /// Returns [`LetoError`](crate::domain::error::LetoError) when the index is
    /// out of bounds for the source.
    fn assign_get(&self, index: [usize; N]) -> Result<&T>;

    /// Borrow the source as a validated strided view when its representation
    /// supports zero-copy traversal.
    ///
    /// The view must have the shape returned by [`Self::assign_shape`] and each
    /// logical element must equal the corresponding [`Self::assign_get`] value.
    /// Implementors outside Leto may retain the default checked-index route.
    #[doc(hidden)]
    #[inline]
    fn assign_view(&self) -> Option<ArrayView<'_, T, N>> {
        None
    }
}

impl<T, S, const N: usize> AssignSource<T, N> for Array<T, S, N>
where
    S: Storage<T>,
{
    #[inline]
    fn assign_shape(&self) -> [usize; N] {
        self.shape()
    }

    #[inline]
    fn assign_get(&self, index: [usize; N]) -> Result<&T> {
        self.get(index)
    }

    #[inline]
    fn assign_view(&self) -> Option<ArrayView<'_, T, N>> {
        Some(self.view())
    }
}

impl<T, const N: usize> AssignSource<T, N> for ArrayView<'_, T, N> {
    #[inline]
    fn assign_shape(&self) -> [usize; N] {
        self.shape()
    }

    #[inline]
    fn assign_get(&self, index: [usize; N]) -> Result<&T> {
        self.get(index)
    }

    #[inline]
    fn assign_view(&self) -> Option<ArrayView<'_, T, N>> {
        Some(self.reborrow())
    }
}
