//! `IntoIterator` impls for `&ArrayView`, `&Array`, and `&ArrayViewMut`
//! (read-only logical-order traversal via [`ElementIter`]).

use super::element_iter::ElementIter;
use crate::application::array::Array;
use crate::application::view::{ArrayView, ArrayViewMut};
use crate::infrastructure::storage::Storage;

/// `for elem in &view` iterates the view's elements in logical row-major order.
impl<'a, T, const N: usize> IntoIterator for &ArrayView<'a, T, N> {
    type Item = &'a T;
    type IntoIter = ElementIter<'a, T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// `for elem in &array` iterates an owned array in logical row-major order.
///
/// The iterator preserves the array's arbitrary strides and does not require a
/// contiguous copy. Mutable iteration intentionally remains fallible through
/// [`Array::indexed_iter_mut`](crate::application::array::Array::indexed_iter_mut),
/// because an infallible `IntoIterator<Item = &mut T>` implementation could not
/// report rejection of zero-stride aliasing layouts.
impl<'a, T, S, const N: usize> IntoIterator for &'a Array<T, S, N>
where
    S: Storage<T>,
{
    type Item = &'a T;
    type IntoIter = ElementIter<'a, T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.view().iter()
    }
}

/// `for elem in &mutable_view` provides a read-only traversal without exposing
/// mutable aliases. Use [`ArrayViewMut::indexed_iter_mut`] for validated mutable
/// traversal.
impl<'a, T, const N: usize> IntoIterator for &'a ArrayViewMut<'_, T, N> {
    type Item = &'a T;
    type IntoIter = ElementIter<'a, T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_view().iter()
    }
}
