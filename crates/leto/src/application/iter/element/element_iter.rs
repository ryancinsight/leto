//! Plain element iteration: [`ElementIter`] (shared) and [`ElementIterMut`]
//! (mutable, delegating to [`IndexedIterMut`]).

use super::indexed_iter::IndexedIterMut;
use super::odometer::{last_logical_cursor, odometer_step, odometer_step_back};
use crate::application::view::ArrayView;
use crate::domain::layout::Layout;

/// Iterator over every element of a view in logical row-major order.
///
/// Yields `&T`. Construct via [`ArrayView::iter`](crate::application::view::ArrayView::iter)
/// or [`Array::iter`](crate::application::array::Array::iter).
pub struct ElementIter<'a, T, const N: usize> {
    contiguous_iter: Option<std::slice::Iter<'a, T>>,
    data: &'a [T],
    layout: Layout<N>,
    shape: [usize; N],
    front: usize,
    back: usize,
    front_index: [usize; N],
    front_offset: usize,
    back_index: [usize; N],
    back_offset: usize,
}

impl<'a, T, const N: usize> ElementIter<'a, T, N> {
    /// Build an element iterator over `view`.
    #[inline]
    pub(crate) fn new(view: &ArrayView<'a, T, N>) -> Self {
        let layout = view.layout();
        let contiguous_iter = if layout.is_c_dense() {
            let start = layout.offset();
            let end = start + layout.size();
            Some(view.data()[start..end].iter())
        } else {
            None
        };
        let back = view.size();
        let (back_index, back_offset) = last_logical_cursor(&layout, back);
        Self {
            contiguous_iter,
            data: view.data(),
            layout,
            shape: layout.shape(),
            front: 0,
            back,
            front_index: [0usize; N],
            front_offset: layout.offset(),
            back_index,
            back_offset,
        }
    }
}

impl<'a, T, const N: usize> Iterator for ElementIter<'a, T, N> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if let Some(ref mut iter) = self.contiguous_iter {
            iter.next()
        } else {
            if self.front >= self.back {
                return None;
            }
            let elem = &self.data[self.front_offset];
            odometer_step(
                &mut self.front_index,
                &self.shape,
                &self.layout.strides(),
                &mut self.front_offset,
            );
            self.front += 1;
            Some(elem)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(ref iter) = self.contiguous_iter {
            iter.size_hint()
        } else {
            let remaining = self.back - self.front;
            (remaining, Some(remaining))
        }
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for ElementIter<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        if let Some(ref mut iter) = self.contiguous_iter {
            iter.next_back()
        } else {
            if self.front >= self.back {
                return None;
            }
            self.back -= 1;
            let elem = &self.data[self.back_offset];
            odometer_step_back(
                &mut self.back_index,
                &self.shape,
                &self.layout.strides(),
                &mut self.back_offset,
            );
            Some(elem)
        }
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for ElementIter<'a, T, N> {}

/// Fallible mutable iterator over logical row-major elements of a view.
///
/// Construction validates storage reachability and logical-offset injectivity
/// before any mutable reference can escape. This makes arbitrary positive and
/// negative strides safe while rejecting zero-stride or otherwise aliased
/// layouts. Construct through [`ArrayViewMut::try_iter_mut`](crate::application::view::ArrayViewMut::try_iter_mut) or
/// [`Array::try_iter_mut`](crate::application::array::Array::try_iter_mut).
pub struct ElementIterMut<'a, T, const N: usize> {
    inner: IndexedIterMut<'a, T, N>,
}

impl<'a, T, const N: usize> ElementIterMut<'a, T, N> {
    /// Build a plain mutable iterator from an already validated indexed iterator.
    #[inline]
    pub(crate) fn from_indexed(inner: IndexedIterMut<'a, T, N>) -> Self {
        Self { inner }
    }
}

impl<'a, T, const N: usize> Iterator for ElementIterMut<'a, T, N> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, value)| value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for ElementIterMut<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(_, value)| value)
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for ElementIterMut<'a, T, N> {}

// SAFETY: ElementIterMut delegates to IndexedIterMut, whose construction and
// range proof prevent duplicate mutable references.
unsafe impl<T: Send, const N: usize> Send for ElementIterMut<'_, T, N> {}
