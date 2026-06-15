//! Logical-order element iteration over array views (ndarray `iter` /
//! `indexed_iter` parity).
//!
//! Both iterators walk every logical element of a view in row-major order,
//! resolving each element's physical offset through the view's strides — so an
//! arbitrarily strided, transposed, or broadcast view iterates in the same
//! logical order as its contiguous materialization. They are
//! [`ExactSizeIterator`] and [`DoubleEndedIterator`]; the two ends share one
//! `[front, back)` cursor so forward and backward consumption meet exactly once.

use crate::application::index::index_from_flat;
use crate::application::view::ArrayView;
use crate::domain::layout::Layout;

/// Resolve the element at logical position `flat` through `layout`'s strides.
///
/// `data` is passed by value (it is a `Copy` `&'a [T]`), so the returned
/// reference carries the data lifetime `'a` rather than the borrow of any
/// iterator struct.
#[inline]
fn elem_at<'a, T, const N: usize>(
    data: &'a [T],
    layout: &Layout<N>,
    shape: &[usize; N],
    flat: usize,
) -> &'a T {
    let index = index_from_flat(flat, shape);
    let offset = layout
        .offset_of(index)
        .expect("invariant: logical index is in bounds for the view shape");
    &data[offset]
}

/// Iterator over every element of a view in logical row-major order.
///
/// Yields `&T`. Construct via [`ArrayView::iter`](crate::application::view::ArrayView::iter)
/// or [`Array::iter`](crate::application::array::Array::iter).
pub struct ElementIter<'a, T, const N: usize> {
    data: &'a [T],
    layout: Layout<N>,
    shape: [usize; N],
    front: usize,
    back: usize,
}

impl<'a, T, const N: usize> ElementIter<'a, T, N> {
    /// Build an element iterator over `view`.
    #[inline]
    pub(crate) fn new(view: &ArrayView<'a, T, N>) -> Self {
        let layout = view.layout();
        Self {
            data: view.data(),
            layout,
            shape: layout.shape,
            front: 0,
            back: view.size(),
        }
    }
}

impl<'a, T, const N: usize> Iterator for ElementIter<'a, T, N> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if self.front >= self.back {
            return None;
        }
        let elem = elem_at(self.data, &self.layout, &self.shape, self.front);
        self.front += 1;
        Some(elem)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for ElementIter<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        Some(elem_at(self.data, &self.layout, &self.shape, self.back))
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for ElementIter<'a, T, N> {}

/// Iterator over `(multi-index, element)` pairs in logical row-major order.
///
/// Yields `([usize; N], &T)`. Construct via
/// [`ArrayView::indexed_iter`](crate::application::view::ArrayView::indexed_iter)
/// or [`Array::indexed_iter`](crate::application::array::Array::indexed_iter).
pub struct IndexedIter<'a, T, const N: usize> {
    data: &'a [T],
    layout: Layout<N>,
    shape: [usize; N],
    front: usize,
    back: usize,
}

impl<'a, T, const N: usize> IndexedIter<'a, T, N> {
    /// Build an indexed iterator over `view`.
    #[inline]
    pub(crate) fn new(view: &ArrayView<'a, T, N>) -> Self {
        let layout = view.layout();
        Self {
            data: view.data(),
            layout,
            shape: layout.shape,
            front: 0,
            back: view.size(),
        }
    }
}

impl<'a, T, const N: usize> Iterator for IndexedIter<'a, T, N> {
    type Item = ([usize; N], &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let index = index_from_flat(self.front, &self.shape);
        let elem = elem_at(self.data, &self.layout, &self.shape, self.front);
        self.front += 1;
        Some((index, elem))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for IndexedIter<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        let index = index_from_flat(self.back, &self.shape);
        Some((
            index,
            elem_at(self.data, &self.layout, &self.shape, self.back),
        ))
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for IndexedIter<'a, T, N> {}

/// `for elem in &view` iterates the view's elements in logical row-major order.
impl<'a, T, const N: usize> IntoIterator for &ArrayView<'a, T, N> {
    type Item = &'a T;
    type IntoIter = ElementIter<'a, T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
