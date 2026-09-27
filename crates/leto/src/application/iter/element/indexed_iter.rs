//! `(multi-index, element)` pair iteration: [`IndexedIter`] (shared) and
//! [`IndexedIterMut`] (mutable, provably disjoint).

use super::odometer::{last_logical_cursor, odometer_step, odometer_step_back, MutableLayoutProof};
use crate::application::index::index_from_flat;
use crate::application::view::{ArrayView, ArrayViewMut};
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;

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
    front_index: [usize; N],
    front_offset: usize,
    back_index: [usize; N],
    back_offset: usize,
}

impl<'a, T, const N: usize> IndexedIter<'a, T, N> {
    /// Build an indexed iterator over `view`.
    #[inline]
    pub(crate) fn new(view: &ArrayView<'a, T, N>) -> Self {
        let layout = view.layout();
        let back = view.size();
        let (back_index, back_offset) = last_logical_cursor(&layout, back);
        Self {
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

impl<'a, T, const N: usize> Iterator for IndexedIter<'a, T, N> {
    type Item = ([usize; N], &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let index = self.front_index;
        let elem = &self.data[self.front_offset];
        odometer_step(
            &mut self.front_index,
            &self.shape,
            &self.layout.strides(),
            &mut self.front_offset,
        );
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
        let index = self.back_index;
        let elem = &self.data[self.back_offset];
        odometer_step_back(
            &mut self.back_index,
            &self.shape,
            &self.layout.strides(),
            &mut self.back_offset,
        );
        Some((index, elem))
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for IndexedIter<'a, T, N> {}

/// Mutable iterator over `(multi-index, element)` pairs in logical row-major order.
///
/// Yields `([usize; N], &mut T)`. Construct via
/// [`ArrayViewMut::indexed_iter_mut`](crate::application::view::ArrayViewMut::indexed_iter_mut)
/// or [`Array::indexed_iter_mut`](crate::application::array::Array::indexed_iter_mut).
pub struct IndexedIterMut<'a, T, const N: usize> {
    ptr: std::ptr::NonNull<T>,
    layout: Layout<N>,
    shape: [usize; N],
    front: usize,
    back: usize,
    front_index: [usize; N],
    front_offset: usize,
    back_index: [usize; N],
    back_offset: usize,
    _marker: std::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T, const N: usize> IndexedIterMut<'a, T, N> {
    /// Build a mutable indexed iterator over the complete logical view.
    pub(crate) fn new(view: ArrayViewMut<'a, T, N>) -> Result<Self> {
        let end = view.layout.size();
        Self::new_range(view, 0, end)
    }

    /// Build a mutable indexed iterator over a validated logical range.
    ///
    /// The range is expressed in row-major logical positions, not physical
    /// storage offsets. It is the primitive used by
    /// [`TaskPartitionsMut`](super::task_partition::TaskPartitionsMut).
    pub(crate) fn new_range(
        view: ArrayViewMut<'a, T, N>,
        start: usize,
        end: usize,
    ) -> Result<Self> {
        Self::from_proof(
            MutableLayoutProof::new(
                view,
                "indexed_iter_mut requires provably disjoint logical offsets",
            )?,
            start,
            end,
        )
    }

    pub(super) fn from_proof(
        proof: MutableLayoutProof<'a, T, N>,
        start: usize,
        end: usize,
    ) -> Result<Self> {
        let MutableLayoutProof {
            ptr,
            layout,
            storage_len: _,
            _marker: _,
        } = proof;
        let size = layout.size();
        if start > end || end > size {
            return Err(LetoError::OutOfBounds {
                index: vec![start, end],
                shape: vec![size],
            });
        }
        let (front_index, front_offset, back_index, back_offset) = if start < end {
            let front_index = index_from_flat(start, &layout.shape());
            let back_index = index_from_flat(end - 1, &layout.shape());
            let front_offset = layout
                .offset_of(front_index)
                .expect("invariant: range start is a valid logical index");
            let back_offset = layout
                .offset_of(back_index)
                .expect("invariant: range end is a valid logical index");
            (front_index, front_offset, back_index, back_offset)
        } else {
            ([0usize; N], layout.offset(), [0usize; N], layout.offset())
        };

        Ok(Self {
            ptr,
            layout,
            shape: layout.shape(),
            front: 0,
            back: end - start,
            front_index,
            front_offset,
            back_index,
            back_offset,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<'a, T, const N: usize> Iterator for IndexedIterMut<'a, T, N> {
    type Item = ([usize; N], &'a mut T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let index = self.front_index;
        let offset = self.front_offset;
        odometer_step(
            &mut self.front_index,
            &self.shape,
            &self.layout.strides(),
            &mut self.front_offset,
        );
        self.front += 1;
        // SAFETY: construction validates storage bounds and rejects layouts
        // whose logical indices can alias the same physical offset. The shared
        // front/back cursor yields each logical index at most once.
        let elem = unsafe { &mut *self.ptr.as_ptr().add(offset) };
        Some((index, elem))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const N: usize> DoubleEndedIterator for IndexedIterMut<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back -= 1;
        let index = self.back_index;
        let offset = self.back_offset;
        odometer_step_back(
            &mut self.back_index,
            &self.shape,
            &self.layout.strides(),
            &mut self.back_offset,
        );
        // SAFETY: construction validates storage bounds and rejects layouts
        // whose logical indices can alias the same physical offset. The shared
        // front/back cursor yields each logical index at most once.
        let elem = unsafe { &mut *self.ptr.as_ptr().add(offset) };
        Some((index, elem))
    }
}

impl<'a, T, const N: usize> ExactSizeIterator for IndexedIterMut<'a, T, N> {}

// SAFETY: construction validates storage bounds and injectivity, and the
// iterator owns the exclusive mutable traversal token for its range.
unsafe impl<T: Send, const N: usize> Send for IndexedIterMut<'_, T, N> {}
