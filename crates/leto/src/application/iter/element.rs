//! Logical-order element iteration over array views (leto `iter` /
//! `indexed_iter` / `indexed_iter_mut` parity).
//!
//! Both iterators walk every logical element of a view in row-major order,
//! resolving each element's physical offset through the view's strides — so an
//! arbitrarily strided, transposed, or broadcast view iterates in the same
//! logical order as its contiguous materialization. They are
//! [`ExactSizeIterator`] and [`DoubleEndedIterator`]; the two ends share one
//! `[front, back)` cursor so forward and backward consumption meet exactly once.

use crate::application::view::{ArrayView, ArrayViewMut};
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;

#[inline]
fn odometer_step<const N: usize>(
    index: &mut [usize; N],
    shape: &[usize; N],
    strides: &[isize; N],
    offset: &mut usize,
) {
    for i in (0..N).rev() {
        index[i] += 1;
        if index[i] < shape[i] {
            *offset = (*offset as isize + strides[i]) as usize;
            break;
        }
        *offset = (*offset as isize - (shape[i] - 1) as isize * strides[i]) as usize;
        index[i] = 0;
    }
}

#[inline]
fn odometer_step_back<const N: usize>(
    index: &mut [usize; N],
    shape: &[usize; N],
    strides: &[isize; N],
    offset: &mut usize,
) {
    for i in (0..N).rev() {
        if index[i] > 0 {
            index[i] -= 1;
            *offset = (*offset as isize - strides[i]) as usize;
            break;
        }
        *offset = (*offset as isize + (shape[i] - 1) as isize * strides[i]) as usize;
        index[i] = shape[i] - 1;
    }
}

fn layout_may_alias_mutable_offsets<const N: usize>(layout: &Layout<N>) -> Result<bool> {
    if layout.checked_size()? <= 1 {
        return Ok(false);
    }

    let mut axes = Vec::with_capacity(N);
    for (&dim, &stride) in layout.shape.iter().zip(layout.strides.iter()) {
        if dim <= 1 {
            continue;
        }
        let stride = stride.unsigned_abs();
        if stride == 0 {
            return Ok(true);
        }
        axes.push((stride, dim));
    }
    axes.sort_unstable_by_key(|&(stride, _)| stride);

    let mut covered_span = 0usize;
    for (stride, dim) in axes {
        if stride <= covered_span {
            return Ok(true);
        }
        let axis_span = dim
            .checked_sub(1)
            .and_then(|extent| extent.checked_mul(stride))
            .ok_or(LetoError::Overflow {
                reason: "mutable iterator alias span calculation",
            })?;
        covered_span = covered_span
            .checked_add(axis_span)
            .ok_or(LetoError::Overflow {
                reason: "mutable iterator alias span accumulation",
            })?;
    }

    Ok(false)
}

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
            let start = layout.offset;
            let end = start + layout.size();
            Some(view.data()[start..end].iter())
        } else {
            None
        };
        let back = view.size();
        let (back_index, back_offset) = if back > 0 {
            let mut idx = [0usize; N];
            for (i, item) in idx.iter_mut().enumerate() {
                *item = layout.shape[i] - 1;
            }
            let offset = layout
                .offset_of(idx)
                .expect("invariant: last index is valid");
            (idx, offset)
        } else {
            ([0usize; N], layout.offset)
        };
        Self {
            contiguous_iter,
            data: view.data(),
            layout,
            shape: layout.shape,
            front: 0,
            back,
            front_index: [0usize; N],
            front_offset: layout.offset,
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
                &self.layout.strides,
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
                &self.layout.strides,
                &mut self.back_offset,
            );
            Some(elem)
        }
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
        let (back_index, back_offset) = if back > 0 {
            let mut idx = [0usize; N];
            for (i, item) in idx.iter_mut().enumerate() {
                *item = layout.shape[i] - 1;
            }
            let offset = layout
                .offset_of(idx)
                .expect("invariant: last index is valid");
            (idx, offset)
        } else {
            ([0usize; N], layout.offset)
        };
        Self {
            data: view.data(),
            layout,
            shape: layout.shape,
            front: 0,
            back,
            front_index: [0usize; N],
            front_offset: layout.offset,
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
            &self.layout.strides,
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
            &self.layout.strides,
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
    /// Build a mutable indexed iterator over `view`.
    pub(crate) fn new(view: ArrayViewMut<'a, T, N>) -> Result<Self> {
        let layout = view.layout;
        layout.validate_storage_len(view.len)?;
        if layout_may_alias_mutable_offsets(&layout)? {
            return Err(LetoError::StorageError {
                reason: "indexed_iter_mut requires provably disjoint logical offsets".to_string(),
            });
        }

        let back = layout.size();
        let (back_index, back_offset) = if back > 0 {
            let mut idx = [0usize; N];
            for (i, item) in idx.iter_mut().enumerate() {
                *item = layout.shape[i] - 1;
            }
            let offset = layout
                .offset_of(idx)
                .expect("invariant: last index is valid");
            (idx, offset)
        } else {
            ([0usize; N], layout.offset)
        };
        Ok(Self {
            ptr: view.ptr,
            layout,
            shape: layout.shape,
            front: 0,
            back,
            front_index: [0usize; N],
            front_offset: layout.offset,
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
            &self.layout.strides,
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
            &self.layout.strides,
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

/// `for elem in &view` iterates the view's elements in logical row-major order.
impl<'a, T, const N: usize> IntoIterator for &ArrayView<'a, T, N> {
    type Item = &'a T;
    type IntoIter = ElementIter<'a, T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
