//! Subview iteration along a single axis (read-only and mutable).

use crate::application::view::{ArrayView, ArrayViewMut};
use crate::domain::layout::Layout;
use crate::domain::remove_axis::RemoveAxis;

/// An iterator yielding read-only subviews along a given axis.
pub struct AxisIter<'a, T, const N: usize, const M: usize> {
    data: &'a [T],
    sub_shape: [usize; M],
    sub_strides: [isize; M],
    step_stride: isize,
    current_offset: isize,
    index: usize,
    len: usize,
}

impl<'a, T, const N: usize, const M: usize> AxisIter<'a, T, N, M> {
    /// Create a new AxisIter from an ArrayView and an axis.
    pub fn new<R>(
        view: &ArrayView<'a, T, N>,
        axis: usize,
        marker: R,
    ) -> crate::domain::error::Result<Self>
    where
        R: RemoveAxis<N, SmallerShape = [usize; M], SmallerStrides = [isize; M]>,
    {
        if axis >= N {
            return Err(crate::domain::error::LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank {N}"),
            });
        }
        let len = view.shape()[axis];
        let sub_shape = marker.remove_shape(view.shape(), axis)?;
        let sub_strides = marker.remove_strides(view.strides(), axis)?;
        let step_stride = view.strides()[axis];
        let current_offset = view.offset() as isize;
        view.layout().validate_storage_len(view.data().len())?;

        Ok(Self {
            data: view.data(),
            sub_shape,
            sub_strides,
            step_stride,
            current_offset,
            index: 0,
            len,
        })
    }
}

impl<'a, T, const N: usize, const M: usize> Iterator for AxisIter<'a, T, N, M> {
    type Item = ArrayView<'a, T, M>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            None
        } else {
            let offset = self.current_offset as usize;
            let layout = Layout::from_parts_unchecked(self.sub_shape, self.sub_strides, offset);
            self.current_offset += self.step_stride;
            self.index += 1;
            Some(ArrayView::new(layout, self.data))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const N: usize, const M: usize> ExactSizeIterator for AxisIter<'a, T, N, M> {}

/// An iterator yielding mutable subviews along a given axis.
pub struct AxisIterMut<'a, T, const N: usize, const M: usize> {
    ptr: *mut T,
    sub_shape: [usize; M],
    sub_strides: [isize; M],
    step_stride: isize,
    current_offset: isize,
    index: usize,
    len: usize,
    _marker: std::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T, const N: usize, const M: usize> AxisIterMut<'a, T, N, M> {
    /// Create a new AxisIterMut from an ArrayViewMut and an axis.
    ///
    /// # Errors
    /// [`crate::domain::error::LetoError`] if `axis >= N`, the layout does not
    /// fit its storage, or the layout is not injective (any aliasing,
    /// zero-stride or otherwise), which would make distinct subviews share
    /// physical elements.
    pub fn new<R>(
        view: ArrayViewMut<'a, T, N>,
        axis: usize,
        marker: R,
    ) -> crate::domain::error::Result<Self>
    where
        R: RemoveAxis<N, SmallerShape = [usize; M], SmallerStrides = [isize; M]>,
    {
        if axis >= N {
            return Err(crate::domain::error::LetoError::StorageError {
                reason: format!("Axis {axis} out of bounds for rank {N}"),
            });
        }
        let len = view.shape()[axis];
        let sub_shape = marker.remove_shape(view.shape(), axis)?;
        let sub_strides = marker.remove_strides(view.strides(), axis)?;
        let step_stride = view.strides()[axis];
        let current_offset = view.offset() as isize;
        view.layout().validate_storage_len(view.len)?;
        // Distinct subviews are disjoint only under full injectivity: a
        // zero-stride check alone admits layouts (e.g. shape [2, 2], strides
        // [1, 1]) whose subviews map onto shared physical elements.
        if !view.layout().is_injective()? {
            return Err(crate::domain::error::LetoError::StorageError {
                reason: "axis mutable iterator requires an injective (non-aliasing) layout"
                    .to_string(),
            });
        }
        // Raw base pointer, not `data_mut()`: materializing the whole window
        // as `&mut [T]` is forbidden when `view` is itself an iterator-yielded
        // sub-view over an interleaved layout (nested iteration stays legal).
        let ptr = view.ptr.as_ptr();

        Ok(Self {
            ptr,
            sub_shape,
            sub_strides,
            step_stride,
            current_offset,
            index: 0,
            len,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<'a, T, const N: usize, const M: usize> Iterator for AxisIterMut<'a, T, N, M> {
    type Item = ArrayViewMut<'a, T, M>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            None
        } else {
            let offset = self.current_offset as usize;
            let layout = Layout::from_parts_unchecked(self.sub_shape, self.sub_strides, offset);
            self.current_offset += self.step_stride;
            self.index += 1;

            let (min_offset, max_offset) = layout.min_max_offsets();
            let span_len = max_offset - min_offset + 1;
            let adjusted_layout = Layout::from_parts_unchecked(
                layout.shape(),
                layout.strides(),
                layout.offset() - min_offset,
            );

            // SAFETY: the constructor validated the parent layout against its
            // storage, and every subview offset is a parent element offset, so
            // `[min_offset, max_offset]` lies inside the parent window and
            // `ptr.add(min_offset)` is in bounds and non-null. Distinct
            // subviews address disjoint elements (injective parent), so
            // per-element access through the yielded view cannot alias a
            // sibling; whole-window slice access is gated by `window_shared`.
            unsafe {
                Some(ArrayViewMut {
                    layout: adjusted_layout,
                    ptr: std::ptr::NonNull::new_unchecked(self.ptr.add(min_offset)),
                    len: span_len,
                    // An interleaved subview window still contains sibling
                    // elements; only a dense window (span == logical size) is
                    // exclusively owned.
                    window_shared: span_len != adjusted_layout.size(),
                    _marker: std::marker::PhantomData,
                })
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const N: usize, const M: usize> ExactSizeIterator for AxisIterMut<'a, T, N, M> {}
