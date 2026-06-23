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
            let layout = Layout::new(self.sub_shape, self.sub_strides, offset);
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
    pub fn new<R>(
        mut view: ArrayViewMut<'a, T, N>,
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
        let data_len = view.data().len();
        view.layout().validate_storage_len(data_len)?;
        if view.layout().has_zero_stride_aliasing() {
            return Err(crate::domain::error::LetoError::StorageError {
                reason: "axis mutable iterator requires non-aliasing layout".to_string(),
            });
        }
        let ptr = view.data_mut().as_mut_ptr();

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
            let layout = Layout::new(self.sub_shape, self.sub_strides, offset);
            self.current_offset += self.step_stride;
            self.index += 1;

            let (min_offset, max_offset) = layout.min_max_offsets();
            let span_len = max_offset - min_offset + 1;
            let adjusted_layout =
                Layout::new(layout.shape, layout.strides, layout.offset - min_offset);

            unsafe {
                Some(ArrayViewMut {
                    layout: adjusted_layout,
                    ptr: std::ptr::NonNull::new_unchecked(self.ptr.add(min_offset)),
                    len: span_len,
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
