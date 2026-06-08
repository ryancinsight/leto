use crate::domain::error::Result;
use crate::domain::layout::Layout;
use crate::domain::slice::SliceArg;

/// A read-only zero-copy view of an N-dimensional strided array.
pub struct ArrayView<'a, T, const N: usize> {
    layout: Layout<N>,
    data: &'a [T],
}

impl<'a, T, const N: usize> ArrayView<'a, T, N> {
    /// Create a new ArrayView from a layout and raw slice.
    #[inline]
    pub const fn new(layout: Layout<N>, data: &'a [T]) -> Self {
        Self { layout, data }
    }

    /// Returns the shape of the view.
    #[inline]
    pub const fn shape(&self) -> [usize; N] {
        self.layout.shape
    }

    /// Returns the strides of the view.
    #[inline]
    pub const fn strides(&self) -> [isize; N] {
        self.layout.strides
    }

    /// Returns the offset of the view.
    #[inline]
    pub const fn offset(&self) -> usize {
        self.layout.offset
    }

    /// Returns the total logical size of the view.
    #[inline]
    pub fn size(&self) -> usize {
        self.layout.size()
    }

    /// Returns the layout of the view.
    #[inline]
    pub const fn layout(&self) -> Layout<N> {
        self.layout
    }

    /// Returns the raw data slice.
    #[inline]
    pub const fn data(&self) -> &'a [T] {
        self.data
    }

    /// Get a reference to the element at the specified index.
    #[inline]
    pub fn get(&self, index: [usize; N]) -> Result<&T> {
        let offset = self.layout.offset_of(index)?;
        Ok(&self.data[offset])
    }

    /// Slice the view, returning a sub-view.
    #[inline]
    pub fn slice(&self, ranges: &[(usize, usize, isize); N]) -> Result<ArrayView<'a, T, N>> {
        let sliced_layout = self.layout.slice(ranges)?;
        Ok(ArrayView::new(sliced_layout, self.data))
    }

    /// Slice the view with ndarray-style arguments.
    #[inline]
    pub fn slice_with<const M: usize>(&self, args: &[SliceArg]) -> Result<ArrayView<'a, T, M>> {
        let sliced_layout = self.layout.slice_with(args)?;
        Ok(ArrayView::new(sliced_layout, self.data))
    }

    /// Transpose the view by permuting axes.
    #[inline]
    pub fn transpose(&self, axes: [usize; N]) -> Result<ArrayView<'a, T, N>> {
        let transposed_layout = self.layout.transpose(axes)?;
        Ok(ArrayView::new(transposed_layout, self.data))
    }

    /// Broadcast the view to a larger dimensional shape.
    #[inline]
    pub fn broadcast<const M: usize>(
        &self,
        target_shape: [usize; M],
    ) -> Result<ArrayView<'a, T, M>> {
        let broadcasted_layout = self.layout.broadcast(target_shape)?;
        Ok(ArrayView::new(broadcasted_layout, self.data))
    }

    /// Expose the underlying slice if it is C-contiguous.
    #[inline]
    pub fn as_slice(&self) -> Option<&'a [T]> {
        if self.layout.is_c_contiguous() {
            let start = self.layout.offset;
            let end = start + self.layout.size();
            Some(&self.data[start..end])
        } else {
            None
        }
    }
}

// ── ArrayViewMut ──

/// A mutable zero-copy view of an N-dimensional strided array.
pub struct ArrayViewMut<'a, T, const N: usize> {
    layout: Layout<N>,
    data: &'a mut [T],
}

impl<'a, T, const N: usize> ArrayViewMut<'a, T, N> {
    /// Create a new ArrayViewMut from a layout and mutable slice.
    #[inline]
    pub fn new(layout: Layout<N>, data: &'a mut [T]) -> Self {
        Self { layout, data }
    }

    /// Returns the shape of the view.
    #[inline]
    pub const fn shape(&self) -> [usize; N] {
        self.layout.shape
    }

    /// Returns the strides of the view.
    #[inline]
    pub const fn strides(&self) -> [isize; N] {
        self.layout.strides
    }

    /// Returns the offset of the view.
    #[inline]
    pub const fn offset(&self) -> usize {
        self.layout.offset
    }

    /// Returns the total logical size of the view.
    #[inline]
    pub fn size(&self) -> usize {
        self.layout.size()
    }

    /// Returns the layout of the view.
    #[inline]
    pub const fn layout(&self) -> Layout<N> {
        self.layout
    }

    /// Returns the raw data slice as read-only.
    #[inline]
    pub fn data(&self) -> &[T] {
        self.data
    }

    /// Returns the raw mutable data slice.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    /// Get a reference to the element at the specified index.
    #[inline]
    pub fn get(&self, index: [usize; N]) -> Result<&T> {
        let offset = self.layout.offset_of(index)?;
        Ok(&self.data[offset])
    }

    /// Get a mutable reference to the element at the specified index.
    #[inline]
    pub fn get_mut(&mut self, index: [usize; N]) -> Result<&mut T> {
        let offset = self.layout.offset_of(index)?;
        Ok(&mut self.data[offset])
    }

    /// Slice the mutable view, returning a sub-view.
    #[inline]
    pub fn slice_mut(self, ranges: &[(usize, usize, isize); N]) -> Result<ArrayViewMut<'a, T, N>> {
        let sliced_layout = self.layout.slice(ranges)?;
        Ok(ArrayViewMut::new(sliced_layout, self.data))
    }

    /// Slice the mutable view with ndarray-style arguments.
    #[inline]
    pub fn slice_with_mut<const M: usize>(
        self,
        args: &[SliceArg],
    ) -> Result<ArrayViewMut<'a, T, M>> {
        let sliced_layout = self.layout.slice_with(args)?;
        Ok(ArrayViewMut::new(sliced_layout, self.data))
    }

    /// Transpose the mutable view by permuting axes.
    #[inline]
    pub fn transpose_mut(self, axes: [usize; N]) -> Result<ArrayViewMut<'a, T, N>> {
        let transposed_layout = self.layout.transpose(axes)?;
        Ok(ArrayViewMut::new(transposed_layout, self.data))
    }

    /// Broadcast the mutable view to a larger dimensional shape.
    /// Note: Broadcasted views cannot be written to safely without stride collision if dimensions are expanded (strides are 0).
    /// Therefore, returning a mutable broadcasted view must be used with caution, but is permitted here.
    #[inline]
    pub fn broadcast_mut<const M: usize>(
        self,
        target_shape: [usize; M],
    ) -> Result<ArrayViewMut<'a, T, M>> {
        let broadcasted_layout = self.layout.broadcast(target_shape)?;
        Ok(ArrayViewMut::new(broadcasted_layout, self.data))
    }

    /// Expose the underlying slice if it is C-contiguous.
    #[inline]
    pub fn as_slice(&self) -> Option<&[T]> {
        if self.layout.is_c_contiguous() {
            let start = self.layout.offset;
            let end = start + self.layout.size();
            Some(&self.data[start..end])
        } else {
            None
        }
    }

    /// Expose the underlying mutable slice if it is C-contiguous.
    #[inline]
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        if self.layout.is_c_contiguous() {
            let start = self.layout.offset;
            let end = start + self.layout.size();
            Some(&mut self.data[start..end])
        } else {
            None
        }
    }
}
