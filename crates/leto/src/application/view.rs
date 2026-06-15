use crate::application::array::Array;
use crate::application::index::index_from_flat;
use crate::application::iter::{ElementIter, IndexedIter, Windows};
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::domain::slice::SliceArg;
use crate::infrastructure::storage::VecStorage;

/// Computes the physical `[offset, offset + size)` range covered by a layout
/// whose elements form a single dense block. Returns `None` only on size
/// overflow. Shared by the contiguous-slice accessors of both view types.
#[inline]
fn dense_block_range<const N: usize>(layout: &Layout<N>) -> Option<core::ops::Range<usize>> {
    let start = layout.offset;
    let end = start.checked_add(layout.checked_size().ok()?)?;
    Some(start..end)
}

/// A read-only zero-copy view of an N-dimensional strided array.
pub struct ArrayView<'a, T, const N: usize> {
    pub(crate) layout: Layout<N>,
    pub(crate) data: &'a [T],
}

impl<'a, T, const N: usize> ArrayView<'a, T, N> {
    /// Create a new ArrayView from a layout and raw slice.
    #[inline]
    pub const fn new(layout: Layout<N>, data: &'a [T]) -> Self {
        Self { layout, data }
    }

    /// Create a bounds-checked ArrayView from a layout and raw slice.
    #[inline]
    pub fn try_new(layout: Layout<N>, data: &'a [T]) -> Result<Self> {
        layout.validate_storage_len(data.len())?;
        Ok(Self { layout, data })
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

    /// Iterator over the view's elements in logical row-major order
    /// (ndarray `iter` parity). The iterator borrows the view's data for `'a`,
    /// so it outlives a temporary view produced by `array.view().iter()`.
    #[inline]
    pub fn iter(&self) -> ElementIter<'a, T, N> {
        ElementIter::new(self)
    }

    /// Iterator over `(multi-index, &element)` pairs in logical row-major order
    /// (ndarray `indexed_iter` parity).
    #[inline]
    pub fn indexed_iter(&self) -> IndexedIter<'a, T, N> {
        IndexedIter::new(self)
    }

    /// Zero-copy iterator over every sliding window of shape `window_shape`
    /// (ndarray `windows` parity). Each yielded view shares this view's strides
    /// and backing storage.
    ///
    /// # Errors
    /// [`LetoError`] if any `window_shape[i]` is `0` or exceeds `shape[i]`.
    #[inline]
    pub fn windows(&self, window_shape: [usize; N]) -> Result<Windows<'a, T, N>> {
        Windows::new(self, window_shape)
    }

    /// Get a reference to the element at the specified index.
    #[inline]
    pub fn get(&self, index: [usize; N]) -> Result<&T> {
        let offset = self.layout.offset_of(index)?;
        if offset >= self.data.len() {
            return Err(LetoError::StorageError {
                reason: format!(
                    "physical offset {offset} exceeds backing slice length {}",
                    self.data.len()
                ),
            });
        }
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

    /// Reinterpret this view with a new shape without copying.
    ///
    /// The current layout must be dense row-major and the new shape must have
    /// the same logical element count.
    #[inline]
    pub fn reshape<const M: usize>(&self, shape: [usize; M]) -> Result<ArrayView<'a, T, M>> {
        let reshaped_layout = self.layout.reshape(shape)?;
        Ok(ArrayView::new(reshaped_layout, self.data))
    }

    /// Named alias for [`transpose`](Self::transpose).
    #[inline]
    pub fn permute(&self, axes: [usize; N]) -> Result<ArrayView<'a, T, N>> {
        self.transpose(axes)
    }

    /// Materialize this view into C-contiguous row-major storage.
    ///
    /// Dense row-major views clone the exposed slice. Strided, transposed, or
    /// broadcasted views are copied in logical row-major order.
    pub fn to_contiguous(&self) -> Array<T, VecStorage<T>, N>
    where
        T: Clone,
    {
        let data = match self.as_slice() {
            Some(slice) => slice.to_vec(),
            None => {
                let size = self.layout.size();
                let shape = self.shape();
                let mut values = Vec::with_capacity(size);
                for flat_idx in 0..size {
                    let index = index_from_flat(flat_idx, &shape);
                    values.push(self.get(index).expect("validated logical index").clone());
                }
                values
            }
        };
        Array::<T, VecStorage<T>, N>::from_shape_vec(self.shape(), data)
            .expect("logical row-major materialization has matching shape and storage")
    }

    /// Returns true when the view is canonically C-contiguous at offset 0.
    #[inline]
    pub fn is_c_contiguous(&self) -> bool {
        self.layout.is_c_contiguous()
    }

    /// Returns true when the view is canonically Fortran-contiguous at offset 0.
    #[inline]
    pub fn is_f_contiguous(&self) -> bool {
        self.layout.is_f_contiguous()
    }

    /// Returns true when the view's elements occupy a dense block in some
    /// memory order (C or F), independent of offset.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Expose the underlying slice if the elements form a dense row-major
    /// (C-order) block, independent of offset.
    #[inline]
    pub fn as_slice(&self) -> Option<&'a [T]> {
        if self.layout.is_c_dense() {
            self.data.get(dense_block_range(&self.layout)?)
        } else {
            None
        }
    }

    /// Expose the underlying slice if the elements form a dense block in some
    /// memory order (C or F), independent of offset. The returned slice is in
    /// physical memory order, matching `ndarray::as_slice_memory_order`.
    #[inline]
    pub fn as_slice_memory_order(&self) -> Option<&'a [T]> {
        if self.layout.is_contiguous() {
            self.data.get(dense_block_range(&self.layout)?)
        } else {
            None
        }
    }

    /// Return an iterator yielding read-only subviews of rank `M` (where `M = N - 1`) along `axis`.
    #[inline]
    pub fn axis_iter<const M: usize>(
        &self,
        axis: usize,
    ) -> Result<crate::application::iter::AxisIter<'_, T, N, M>>
    where
        crate::domain::remove_axis::RankMarker<N>: crate::domain::remove_axis::RemoveAxis<
            N,
            SmallerShape = [usize; M],
            SmallerStrides = [isize; M],
        >,
    {
        crate::application::iter::AxisIter::new(
            self,
            axis,
            crate::domain::remove_axis::RankMarker::<N>,
        )
    }

    /// Return an iterator yielding read-only 1-D lane views *along* `axis`
    /// (ndarray `lanes` parity; `M = N - 1` is the complement rank). Dual of
    /// [`axis_iter`](Self::axis_iter): one lane per complement coordinate.
    ///
    /// # Errors
    /// [`LetoError`] if `axis >= N` or the layout does not fit its storage.
    #[inline]
    pub fn lanes<const M: usize>(
        &self,
        axis: usize,
    ) -> Result<crate::application::iter::Lanes<'a, T, N, M>>
    where
        crate::domain::remove_axis::RankMarker<N>: crate::domain::remove_axis::RemoveAxis<
            N,
            SmallerShape = [usize; M],
            SmallerStrides = [isize; M],
        >,
    {
        crate::application::iter::Lanes::new(
            self,
            axis,
            crate::domain::remove_axis::RankMarker::<N>,
        )
    }

    /// Reborrow the read-only view with a shorter lifetime.
    #[inline]
    pub fn reborrow(&self) -> ArrayView<'_, T, N> {
        ArrayView::new(self.layout, self.data)
    }
}

// ── ArrayViewMut ──

/// A mutable zero-copy view of an N-dimensional strided array.
pub struct ArrayViewMut<'a, T, const N: usize> {
    pub(crate) layout: Layout<N>,
    pub(crate) data: &'a mut [T],
}

impl<'a, T, const N: usize> ArrayViewMut<'a, T, N> {
    /// Create a new ArrayViewMut from a layout and mutable slice.
    #[inline]
    pub fn new(layout: Layout<N>, data: &'a mut [T]) -> Self {
        Self { layout, data }
    }

    /// Create a bounds-checked ArrayViewMut from a layout and mutable slice.
    #[inline]
    pub fn try_new(layout: Layout<N>, data: &'a mut [T]) -> Result<Self> {
        layout.validate_storage_len(data.len())?;
        Ok(Self { layout, data })
    }

    /// Reborrow the mutable view with a shorter lifetime.
    #[inline]
    pub fn reborrow(&mut self) -> ArrayViewMut<'_, T, N> {
        ArrayViewMut::new(self.layout, self.data)
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
        if offset >= self.data.len() {
            return Err(LetoError::StorageError {
                reason: format!(
                    "physical offset {offset} exceeds backing slice length {}",
                    self.data.len()
                ),
            });
        }
        Ok(&self.data[offset])
    }

    /// Get a mutable reference to the element at the specified index.
    #[inline]
    pub fn get_mut(&mut self, index: [usize; N]) -> Result<&mut T> {
        let offset = self.layout.offset_of(index)?;
        if offset >= self.data.len() {
            return Err(LetoError::StorageError {
                reason: format!(
                    "physical offset {offset} exceeds backing slice length {}",
                    self.data.len()
                ),
            });
        }
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
    ///
    /// Returns an error when broadcasting would introduce zero-stride aliasing.
    #[inline]
    pub fn broadcast_mut<const M: usize>(
        self,
        target_shape: [usize; M],
    ) -> Result<ArrayViewMut<'a, T, M>> {
        let broadcasted_layout = self.layout.broadcast(target_shape)?;
        if broadcasted_layout.has_zero_stride_aliasing() {
            return Err(LetoError::IncompatibleBroadcast {
                from: self.layout.shape.to_vec(),
                to: target_shape.to_vec(),
            });
        }
        Ok(ArrayViewMut::new(broadcasted_layout, self.data))
    }

    /// Reinterpret this mutable view with a new shape without copying.
    ///
    /// The current layout must be dense row-major and the new shape must have
    /// the same logical element count.
    #[inline]
    pub fn reshape_mut<const M: usize>(self, shape: [usize; M]) -> Result<ArrayViewMut<'a, T, M>> {
        let reshaped_layout = self.layout.reshape(shape)?;
        Ok(ArrayViewMut::new(reshaped_layout, self.data))
    }

    /// Named alias for [`transpose_mut`](Self::transpose_mut).
    #[inline]
    pub fn permute_mut(self, axes: [usize; N]) -> Result<ArrayViewMut<'a, T, N>> {
        self.transpose_mut(axes)
    }

    /// Materialize this mutable view into C-contiguous row-major storage.
    pub fn to_contiguous(&self) -> Array<T, VecStorage<T>, N>
    where
        T: Clone,
    {
        let view = ArrayView::new(self.layout, self.data());
        view.to_contiguous()
    }

    /// Returns true when the view is canonically C-contiguous at offset 0.
    #[inline]
    pub fn is_c_contiguous(&self) -> bool {
        self.layout.is_c_contiguous()
    }

    /// Returns true when the view is canonically Fortran-contiguous at offset 0.
    #[inline]
    pub fn is_f_contiguous(&self) -> bool {
        self.layout.is_f_contiguous()
    }

    /// Returns true when the view's elements occupy a dense block in some
    /// memory order (C or F), independent of offset.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Expose the underlying slice if the elements form a dense row-major
    /// (C-order) block, independent of offset.
    #[inline]
    pub fn as_slice(&self) -> Option<&[T]> {
        if self.layout.is_c_dense() {
            self.data.get(dense_block_range(&self.layout)?)
        } else {
            None
        }
    }

    /// Expose the underlying mutable slice if the elements form a dense
    /// row-major (C-order) block, independent of offset.
    #[inline]
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        if self.layout.is_c_dense() {
            self.data.get_mut(dense_block_range(&self.layout)?)
        } else {
            None
        }
    }

    /// Expose the underlying slice if the elements form a dense block in some
    /// memory order (C or F), independent of offset. Physical memory order.
    #[inline]
    pub fn as_slice_memory_order(&self) -> Option<&[T]> {
        if self.layout.is_contiguous() {
            self.data.get(dense_block_range(&self.layout)?)
        } else {
            None
        }
    }

    /// Expose the underlying mutable slice if the elements form a dense block
    /// in some memory order (C or F), independent of offset. This is the
    /// `ndarray::as_slice_memory_order_mut` analogue Apollo's in-place FFT
    /// butterfly kernels require.
    #[inline]
    pub fn as_mut_slice_memory_order(&mut self) -> Option<&mut [T]> {
        if self.layout.is_contiguous() {
            self.data.get_mut(dense_block_range(&self.layout)?)
        } else {
            None
        }
    }

    /// Return an iterator yielding mutable subviews of rank `M` (where `M = N - 1`) along `axis`.
    #[inline]
    pub fn axis_iter_mut<const M: usize>(
        self,
        axis: usize,
    ) -> Result<crate::application::iter::AxisIterMut<'a, T, N, M>>
    where
        crate::domain::remove_axis::RankMarker<N>: crate::domain::remove_axis::RemoveAxis<
            N,
            SmallerShape = [usize; M],
            SmallerStrides = [isize; M],
        >,
    {
        crate::application::iter::AxisIterMut::new(
            self,
            axis,
            crate::domain::remove_axis::RankMarker::<N>,
        )
    }

    /// Return an iterator yielding mutable 1-D lane views *along* `axis`
    /// (ndarray `lanes_mut` parity; `M = N - 1` is the complement rank).
    ///
    /// # Errors
    /// [`LetoError`] if `axis >= N`, the layout does not fit its storage, or the
    /// layout aliases (a zero stride).
    #[inline]
    pub fn lanes_mut<const M: usize>(
        self,
        axis: usize,
    ) -> Result<crate::application::iter::LanesMut<'a, T, N, M>>
    where
        crate::domain::remove_axis::RankMarker<N>: crate::domain::remove_axis::RemoveAxis<
            N,
            SmallerShape = [usize; M],
            SmallerStrides = [isize; M],
        >,
    {
        crate::application::iter::LanesMut::new(
            self,
            axis,
            crate::domain::remove_axis::RankMarker::<N>,
        )
    }
}
