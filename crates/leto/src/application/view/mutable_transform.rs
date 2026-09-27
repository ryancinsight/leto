use super::immutable::ArrayView;
use super::mutable_core::ArrayViewMut;
use super::raw::{
    assert_exclusive_window, checked_block_range, raw_range_from_ptr, raw_range_from_ptr_mut,
    raw_slice_from_ptr_mut,
};
use crate::application::array::Array;
use crate::application::iter::{ElementIterMut, IndexedIterMut, TaskPartitionsMut};
use crate::domain::error::{LetoError, Result};
use crate::domain::slice::SliceArg;
use crate::infrastructure::storage::VecStorage;

impl<'a, T, const N: usize> ArrayViewMut<'a, T, N> {
    /// Iterator over `(multi-index, &mut element)` pairs in logical row-major
    /// order (leto `indexed_iter_mut` parity).
    ///
    /// # Errors
    /// Returns [`LetoError`] if the layout is out of bounds or cannot prove
    /// that each logical index addresses a distinct physical element.
    #[inline]
    pub fn indexed_iter_mut(self) -> Result<IndexedIterMut<'a, T, N>> {
        IndexedIterMut::new(self)
    }

    /// Iterator over mutable elements in logical row-major order.
    ///
    /// # Errors
    /// Returns [`LetoError`] if the layout is out of bounds or cannot prove
    /// that each logical index addresses a distinct physical element.
    #[inline]
    pub fn try_iter_mut(self) -> Result<ElementIterMut<'a, T, N>> {
        Ok(ElementIterMut::from_indexed(self.indexed_iter_mut()?))
    }

    /// Split this logical row-major domain into disjoint mutable task partitions.
    ///
    /// Partitions expose only range-limited element iterators and never expose
    /// the complete backing slice, which makes them suitable for a scheduler
    /// boundary without creating overlapping mutable access paths.
    ///
    /// # Errors
    /// Returns [`LetoError`] when `chunk_size` is zero, storage is invalid, or
    /// the layout is not provably injective.
    #[inline]
    pub fn task_partitions_mut(self, chunk_size: usize) -> Result<TaskPartitionsMut<'a, T, N>> {
        TaskPartitionsMut::new(self, chunk_size)
    }

    /// Consume the view and return the backing mutable slice with lifetime `'a`.
    ///
    /// # Panics
    ///
    /// Panics when the view was yielded by a mutable lane/axis iterator over an
    /// interleaved layout (see [`as_view`](Self::as_view)).
    #[inline]
    pub fn into_slice(self) -> &'a mut [T] {
        assert_exclusive_window(self.window_shared);
        // SAFETY: self.ptr is valid for self.len elements and lifetime 'a, and
        // the assertion above establishes the window is exclusively owned.
        unsafe { raw_slice_from_ptr_mut(self.ptr, self.len) }
    }

    /// Slice the mutable view, returning a sub-view.
    #[inline]
    pub fn slice_mut(self, ranges: &[(usize, usize, isize); N]) -> Result<ArrayViewMut<'a, T, N>> {
        let sliced_layout = self.layout.slice(ranges)?;
        Ok(self.with_layout(sliced_layout))
    }

    /// Slice the mutable view with leto-style arguments.
    #[inline]
    pub fn slice_with_mut<const M: usize>(
        self,
        args: &[SliceArg],
    ) -> Result<ArrayViewMut<'a, T, M>> {
        let sliced_layout = self.layout.slice_with(args)?;
        Ok(self.with_layout(sliced_layout))
    }

    /// Transpose the mutable view by permuting axes.
    #[inline]
    pub fn transpose_mut(self, axes: [usize; N]) -> Result<ArrayViewMut<'a, T, N>> {
        let transposed_layout = self.layout.transpose(axes)?;
        Ok(self.with_layout(transposed_layout))
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
                from: self.layout.shape().to_vec(),
                to: target_shape.to_vec(),
            });
        }
        Ok(self.with_layout(broadcasted_layout))
    }

    /// Reinterpret this mutable view with a new shape without copying.
    ///
    /// The current layout must be dense row-major and the new shape must have
    /// the same logical element count.
    #[inline]
    pub fn reshape_mut<const M: usize>(self, shape: [usize; M]) -> Result<ArrayViewMut<'a, T, M>> {
        let reshaped_layout = self.layout.reshape(shape)?;
        Ok(self.with_layout(reshaped_layout))
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
        if self.window_shared {
            // A whole-window borrow would alias sibling views' elements, so
            // clone element-by-element through checked per-element access.
            let size = self.size();
            let shape = self.shape();
            let mut values: Vec<T> = Vec::with_capacity(size);
            let mut index = [0usize; N];
            for _ in 0..size {
                values.push(
                    self.get(index)
                        .expect("invariant: logical index is in bounds")
                        .clone(),
                );
                for d in (0..N).rev() {
                    index[d] += 1;
                    if index[d] < shape[d] {
                        break;
                    }
                    index[d] = 0;
                }
            }
            return Array::<T, VecStorage<T>, N>::from_shape_vec(shape, values)
                .expect("logical row-major materialization has matching shape and storage");
        }
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

    /// Returns true when the view's strides are canonically C (row-major),
    /// independent of the base offset (offset-independent half of
    /// [`is_c_contiguous`](Self::is_c_contiguous)).
    #[inline]
    pub fn is_c_dense(&self) -> bool {
        self.layout.is_c_dense()
    }

    /// Returns true when the view's strides are canonically Fortran
    /// (column-major), independent of the base offset (offset-independent half
    /// of [`is_f_contiguous`](Self::is_f_contiguous)).
    #[inline]
    pub fn is_f_dense(&self) -> bool {
        self.layout.is_f_dense()
    }

    /// Expose the underlying slice if the elements form a dense row-major
    /// (C-order) block, independent of offset.
    #[inline]
    pub fn as_slice(&self) -> Option<&[T]> {
        if self.layout.is_c_dense() {
            let range = checked_block_range(&self.layout, self.len)?;
            // SAFETY: `ptr` is valid for `len` elements and `range` is in
            // bounds; a C-dense layout's block is exactly the view's own
            // elements, so this slice never covers a sibling view's elements
            // even when the window is shared.
            unsafe { Some(raw_range_from_ptr(self.ptr, range)) }
        } else {
            None
        }
    }

    /// Expose the underlying mutable slice if the elements form a dense
    /// row-major (C-order) block, independent of offset.
    #[inline]
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        if self.layout.is_c_dense() {
            let range = checked_block_range(&self.layout, self.len)?;
            // SAFETY: `ptr` is valid for `len` elements and `range` is in
            // bounds; a C-dense block is exactly the view's own elements,
            // which the view exclusively owns even when yielded by a mutable
            // iterator (sibling views' elements are disjoint).
            unsafe { Some(raw_range_from_ptr_mut(self.ptr, range)) }
        } else {
            None
        }
    }

    /// Expose the underlying slice if the elements form a dense block in some
    /// memory order (C or F), independent of offset. Physical memory order.
    #[inline]
    pub fn as_slice_memory_order(&self) -> Option<&[T]> {
        if self.layout.is_contiguous() {
            let range = checked_block_range(&self.layout, self.len)?;
            // SAFETY: `ptr` is valid for `len` elements and `range` is in
            // bounds; a contiguous layout's dense block is exactly the view's
            // own elements (no sibling overlap).
            unsafe { Some(raw_range_from_ptr(self.ptr, range)) }
        } else {
            None
        }
    }

    /// Expose the underlying mutable slice if the elements form a dense block
    /// in some memory order (C or F), independent of offset. This is the
    /// `leto::as_slice_memory_order_mut` analogue Apollo's in-place FFT
    /// butterfly kernels require.
    #[inline]
    pub fn as_mut_slice_memory_order(&mut self) -> Option<&mut [T]> {
        if self.layout.is_contiguous() {
            let range = checked_block_range(&self.layout, self.len)?;
            // SAFETY: `ptr` is valid for `len` elements and `range` is in
            // bounds; a contiguous layout's dense block is exactly the view's
            // own elements, exclusively owned even for iterator-yielded
            // sub-views (siblings are disjoint).
            unsafe { Some(raw_range_from_ptr_mut(self.ptr, range)) }
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
    /// (leto `lanes_mut` parity; `M = N - 1` is the complement rank).
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
