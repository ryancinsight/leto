#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use crate::application::array::Array;
use crate::application::iter::{
    AxisChunks, ElementIter, ElementIterMut, ExactChunks, IndexedIter, IndexedIterMut,
    TaskPartitionsMut, Windows,
};
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::domain::slice::SliceArg;
use crate::infrastructure::storage::{SliceStorage, VecStorage};

/// Computes the physical `[offset, offset + size)` range covered by a layout
/// whose elements form a single dense block. Returns `None` only on size
/// overflow. Shared by the contiguous-slice accessors of both view types.
#[inline]
fn dense_block_range<const N: usize>(layout: &Layout<N>) -> Option<core::ops::Range<usize>> {
    let start = layout.offset();
    let end = start.checked_add(layout.checked_size().ok()?)?;
    Some(start..end)
}

/// A read-only zero-copy view of an N-dimensional strided array.
#[derive(Clone, Copy)]
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
        self.layout.shape()
    }

    /// Returns the strides of the view.
    #[inline]
    pub const fn strides(&self) -> [isize; N] {
        self.layout.strides()
    }

    /// Returns the offset of the view.
    #[inline]
    pub const fn offset(&self) -> usize {
        self.layout.offset()
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
    /// (leto `iter` parity). The iterator borrows the view's data for `'a`,
    /// so it outlives a temporary view produced by `array.view().iter()`.
    #[inline]
    pub fn iter(&self) -> ElementIter<'a, T, N> {
        ElementIter::new(self)
    }

    /// Iterator over `(multi-index, &element)` pairs in logical row-major order
    /// (leto `indexed_iter` parity).
    #[inline]
    pub fn indexed_iter(&self) -> IndexedIter<'a, T, N> {
        IndexedIter::new(self)
    }

    /// Zero-copy iterator over non-overlapping chunks of `chunk_shape`
    /// (leto `exact_chunks` parity). Each yielded view shares this view's
    /// strides and backing storage.
    ///
    /// # Errors
    /// [`LetoError`] if any `chunk_shape[i]` is `0` or the chunk grid overflows
    /// `usize`.
    #[inline]
    pub fn exact_chunks(&self, chunk_shape: [usize; N]) -> Result<ExactChunks<'a, T, N>> {
        ExactChunks::new(self, chunk_shape)
    }

    /// Zero-copy iterator over chunks along `axis` (leto
    /// `axis_chunks_iter` parity). The final yielded view carries the
    /// remainder when present.
    ///
    /// # Errors
    /// [`LetoError`] if `axis >= N` or `chunk_len == 0`.
    #[inline]
    pub fn axis_chunks_iter(&self, axis: usize, chunk_len: usize) -> Result<AxisChunks<'a, T, N>> {
        AxisChunks::new(self, axis, chunk_len)
    }

    /// Zero-copy iterator over every sliding window of shape `window_shape`
    /// (leto `windows` parity). Each yielded view shares this view's strides
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

    /// Slice the view with leto-style arguments.
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
                let mut values: Vec<T> = Vec::with_capacity(size);
                values.extend(self.iter().cloned());
                values
            }
        };
        Array::<T, VecStorage<T>, N>::from_shape_vec(self.shape(), data)
            .expect("logical row-major materialization has matching shape and storage")
    }

    /// Wrap this view as a **zero-copy** borrowed [`Array`] over [`SliceStorage`],
    /// sharing the view's layout (offset + strides) and backing slice with no
    /// allocation or copy.
    ///
    /// Because the borrowed array carries the same layout, it indexes
    /// identically to the view for both contiguous and strided/offset views, so
    /// it can feed any storage-generic `Array<T, S, N>` consumer without the
    /// [`to_contiguous`](Self::to_contiguous) materialization. Prefer this over
    /// `to_contiguous` when the consumer only reads the input.
    #[inline]
    #[must_use]
    pub fn as_array(&self) -> Array<T, SliceStorage<'a, T>, N> {
        // The (layout, data) pair is exactly the one this view already indexes
        // through, so the borrowed array's `get` (layout.offset_of into
        // storage.as_slice) reproduces the view's element access bit-for-bit.
        Array::new(self.layout, SliceStorage::new(self.data))
            .expect("view layout is valid for its backing slice")
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
    pub fn as_slice(&self) -> Option<&'a [T]> {
        if self.layout.is_c_dense() {
            self.data.get(dense_block_range(&self.layout)?)
        } else {
            None
        }
    }

    /// Expose the underlying slice if the elements form a dense block in some
    /// memory order (C or F), independent of offset. The returned slice is in
    /// physical memory order, matching `leto::as_slice_memory_order`.
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
    /// (leto `lanes` parity; `M = N - 1` is the complement rank). Dual of
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
    pub(crate) ptr: std::ptr::NonNull<T>,
    pub(crate) len: usize,
    /// Whether the physical window `[ptr, ptr + len)` may contain elements
    /// owned by sibling views (mutable lane/axis iteration over an interleaved
    /// layout). A shared window forbids materializing the window as a slice —
    /// [`data`](Self::data), [`data_mut`](Self::data_mut),
    /// [`into_slice`](Self::into_slice), and [`as_view`](Self::as_view) —
    /// because a sibling's element references would alias it; per-element
    /// access ([`get`](Self::get), [`get_mut`](Self::get_mut), indexing) stays
    /// available since the layouts of sibling views address disjoint elements.
    pub(crate) window_shared: bool,
    pub(crate) _marker: std::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T, const N: usize> ArrayViewMut<'a, T, N> {
    /// Create a new ArrayViewMut from a layout and mutable slice.
    #[inline]
    pub fn new(layout: Layout<N>, data: &'a mut [T]) -> Self {
        Self {
            layout,
            // SAFETY: a slice's data pointer is never null, including for
            // empty slices (it is dangling-but-aligned, still non-null).
            ptr: unsafe { std::ptr::NonNull::new_unchecked(data.as_mut_ptr()) },
            len: data.len(),
            // The whole window comes from one exclusive `&mut [T]`, so no
            // sibling view can own any part of it.
            window_shared: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a bounds-checked ArrayViewMut from a layout and mutable slice.
    #[inline]
    pub fn try_new(layout: Layout<N>, data: &'a mut [T]) -> Result<Self> {
        layout.validate_storage_len(data.len())?;
        Ok(Self::new(layout, data))
    }

    /// Reborrow the mutable view with a shorter lifetime.
    #[inline]
    pub fn reborrow(&mut self) -> ArrayViewMut<'_, T, N> {
        ArrayViewMut {
            layout: self.layout,
            ptr: self.ptr,
            len: self.len,
            window_shared: self.window_shared,
            _marker: std::marker::PhantomData,
        }
    }

    /// Borrow this mutable view as an immutable [`ArrayView`] (leto `.view()`
    /// parity), sharing the same layout and backing memory.
    ///
    /// # Panics
    ///
    /// Panics when the view was yielded by a mutable lane/axis iterator over an
    /// interleaved layout: its physical window contains sibling views' elements,
    /// so materializing it as a shared slice would alias their `&mut` element
    /// references. Use [`get`](Self::get) or indexing for element reads there.
    #[inline]
    pub fn as_view(&self) -> ArrayView<'_, T, N> {
        assert!(
            !self.window_shared,
            "window is shared with sibling lane/axis views; a whole-window \
             slice would alias their elements (use per-element access instead)"
        );
        // SAFETY: `ptr` is valid for `len` elements for the duration of the
        // borrow of `self`, and the assertion above establishes the window is
        // exclusively owned, so no sibling view can mint `&mut` into it.
        let data = unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) };
        ArrayView::new(self.layout, data)
    }

    /// Returns the shape of the view.
    #[inline]
    pub const fn shape(&self) -> [usize; N] {
        self.layout.shape()
    }

    /// Returns the strides of the view.
    #[inline]
    pub const fn strides(&self) -> [isize; N] {
        self.layout.strides()
    }

    /// Returns the offset of the view.
    #[inline]
    pub const fn offset(&self) -> usize {
        self.layout.offset()
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

    /// Returns true when this view exclusively owns its physical window, so
    /// whole-window accessors ([`data`](Self::data), [`data_mut`](Self::data_mut),
    /// [`into_slice`](Self::into_slice), [`as_view`](Self::as_view)) are
    /// available. Views constructed from a slice always own their window;
    /// views yielded by mutable lane/axis iterators own it only when the
    /// yielded window is dense (span equals logical size), because an
    /// interleaved window still contains sibling views' elements.
    #[inline]
    pub const fn has_exclusive_window(&self) -> bool {
        !self.window_shared
    }

    /// Returns the raw data slice as read-only.
    ///
    /// # Panics
    ///
    /// Panics when the view was yielded by a mutable lane/axis iterator over an
    /// interleaved layout (see [`as_view`](Self::as_view)).
    #[inline]
    pub fn data(&self) -> &[T] {
        assert!(
            !self.window_shared,
            "window is shared with sibling lane/axis views; a whole-window \
             slice would alias their elements (use per-element access instead)"
        );
        // SAFETY: self.ptr is valid for self.len elements, and the assertion
        // above establishes the window is exclusively owned.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns the raw mutable data slice.
    ///
    /// # Panics
    ///
    /// Panics when the view was yielded by a mutable lane/axis iterator over an
    /// interleaved layout (see [`as_view`](Self::as_view)).
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        assert!(
            !self.window_shared,
            "window is shared with sibling lane/axis views; a whole-window \
             slice would alias their elements (use per-element access instead)"
        );
        // SAFETY: self.ptr is valid for self.len elements, and the assertion
        // above establishes the window is exclusively owned.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

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
        assert!(
            !self.window_shared,
            "window is shared with sibling lane/axis views; a whole-window \
             slice would alias their elements (use per-element access instead)"
        );
        // SAFETY: self.ptr is valid for self.len elements and lifetime 'a, and
        // the assertion above establishes the window is exclusively owned.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Get a reference to the element at the specified index.
    #[inline]
    pub fn get(&self, index: [usize; N]) -> Result<&T> {
        let offset = self.layout.offset_of(index)?;
        if offset >= self.len {
            return Err(LetoError::StorageError {
                reason: format!(
                    "physical offset {offset} exceeds backing slice length {}",
                    self.len
                ),
            });
        }
        // SAFETY: self.ptr is valid for self.len elements.
        unsafe { Ok(&*self.ptr.as_ptr().add(offset)) }
    }

    /// Get a mutable reference to the element at the specified index.
    #[inline]
    pub fn get_mut(&mut self, index: [usize; N]) -> Result<&mut T> {
        let offset = self.layout.offset_of(index)?;
        if offset >= self.len {
            return Err(LetoError::StorageError {
                reason: format!(
                    "physical offset {offset} exceeds backing slice length {}",
                    self.len
                ),
            });
        }
        // SAFETY: self.ptr is valid for self.len elements.
        unsafe { Ok(&mut *self.ptr.as_ptr().add(offset)) }
    }

    /// Set every element of the view to a clone of `value` (leto `fill`
    /// parity). Contiguous views fill their dense block directly; strided
    /// views walk logical row-major order, so it is correct for any strides.
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        // Dense block in either memory order: one slice fill instead of a
        // per-element odometer with checked offset arithmetic. The range is
        // exactly the view's own elements, so this stays correct for
        // iterator-yielded sub-views.
        if let Some(slice) = self.as_mut_slice_memory_order() {
            slice.fill(value);
            return;
        }
        let shape = self.shape();
        let size = self.size();
        if size == 0 {
            return;
        }
        let mut index = [0usize; N];
        for _ in 0..size {
            *self
                .get_mut(index)
                .expect("invariant: logical index is in bounds") = value.clone();
            // row-major odometer increment of the multi-index.
            for d in (0..N).rev() {
                index[d] += 1;
                if index[d] < shape[d] {
                    break;
                }
                index[d] = 0;
            }
        }
    }

    /// Slice the mutable view, returning a sub-view.
    #[inline]
    pub fn slice_mut(self, ranges: &[(usize, usize, isize); N]) -> Result<ArrayViewMut<'a, T, N>> {
        let sliced_layout = self.layout.slice(ranges)?;
        Ok(ArrayViewMut {
            layout: sliced_layout,
            ptr: self.ptr,
            len: self.len,
            window_shared: self.window_shared,
            _marker: std::marker::PhantomData,
        })
    }

    /// Slice the mutable view with leto-style arguments.
    #[inline]
    pub fn slice_with_mut<const M: usize>(
        self,
        args: &[SliceArg],
    ) -> Result<ArrayViewMut<'a, T, M>> {
        let sliced_layout = self.layout.slice_with(args)?;
        Ok(ArrayViewMut {
            layout: sliced_layout,
            ptr: self.ptr,
            len: self.len,
            window_shared: self.window_shared,
            _marker: std::marker::PhantomData,
        })
    }

    /// Transpose the mutable view by permuting axes.
    #[inline]
    pub fn transpose_mut(self, axes: [usize; N]) -> Result<ArrayViewMut<'a, T, N>> {
        let transposed_layout = self.layout.transpose(axes)?;
        Ok(ArrayViewMut {
            layout: transposed_layout,
            ptr: self.ptr,
            len: self.len,
            window_shared: self.window_shared,
            _marker: std::marker::PhantomData,
        })
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
        Ok(ArrayViewMut {
            layout: broadcasted_layout,
            ptr: self.ptr,
            len: self.len,
            window_shared: self.window_shared,
            _marker: std::marker::PhantomData,
        })
    }

    /// Reinterpret this mutable view with a new shape without copying.
    ///
    /// The current layout must be dense row-major and the new shape must have
    /// the same logical element count.
    #[inline]
    pub fn reshape_mut<const M: usize>(self, shape: [usize; M]) -> Result<ArrayViewMut<'a, T, M>> {
        let reshaped_layout = self.layout.reshape(shape)?;
        Ok(ArrayViewMut {
            layout: reshaped_layout,
            ptr: self.ptr,
            len: self.len,
            window_shared: self.window_shared,
            _marker: std::marker::PhantomData,
        })
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
            let range = dense_block_range(&self.layout)?;
            if range.end <= self.len {
                // SAFETY: `ptr` is valid for `len` elements and
                // `range.end <= len`, so the sub-range is in bounds; a C-dense
                // layout's block is exactly the view's own elements, so this
                // slice never covers a sibling view's elements even when the
                // window is shared.
                unsafe {
                    Some(std::slice::from_raw_parts(
                        self.ptr.as_ptr().add(range.start),
                        range.len(),
                    ))
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Expose the underlying mutable slice if the elements form a dense
    /// row-major (C-order) block, independent of offset.
    #[inline]
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        if self.layout.is_c_dense() {
            let range = dense_block_range(&self.layout)?;
            if range.end <= self.len {
                // SAFETY: `ptr` is valid for `len` elements and
                // `range.end <= len`; a C-dense block is exactly the view's own
                // elements, which the view exclusively owns even when yielded
                // by a mutable iterator (sibling views' elements are disjoint).
                unsafe {
                    Some(std::slice::from_raw_parts_mut(
                        self.ptr.as_ptr().add(range.start),
                        range.len(),
                    ))
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Expose the underlying slice if the elements form a dense block in some
    /// memory order (C or F), independent of offset. Physical memory order.
    #[inline]
    pub fn as_slice_memory_order(&self) -> Option<&[T]> {
        if self.layout.is_contiguous() {
            let range = dense_block_range(&self.layout)?;
            if range.end <= self.len {
                // SAFETY: `ptr` is valid for `len` elements and
                // `range.end <= len`; a contiguous layout's dense block is
                // exactly the view's own elements (no sibling overlap).
                unsafe {
                    Some(std::slice::from_raw_parts(
                        self.ptr.as_ptr().add(range.start),
                        range.len(),
                    ))
                }
            } else {
                None
            }
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
            let range = dense_block_range(&self.layout)?;
            if range.end <= self.len {
                // SAFETY: `ptr` is valid for `len` elements and
                // `range.end <= len`; a contiguous layout's dense block is
                // exactly the view's own elements, exclusively owned even for
                // iterator-yielded sub-views (siblings are disjoint).
                unsafe {
                    Some(std::slice::from_raw_parts_mut(
                        self.ptr.as_ptr().add(range.start),
                        range.len(),
                    ))
                }
            } else {
                None
            }
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

// ── Index / IndexMut for ArrayView and ArrayViewMut ──────────────────────────

/// Enable `view[[i, j, k]]` syntax, matching `leto::ArrayView` ergonomics.
impl<'a, T, const N: usize> std::ops::Index<[usize; N]> for ArrayView<'a, T, N> {
    type Output = T;
    #[inline]
    fn index(&self, index: [usize; N]) -> &T {
        let offset = self
            .layout
            .offset_of(index)
            .expect("ArrayView index out of bounds");
        &self.data[offset]
    }
}

/// Enable `view[[i, j, k]]` syntax on mutable views.
impl<'a, T, const N: usize> std::ops::Index<[usize; N]> for ArrayViewMut<'a, T, N> {
    type Output = T;
    #[inline]
    fn index(&self, index: [usize; N]) -> &T {
        let offset = self
            .layout
            .offset_of(index)
            .expect("ArrayViewMut index out of bounds");
        assert!(
            offset < self.len,
            "ArrayViewMut index physical offset {offset} exceeds backing length {}",
            self.len
        );
        // SAFETY: ptr is valid for len elements for the lifetime of the view,
        // and the assertion above establishes `offset < len`.
        unsafe { &*self.ptr.as_ptr().add(offset) }
    }
}

/// Enable `view[[i, j, k]] = value` syntax on mutable views.
impl<'a, T, const N: usize> std::ops::IndexMut<[usize; N]> for ArrayViewMut<'a, T, N> {
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut T {
        let offset = self
            .layout
            .offset_of(index)
            .expect("ArrayViewMut index_mut out of bounds");
        assert!(
            offset < self.len,
            "ArrayViewMut index_mut physical offset {offset} exceeds backing length {}",
            self.len
        );
        // SAFETY: ptr is valid for len elements for the lifetime of the view,
        // and the assertion above establishes `offset < len`.
        unsafe { &mut *self.ptr.as_ptr().add(offset) }
    }
}

/// Enable `view[i]` (usize) syntax for 1-D views (leto `ArrayView1` parity).
impl<'a, T> std::ops::Index<usize> for ArrayView<'a, T, 1> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &T {
        let offset = self
            .layout
            .offset_of([index])
            .expect("ArrayView1 index out of bounds");
        &self.data[offset]
    }
}

/// Enable `view[i]` (usize) mutable syntax for 1-D views.
impl<'a, T> std::ops::Index<usize> for ArrayViewMut<'a, T, 1> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &T {
        let offset = self
            .layout
            .offset_of([index])
            .expect("ArrayViewMut1 index out of bounds");
        assert!(
            offset < self.len,
            "ArrayViewMut1 index physical offset {offset} exceeds backing length {}",
            self.len
        );
        // SAFETY: ptr is valid for len elements, and the assertion above
        // establishes `offset < len`.
        unsafe { &*self.ptr.as_ptr().add(offset) }
    }
}

/// Enable `view[i] = value` syntax for 1-D mutable views.
impl<'a, T> std::ops::IndexMut<usize> for ArrayViewMut<'a, T, 1> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        let offset = self
            .layout
            .offset_of([index])
            .expect("ArrayViewMut1 index_mut out of bounds");
        assert!(
            offset < self.len,
            "ArrayViewMut1 index_mut physical offset {offset} exceeds backing length {}",
            self.len
        );
        // SAFETY: ptr is valid for len elements, and the assertion above
        // establishes `offset < len`.
        unsafe { &mut *self.ptr.as_ptr().add(offset) }
    }
}

#[cfg(test)]
mod as_array_tests {
    use crate::application::array::Array;
    use crate::infrastructure::storage::VecStorage;

    #[test]
    fn as_array_matches_view_indexing_including_strided() {
        // 3x4 source; take a strided sub-view (every other column) and confirm
        // the zero-copy borrowed array indexes identically to the view.
        let src = Array::<f64, VecStorage<f64>, 2>::from_shape_vec(
            [3, 4],
            (0..12).map(|i| i as f64).collect(),
        )
        .unwrap();
        let view = src.view();
        let borrowed = view.as_array();
        assert_eq!(borrowed.shape(), view.shape());
        for r in 0..3 {
            for c in 0..4 {
                assert_eq!(*borrowed.get([r, c]).unwrap(), *view.get([r, c]).unwrap());
                assert_eq!(borrowed[[r, c]], src[[r, c]]);
            }
        }

        // Strided sub-view: columns [1,3) step is exercised via slice.
        let strided = src.view().slice(&[(0, 3, 1), (1, 4, 2)]).unwrap();
        let strided_borrowed = strided.as_array();
        assert_eq!(strided_borrowed.shape(), strided.shape());
        for r in 0..strided.shape()[0] {
            for c in 0..strided.shape()[1] {
                assert_eq!(
                    *strided_borrowed.get([r, c]).unwrap(),
                    *strided.get([r, c]).unwrap(),
                    "strided borrowed array must match the view at [{r},{c}]"
                );
            }
        }
    }
}
