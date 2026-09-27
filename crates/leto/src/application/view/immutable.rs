use super::raw::dense_block_range;
use crate::application::array::Array;
use crate::application::iter::{AxisChunks, ElementIter, ExactChunks, IndexedIter, Windows};
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::domain::slice::SliceArg;
use crate::infrastructure::storage::{SliceStorage, VecStorage};

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
    /// Dense row-major views clone the exposed slice; rank-2 Fortran-dense
    /// views run the tiled transpose copy. Other strided, transposed, or
    /// broadcasted views are copied in logical row-major order.
    pub fn to_contiguous(&self) -> Array<T, VecStorage<T>, N>
    where
        T: Clone,
    {
        let data = match self.as_slice() {
            Some(slice) => slice.to_vec(),
            None => {
                // Rank-2 F-dense views transpose through the cache-blocked
                // kernel instead of the per-element odometer (an order of
                // magnitude fewer offset computations on large planes).
                if let (&[rows, cols], Some(source)) = (
                    self.shape().as_slice(),
                    self.as_slice_memory_order().filter(|_| self.is_f_dense()),
                ) {
                    let mut values = source.to_vec();
                    crate::application::assign::transpose_copy(source, &mut values, cols, rows)
                        .expect("invariant: dense view fits its transpose storage");
                    values
                } else {
                    let size = self.layout.size();
                    let mut values: Vec<T> = Vec::with_capacity(size);
                    values.extend(self.iter().cloned());
                    values
                }
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
