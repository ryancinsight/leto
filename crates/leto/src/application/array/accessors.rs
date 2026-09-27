use super::types::Array;
use crate::application::iter::{AxisChunks, ElementIter, ExactChunks, IndexedIter, Lanes, Windows};
use crate::application::view::ArrayView;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::domain::slice::SliceArg;
use crate::infrastructure::storage::Storage;

impl<T, S, const N: usize> Array<T, S, N>
where
    S: Storage<T>,
{
    /// Returns the shape of the array.
    #[inline]
    pub const fn shape(&self) -> [usize; N] {
        self.layout.shape()
    }

    /// Returns the strides of the array.
    #[inline]
    pub const fn strides(&self) -> [isize; N] {
        self.layout.strides()
    }

    /// Returns the starting offset of the array.
    #[inline]
    pub const fn offset(&self) -> usize {
        self.layout.offset()
    }

    /// Returns the total logical size of the array.
    #[inline]
    pub fn size(&self) -> usize {
        self.layout.size()
    }

    /// Returns the total logical element count.
    #[inline]
    pub fn len(&self) -> usize {
        self.layout.size()
    }

    /// Returns `true` when any axis has zero length.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layout.size() == 0
    }

    /// Returns a pointer to the first logical element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.storage.as_slice()[self.layout.offset()..].as_ptr()
    }

    /// Returns the layout of the array.
    #[inline]
    pub const fn layout(&self) -> Layout<N> {
        self.layout
    }

    /// Returns a reference to the underlying storage backing.
    #[inline]
    pub const fn storage(&self) -> &S {
        &self.storage
    }

    /// Returns a mutable reference to the underlying storage backing.
    #[inline]
    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    /// Consume the array and return its underlying storage.
    #[inline]
    pub fn into_storage(self) -> S {
        self.storage
    }

    /// Returns an immutable read-only view of this array.
    #[inline]
    pub fn view(&self) -> ArrayView<'_, T, N> {
        ArrayView::new(self.layout, self.storage.as_slice())
    }

    /// The elements as one contiguous slice in logical row-major order, or
    /// `None` if the array is not C-contiguous (leto `as_slice` parity).
    #[inline]
    pub fn as_slice(&self) -> Option<&[T]> {
        self.view().as_slice()
    }

    /// Expose the dense physical-memory slice when this array is contiguous.
    #[inline]
    pub fn as_slice_memory_order(&self) -> Option<&[T]> {
        self.view().as_slice_memory_order()
    }

    /// Iterator over the array's elements in logical row-major order
    /// (leto `iter` parity), respecting arbitrary strides.
    #[inline]
    pub fn iter(&self) -> ElementIter<'_, T, N> {
        self.view().iter()
    }

    /// Iterator over `(multi-index, &element)` pairs in logical row-major order
    /// (leto `indexed_iter` parity).
    #[inline]
    pub fn indexed_iter(&self) -> IndexedIter<'_, T, N> {
        self.view().indexed_iter()
    }

    /// Zero-copy iterator over non-overlapping chunks of `chunk_shape`
    /// (leto `exact_chunks` parity).
    ///
    /// Remainders along any axis are skipped. Each yielded view has shape
    /// `chunk_shape` and shares this array's backing storage.
    ///
    /// # Errors
    /// [`LetoError`] if any `chunk_shape[i]` is `0` or the chunk grid overflows
    /// `usize`.
    #[inline]
    pub fn exact_chunks(&self, chunk_shape: [usize; N]) -> Result<ExactChunks<'_, T, N>> {
        self.view().exact_chunks(chunk_shape)
    }

    /// Zero-copy iterator over chunks along `axis` (leto
    /// `axis_chunks_iter` parity).
    ///
    /// The final yielded view carries the remainder when `shape[axis]` is not
    /// divisible by `chunk_len`.
    ///
    /// # Errors
    /// [`LetoError`] if `axis >= N` or `chunk_len == 0`.
    #[inline]
    pub fn axis_chunks_iter(&self, axis: usize, chunk_len: usize) -> Result<AxisChunks<'_, T, N>> {
        self.view().axis_chunks_iter(axis, chunk_len)
    }

    /// Zero-copy iterator over every sliding window of shape `window_shape`
    /// (leto `windows` parity).
    ///
    /// # Errors
    /// [`LetoError`] if any `window_shape[i]` is `0` or exceeds
    /// `shape[i]`.
    #[inline]
    pub fn windows(&self, window_shape: [usize; N]) -> Result<Windows<'_, T, N>> {
        self.view().windows(window_shape)
    }

    /// Zero-copy iterator over the read-only 1-D lanes along `axis`
    /// (leto `lanes` parity; `M = N - 1`).
    ///
    /// # Errors
    /// [`LetoError`] if `axis >= N` or the layout does not fit
    /// its storage.
    #[inline]
    pub fn lanes<const M: usize>(&self, axis: usize) -> Result<Lanes<'_, T, N, M>>
    where
        crate::domain::remove_axis::RankMarker<N>: crate::domain::remove_axis::RemoveAxis<
            N,
            SmallerShape = [usize; M],
            SmallerStrides = [isize; M],
        >,
    {
        self.view().lanes(axis)
    }

    /// Slice the array, returning a read-only view.
    #[inline]
    pub fn slice(&self, ranges: &[(usize, usize, isize); N]) -> Result<ArrayView<'_, T, N>> {
        self.view().slice(ranges)
    }

    /// Slice the array with leto-style arguments, returning a read-only view.
    #[inline]
    pub fn slice_with<const M: usize>(&self, args: &[SliceArg]) -> Result<ArrayView<'_, T, M>> {
        self.view().slice_with(args)
    }

    /// Fix one axis at `index`, reducing the rank by 1 (leto `index_axis` parity).
    ///
    /// `M` must equal `N - 1`; a mismatch returns `LetoError` from `slice_with`.
    /// The caller expresses the output rank explicitly, for example:
    ///
    /// ```
    /// # use leto::{Array4, VecStorage};
    /// # let a = Array4::<f64>::zeros([2, 3, 4, 5]);
    /// let view3 = a.index_axis::<3>(0, 1).unwrap(); // fix axis 0 at index 1
    /// assert_eq!(view3.shape(), [3, 4, 5]);
    /// ```
    #[inline]
    pub fn index_axis<const M: usize>(
        &self,
        axis: usize,
        index: usize,
    ) -> Result<ArrayView<'_, T, M>> {
        let args: Vec<SliceArg> = (0..N)
            .map(|i| {
                if i == axis {
                    SliceArg::Index(index as isize)
                } else {
                    SliceArg::All
                }
            })
            .collect();
        self.slice_with::<M>(&args)
    }

    /// Transpose the array, returning a read-only view.
    #[inline]
    pub fn transpose(&self, axes: [usize; N]) -> Result<ArrayView<'_, T, N>> {
        self.view().transpose(axes)
    }

    /// Broadcast the array, returning a read-only view.
    #[inline]
    pub fn broadcast<const M: usize>(
        &self,
        target_shape: [usize; M],
    ) -> Result<ArrayView<'_, T, M>> {
        self.view().broadcast(target_shape)
    }

    /// Reinterpret this array with a new shape without copying.
    ///
    /// The current layout must be dense row-major and the new shape must have
    /// the same logical element count.
    #[inline]
    pub fn reshape<const M: usize>(&self, shape: [usize; M]) -> Result<ArrayView<'_, T, M>> {
        self.view().reshape(shape)
    }

    /// Consume this array and reinterpret its storage with a new shape without copying.
    ///
    /// The current layout must be dense row-major and the new shape must have
    /// the same logical element count.
    #[inline]
    pub fn into_shape<const M: usize>(self, shape: [usize; M]) -> Result<Array<T, S, M>> {
        let reshaped_layout = self.layout.reshape(shape)?;
        Array::new(reshaped_layout, self.storage)
    }

    /// Named alias for [`transpose`](Self::transpose).
    #[inline]
    pub fn permute(&self, axes: [usize; N]) -> Result<ArrayView<'_, T, N>> {
        self.view().permute(axes)
    }

    /// Materialize this array into C-contiguous row-major storage.
    ///
    /// Dense row-major arrays clone the exposed slice. Strided, transposed, or
    /// broadcasted arrays are copied in logical row-major order.
    pub fn to_contiguous(&self) -> Array<T, crate::infrastructure::storage::VecStorage<T>, N>
    where
        T: Clone,
    {
        self.view().to_contiguous()
    }

    /// Get a reference to the element at the specified index.
    #[inline]
    pub fn get(&self, index: [usize; N]) -> Result<&T> {
        let offset = self.layout.offset_of(index)?;
        let slice = self.storage.as_slice();
        if offset >= slice.len() {
            return Err(LetoError::StorageError {
                reason: format!(
                    "physical offset {offset} exceeds backing slice length {}",
                    slice.len()
                ),
            });
        }
        Ok(&slice[offset])
    }

    /// Map every logical element into a newly allocated array with the same shape.
    #[inline]
    pub fn mapv<U, F>(&self, mut f: F) -> Array<U, crate::infrastructure::storage::VecStorage<U>, N>
    where
        F: FnMut(T) -> U,
        T: Copy,
    {
        let values = self.iter().map(|&value| f(value)).collect();
        Array::<U, crate::infrastructure::storage::VecStorage<U>, N>::from_shape_vec(
            self.shape(),
            values,
        )
        .expect("invariant: logical map preserves element count")
    }

    /// Zip two arrays elementwise into a newly allocated array with this shape.
    ///
    /// # Panics
    /// Panics when the shapes differ.
    #[inline]
    pub fn zip_map<S2, U, F>(
        &self,
        rhs: &Array<T, S2, N>,
        mut f: F,
    ) -> Array<U, crate::infrastructure::storage::VecStorage<U>, N>
    where
        F: FnMut(T, T) -> U,
        T: Copy,
        S2: Storage<T>,
    {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "invariant: zip_map requires identical shapes"
        );
        let values = self
            .iter()
            .zip(rhs.iter())
            .map(|(&left, &right)| f(left, right))
            .collect();
        Array::<U, crate::infrastructure::storage::VecStorage<U>, N>::from_shape_vec(
            self.shape(),
            values,
        )
        .expect("invariant: zip_map preserves element count")
    }
}
