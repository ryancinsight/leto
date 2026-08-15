#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use crate::application::iter::{
    AxisChunks, ElementIter, ElementIterMut, ExactChunks, IndexedIter, IndexedIterMut, Lanes,
    LanesMut, TaskPartitionsMut, Windows,
};
use crate::application::view::{ArrayView, ArrayViewMut};
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::domain::slice::SliceArg;
use crate::infrastructure::storage::{Storage, StorageMut};
use serde::de::Error as DeserializeError;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::marker::PhantomData;

/// Read-only source for assigning values into an array.
pub trait AssignSource<T, const N: usize> {
    /// Shape of the assignment source.
    fn assign_shape(&self) -> [usize; N];

    /// Value at a logical index.
    ///
    /// # Errors
    /// Returns [`LetoError`] when the index is out of bounds for the source.
    fn assign_get(&self, index: [usize; N]) -> Result<&T>;
}

impl<T, S, const N: usize> AssignSource<T, N> for Array<T, S, N>
where
    S: Storage<T>,
{
    #[inline]
    fn assign_shape(&self) -> [usize; N] {
        self.shape()
    }

    #[inline]
    fn assign_get(&self, index: [usize; N]) -> Result<&T> {
        self.get(index)
    }
}

impl<T, const N: usize> AssignSource<T, N> for ArrayView<'_, T, N> {
    #[inline]
    fn assign_shape(&self) -> [usize; N] {
        self.shape()
    }

    #[inline]
    fn assign_get(&self, index: [usize; N]) -> Result<&T> {
        self.get(index)
    }
}

/// An N-dimensional strided array.
#[derive(Debug, Clone)]
pub struct Array<T, S, const N: usize> {
    pub(crate) layout: Layout<N>,
    pub(crate) storage: S,
    pub(crate) _marker: PhantomData<T>,
}

impl<T, S, const N: usize> Serialize for Array<T, S, N>
where
    S: Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> core::result::Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        let mut state = serializer.serialize_struct("Array", 2)?;
        state.serialize_field("layout", &self.layout)?;
        state.serialize_field("storage", &self.storage)?;
        state.end()
    }
}

impl<'de, T, S, const N: usize> Deserialize<'de> for Array<T, S, N>
where
    S: Deserialize<'de> + Storage<T>,
{
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ArrayParts<S, const N: usize> {
            layout: Layout<N>,
            storage: S,
        }

        let parts = ArrayParts::<S, N>::deserialize(deserializer)?;
        Self::new(parts.layout, parts.storage).map_err(D::Error::custom)
    }
}

impl<T, S, const N: usize> Array<T, S, N>
where
    S: Storage<T>,
{
    /// Create a new Array from a layout and a storage backing.
    ///
    /// # Errors
    /// Returns an error if the layout accesses memory before offset 0, or exceeds the storage bounds.
    pub fn new(layout: Layout<N>, storage: S) -> Result<Self> {
        layout.validate_storage_len(storage.len())?;

        Ok(Self {
            layout,
            storage,
            _marker: PhantomData,
        })
    }

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
        ElementIter::new(&self.view())
    }

    /// Iterator over `(multi-index, &element)` pairs in logical row-major order
    /// (leto `indexed_iter` parity).
    #[inline]
    pub fn indexed_iter(&self) -> IndexedIter<'_, T, N> {
        IndexedIter::new(&self.view())
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
        ExactChunks::new(&self.view(), chunk_shape)
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
        AxisChunks::new(&self.view(), axis, chunk_len)
    }

    /// Zero-copy iterator over every sliding window of shape `window_shape`
    /// (leto `windows` parity).
    ///
    /// # Errors
    /// [`LetoError`] if any `window_shape[i]` is `0` or exceeds
    /// `shape[i]`.
    #[inline]
    pub fn windows(&self, window_shape: [usize; N]) -> Result<Windows<'_, T, N>> {
        Windows::new(&self.view(), window_shape)
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
        let sliced_layout = self.layout.slice(ranges)?;
        Ok(ArrayView::new(sliced_layout, self.storage.as_slice()))
    }

    /// Slice the array with leto-style arguments, returning a read-only view.
    #[inline]
    pub fn slice_with<const M: usize>(&self, args: &[SliceArg]) -> Result<ArrayView<'_, T, M>> {
        let sliced_layout = self.layout.slice_with(args)?;
        Ok(ArrayView::new(sliced_layout, self.storage.as_slice()))
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
        let transposed_layout = self.layout.transpose(axes)?;
        Ok(ArrayView::new(transposed_layout, self.storage.as_slice()))
    }

    /// Broadcast the array, returning a read-only view.
    #[inline]
    pub fn broadcast<const M: usize>(
        &self,
        target_shape: [usize; M],
    ) -> Result<ArrayView<'_, T, M>> {
        let broadcasted_layout = self.layout.broadcast(target_shape)?;
        Ok(ArrayView::new(broadcasted_layout, self.storage.as_slice()))
    }

    /// Reinterpret this array with a new shape without copying.
    ///
    /// The current layout must be dense row-major and the new shape must have
    /// the same logical element count.
    #[inline]
    pub fn reshape<const M: usize>(&self, shape: [usize; M]) -> Result<ArrayView<'_, T, M>> {
        let reshaped_layout = self.layout.reshape(shape)?;
        Ok(ArrayView::new(reshaped_layout, self.storage.as_slice()))
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
        self.transpose(axes)
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

impl<T, S, const N: usize> Array<T, S, N>
where
    S: StorageMut<T>,
{
    /// Returns a mutable view of this array.
    #[inline]
    pub fn view_mut(&mut self) -> ArrayViewMut<'_, T, N> {
        ArrayViewMut::new(self.layout, self.storage.as_mut_slice())
    }

    /// Fill every physical storage slot with `value`.
    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.storage.as_mut_slice().fill(value);
    }

    /// Iterator over the array's elements as mutable references, in logical
    /// row-major order (`leto iter_mut` parity).
    ///
    /// # Panics
    /// Panics if the layout is not C-contiguous.
    #[inline]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.as_slice_mut()
            .expect("iter_mut: array must be C-contiguous")
            .iter_mut()
    }

    /// Iterator over `(multi-index, &mut element)` pairs in logical row-major
    /// order (leto `indexed_iter_mut` parity).
    ///
    /// # Errors
    /// Returns [`LetoError`] if the layout is out of bounds or cannot prove
    /// that each logical index addresses a distinct physical element.
    #[inline]
    pub fn indexed_iter_mut(&mut self) -> Result<IndexedIterMut<'_, T, N>> {
        self.view_mut().indexed_iter_mut()
    }

    /// Iterator over mutable elements in logical row-major order, preserving
    /// arbitrary positive and negative strides without materializing a copy.
    ///
    /// # Errors
    /// Returns [`LetoError`] if the layout is out of bounds or cannot prove
    /// that each logical index addresses a distinct physical element. In
    /// particular, zero-stride broadcast layouts are rejected before any
    /// mutable reference is yielded.
    #[inline]
    pub fn try_iter_mut(&mut self) -> Result<ElementIterMut<'_, T, N>> {
        self.view_mut().try_iter_mut()
    }

    /// Split the logical row-major domain into disjoint mutable task partitions.
    ///
    /// The partition iterator validates the complete layout once and leaves
    /// scheduling to the caller. Each partition is a move-only range token
    /// suitable for a scoped execution provider.
    ///
    /// # Errors
    /// Returns [`LetoError`] when `chunk_size` is zero, storage is invalid, or
    /// the layout is not provably injective.
    #[inline]
    pub fn task_partitions_mut(
        &mut self,
        chunk_size: usize,
    ) -> Result<TaskPartitionsMut<'_, T, N>> {
        self.view_mut().task_partitions_mut(chunk_size)
    }

    /// The elements as one mutable contiguous slice in logical row-major order,
    /// or `None` if the array is not C-contiguous (leto `as_slice_mut`
    /// parity). The safe basis for in-place element iteration: `if let Some(s) =
    /// a.as_slice_mut() { for x in s.iter_mut() { … } }`.
    #[inline]
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        if self.layout.is_c_dense() {
            let start = self.layout.offset();
            let end = start.checked_add(self.layout.checked_size().ok()?)?;
            self.storage.as_mut_slice().get_mut(start..end)
        } else {
            None
        }
    }

    /// Expose the mutable dense physical-memory slice when this array is contiguous.
    #[inline]
    pub fn as_slice_memory_order_mut(&mut self) -> Option<&mut [T]> {
        if self.layout.is_contiguous() {
            let start = self.layout.offset();
            let end = start.checked_add(self.layout.checked_size().ok()?)?;
            self.storage.as_mut_slice().get_mut(start..end)
        } else {
            None
        }
    }

    /// Zero-copy iterator over the mutable 1-D lanes along `axis`
    /// (leto `lanes_mut` parity; `M = N - 1`).
    ///
    /// # Errors
    /// [`LetoError`] if `axis >= N`, the layout does not fit
    /// its storage, or the layout aliases (a zero stride).
    #[inline]
    pub fn lanes_mut<const M: usize>(&mut self, axis: usize) -> Result<LanesMut<'_, T, N, M>>
    where
        crate::domain::remove_axis::RankMarker<N>: crate::domain::remove_axis::RemoveAxis<
            N,
            SmallerShape = [usize; M],
            SmallerStrides = [isize; M],
        >,
    {
        self.view_mut().lanes_mut(axis)
    }

    /// Slice the array, returning a mutable view.
    #[inline]
    pub fn slice_mut(
        &mut self,
        ranges: &[(usize, usize, isize); N],
    ) -> Result<ArrayViewMut<'_, T, N>> {
        let sliced_layout = self.layout.slice(ranges)?;
        Ok(ArrayViewMut::new(
            sliced_layout,
            self.storage.as_mut_slice(),
        ))
    }

    /// Slice the array with leto-style arguments, returning a mutable view.
    #[inline]
    pub fn slice_with_mut<const M: usize>(
        &mut self,
        args: &[SliceArg],
    ) -> Result<ArrayViewMut<'_, T, M>> {
        let sliced_layout = self.layout.slice_with(args)?;
        Ok(ArrayViewMut::new(
            sliced_layout,
            self.storage.as_mut_slice(),
        ))
    }

    /// Fix one axis at `index`, reducing the rank by 1 (leto `index_axis_mut` parity).
    ///
    /// `M` must equal `N - 1`; a mismatch returns `LetoError` from `slice_with_mut`.
    #[inline]
    pub fn index_axis_mut<const M: usize>(
        &mut self,
        axis: usize,
        index: usize,
    ) -> Result<ArrayViewMut<'_, T, M>> {
        let args: Vec<SliceArg> = (0..N)
            .map(|i| {
                if i == axis {
                    SliceArg::Index(index as isize)
                } else {
                    SliceArg::All
                }
            })
            .collect();
        self.slice_with_mut::<M>(&args)
    }

    /// Transpose the array, returning a mutable view.
    #[inline]
    pub fn transpose_mut(&mut self, axes: [usize; N]) -> Result<ArrayViewMut<'_, T, N>> {
        let transposed_layout = self.layout.transpose(axes)?;
        Ok(ArrayViewMut::new(
            transposed_layout,
            self.storage.as_mut_slice(),
        ))
    }

    /// Reinterpret this mutable array with a new shape without copying.
    ///
    /// The current layout must be dense row-major and the new shape must have
    /// the same logical element count.
    #[inline]
    pub fn reshape_mut<const M: usize>(
        &mut self,
        shape: [usize; M],
    ) -> Result<ArrayViewMut<'_, T, M>> {
        let reshaped_layout = self.layout.reshape(shape)?;
        Ok(ArrayViewMut::new(
            reshaped_layout,
            self.storage.as_mut_slice(),
        ))
    }

    /// Named mutable alias for [`transpose_mut`](Self::transpose_mut).
    #[inline]
    pub fn permute_mut(&mut self, axes: [usize; N]) -> Result<ArrayViewMut<'_, T, N>> {
        self.transpose_mut(axes)
    }

    /// Get a mutable reference to the element at the specified index.
    #[inline]
    pub fn get_mut(&mut self, index: [usize; N]) -> Result<&mut T> {
        let offset = self.layout.offset_of(index)?;
        let slice = self.storage.as_mut_slice();
        if offset >= slice.len() {
            return Err(LetoError::StorageError {
                reason: format!(
                    "physical offset {offset} exceeds backing slice length {}",
                    slice.len()
                ),
            });
        }
        Ok(&mut slice[offset])
    }

    /// Assign all elements from another array with the same shape.
    ///
    /// # Errors
    /// Returns [`LetoError`] when the shapes differ.
    #[inline]
    pub fn try_assign<Rhs>(&mut self, rhs: &Rhs) -> Result<()>
    where
        T: Copy,
        Rhs: AssignSource<T, N>,
    {
        let shape = self.shape();
        if shape != rhs.assign_shape() {
            return Err(LetoError::ShapeMismatch {
                lhs: shape.to_vec(),
                rhs: rhs.assign_shape().to_vec(),
            });
        }

        for linear in 0..self.size() {
            let index = linear_to_index(linear, shape);
            *self.get_mut(index)? = *rhs.assign_get(index)?;
        }
        Ok(())
    }

    /// Assign all elements from another array with the same shape.
    ///
    /// # Panics
    /// Panics when the shapes differ.
    #[inline]
    pub fn assign<Rhs>(&mut self, rhs: &Rhs)
    where
        T: Copy,
        Rhs: AssignSource<T, N>,
    {
        self.try_assign(rhs)
            .expect("invariant: assigned arrays have identical shape");
    }

    /// Add `alpha * rhs` to `self` in place.
    ///
    /// # Panics
    /// Panics when the shapes differ.
    #[inline]
    pub fn scaled_add<S2>(&mut self, alpha: T, rhs: &Array<T, S2, N>)
    where
        T: Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
        S2: Storage<T>,
    {
        let shape = self.shape();
        assert_eq!(
            shape,
            rhs.shape(),
            "scaled_add requires matching shapes: lhs {:?}, rhs {:?}",
            shape,
            rhs.shape()
        );
        for linear in 0..self.size() {
            let index = linear_to_index(linear, shape);
            let value = *self
                .get(index)
                .expect("invariant: logical index is in bounds")
                + alpha
                    * *rhs
                        .get(index)
                        .expect("invariant: rhs logical index is in bounds");
            *self
                .get_mut(index)
                .expect("invariant: logical index is in bounds") = value;
        }
    }
}

fn linear_to_index<const N: usize>(mut linear: usize, shape: [usize; N]) -> [usize; N] {
    let mut index = [0; N];
    for axis in (0..N).rev() {
        let extent = shape[axis];
        if extent != 0 {
            index[axis] = linear % extent;
            linear /= extent;
        }
    }
    index
}

impl<T, S, const N: usize> Eq for Array<T, S, N>
where
    T: Eq,
    S: Storage<T>,
{
}

impl<T, S, const N: usize> std::ops::Index<[usize; N]> for Array<T, S, N>
where
    S: Storage<T>,
{
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        self.get(index)
            .expect("invariant: array index is within shape and storage bounds")
    }
}

impl<T, S> std::ops::Index<usize> for Array<T, S, 1>
where
    S: Storage<T>,
{
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.get([index])
            .expect("invariant: array index is within shape and storage bounds")
    }
}

impl<T, S, const N: usize> std::ops::IndexMut<[usize; N]> for Array<T, S, N>
where
    S: StorageMut<T>,
{
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        self.get_mut(index)
            .expect("invariant: array index is within shape and storage bounds")
    }
}

impl<T, S> std::ops::IndexMut<usize> for Array<T, S, 1>
where
    S: StorageMut<T>,
{
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut([index])
            .expect("invariant: array index is within shape and storage bounds")
    }
}

#[cfg(test)]
mod scaled_add_tests {
    use crate::application::array::Array;
    use crate::infrastructure::storage::VecStorage;

    fn array(data: Vec<f64>) -> Array<f64, VecStorage<f64>, 2> {
        Array::<f64, VecStorage<f64>, 2>::from_shape_vec([2, 2], data).unwrap()
    }

    #[test]
    fn scaled_add_accumulates_scaled_source() {
        let mut dst = array(vec![1.0, 2.0, 3.0, 4.0]);
        let src = array(vec![10.0, 20.0, 30.0, 40.0]);

        dst.scaled_add(0.5, &src);

        assert_eq!(
            dst.iter().copied().collect::<Vec<_>>(),
            vec![6.0, 12.0, 18.0, 24.0]
        );
    }

    #[test]
    #[should_panic(expected = "matching shapes")]
    fn scaled_add_shape_mismatch_panics() {
        let mut dst = array(vec![0.0; 4]);
        let src = Array::<f64, VecStorage<f64>, 2>::from_shape_vec([4, 1], vec![0.0; 4]).unwrap();

        dst.scaled_add(1.0, &src);
    }
}

#[cfg(test)]
mod slice_access_tests {
    use crate::application::array::Array;
    use crate::infrastructure::storage::VecStorage;

    #[test]
    fn as_slice_and_as_slice_mut_on_contiguous() {
        let mut a = Array::<f64, VecStorage<f64>, 2>::from_shape_vec(
            [2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        assert_eq!(a.as_slice(), Some(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..]));
        // mutate every element in place through the mutable slice (the iter_mut basis)
        for (i, x) in a.as_slice_mut().unwrap().iter_mut().enumerate() {
            *x = i as f64 * 10.0;
        }
        assert_eq!(a.as_slice(), Some(&[0.0, 10.0, 20.0, 30.0, 40.0, 50.0][..]));
    }
}

/// Logical element-wise equality: two arrays are equal iff they have the same
/// shape and the same elements in logical row-major order (leto parity).
/// Correct across differing strides/storage, unlike a derived `PartialEq`.
impl<T, S1, S2, const N: usize> PartialEq<Array<T, S2, N>> for Array<T, S1, N>
where
    T: PartialEq,
    S1: Storage<T>,
    S2: Storage<T>,
{
    #[inline]
    fn eq(&self, other: &Array<T, S2, N>) -> bool {
        self.shape() == other.shape() && self.iter().eq(other.iter())
    }
}

#[cfg(test)]
mod partial_eq_tests {
    use crate::application::array::Array;
    use crate::infrastructure::storage::VecStorage;

    #[test]
    fn equal_iff_same_shape_and_elements() {
        let a = Array::<f64, VecStorage<f64>, 2>::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0])
            .unwrap();
        let b = Array::<f64, VecStorage<f64>, 2>::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0])
            .unwrap();
        let c = Array::<f64, VecStorage<f64>, 2>::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 9.0])
            .unwrap();
        let d = Array::<f64, VecStorage<f64>, 2>::from_shape_vec([4, 1], vec![1.0, 2.0, 3.0, 4.0])
            .unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d); // same data, different shape
    }
}
