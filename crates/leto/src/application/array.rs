use crate::application::iter::{ElementIter, IndexedIter, Lanes, LanesMut, Windows};
use crate::application::view::{ArrayView, ArrayViewMut};
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
use crate::domain::slice::SliceArg;
use crate::infrastructure::storage::{Storage, StorageMut};
use std::marker::PhantomData;

/// An N-dimensional strided array.
#[derive(Debug, Clone)]
pub struct Array<T, S, const N: usize> {
    pub(crate) layout: Layout<N>,
    pub(crate) storage: S,
    pub(crate) _marker: PhantomData<T>,
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
        self.layout.shape
    }

    /// Returns the strides of the array.
    #[inline]
    pub const fn strides(&self) -> [isize; N] {
        self.layout.strides
    }

    /// Returns the starting offset of the array.
    #[inline]
    pub const fn offset(&self) -> usize {
        self.layout.offset
    }

    /// Returns the total logical size of the array.
    #[inline]
    pub fn size(&self) -> usize {
        self.layout.size()
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
    /// `None` if the array is not C-contiguous (ndarray `as_slice` parity).
    #[inline]
    pub fn as_slice(&self) -> Option<&[T]> {
        if self.layout.is_c_dense() {
            let start = self.layout.offset;
            Some(&self.storage.as_slice()[start..start + self.layout.size()])
        } else {
            None
        }
    }

    /// Iterator over the array's elements in logical row-major order
    /// (ndarray `iter` parity), respecting arbitrary strides.
    #[inline]
    pub fn iter(&self) -> ElementIter<'_, T, N> {
        ElementIter::new(&self.view())
    }

    /// Iterator over `(multi-index, &element)` pairs in logical row-major order
    /// (ndarray `indexed_iter` parity).
    #[inline]
    pub fn indexed_iter(&self) -> IndexedIter<'_, T, N> {
        IndexedIter::new(&self.view())
    }

    /// Zero-copy iterator over every sliding window of shape `window_shape`
    /// (ndarray `windows` parity).
    ///
    /// # Errors
    /// [`LetoError`] if any `window_shape[i]` is `0` or exceeds
    /// `shape[i]`.
    #[inline]
    pub fn windows(&self, window_shape: [usize; N]) -> Result<Windows<'_, T, N>> {
        Windows::new(&self.view(), window_shape)
    }

    /// Zero-copy iterator over the read-only 1-D lanes along `axis`
    /// (ndarray `lanes` parity; `M = N - 1`).
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

    /// Slice the array with ndarray-style arguments, returning a read-only view.
    #[inline]
    pub fn slice_with<const M: usize>(&self, args: &[SliceArg]) -> Result<ArrayView<'_, T, M>> {
        let sliced_layout = self.layout.slice_with(args)?;
        Ok(ArrayView::new(sliced_layout, self.storage.as_slice()))
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

    /// The elements as one mutable contiguous slice in logical row-major order,
    /// or `None` if the array is not C-contiguous (ndarray `as_slice_mut`
    /// parity). The safe basis for in-place element iteration: `if let Some(s) =
    /// a.as_slice_mut() { for x in s.iter_mut() { … } }`.
    #[inline]
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        if self.layout.is_c_dense() {
            let start = self.layout.offset;
            let size = self.layout.size();
            Some(&mut self.storage.as_mut_slice()[start..start + size])
        } else {
            None
        }
    }

    /// Zero-copy iterator over the mutable 1-D lanes along `axis`
    /// (ndarray `lanes_mut` parity; `M = N - 1`).
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

    /// Slice the array with ndarray-style arguments, returning a mutable view.
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
/// shape and the same elements in logical row-major order (ndarray parity).
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
