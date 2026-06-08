use crate::application::view::{ArrayView, ArrayViewMut};
use crate::domain::error::Result;
use crate::domain::layout::Layout;
use crate::domain::slice::SliceArg;
use crate::infrastructure::storage::{Storage, StorageMut};
use std::marker::PhantomData;

/// An N-dimensional strided array.
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

    /// Returns an immutable read-only view of this array.
    #[inline]
    pub fn view(&self) -> ArrayView<'_, T, N> {
        ArrayView::new(self.layout, self.storage.as_slice())
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

    /// Get a reference to the element at the specified index.
    #[inline]
    pub fn get(&self, index: [usize; N]) -> Result<&T> {
        let offset = self.layout.offset_of(index)?;
        Ok(&self.storage.as_slice()[offset])
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

    /// Get a mutable reference to the element at the specified index.
    #[inline]
    pub fn get_mut(&mut self, index: [usize; N]) -> Result<&mut T> {
        let offset = self.layout.offset_of(index)?;
        Ok(&mut self.storage.as_mut_slice()[offset])
    }
}
