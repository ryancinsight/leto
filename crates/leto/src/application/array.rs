use crate::application::view::{ArrayView, ArrayViewMut};
use crate::domain::error::Result;
use crate::domain::layout::Layout;
use crate::domain::slice::SliceArg;
use crate::infrastructure::storage::{Storage, StorageMut};
use std::marker::PhantomData;

/// An N-dimensional strided array.
pub struct Array<T, S, const N: usize> {
    layout: Layout<N>,
    storage: S,
    _marker: PhantomData<T>,
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

// ── ndarray API Parity Constructors ──
use crate::infrastructure::storage::VecStorage;

impl<T, const N: usize> Array<T, VecStorage<T>, N> {
    /// Create a new Array of a given shape filled with the default value of T.
    pub fn zeros(shape: [usize; N]) -> Self
    where
        T: Default + Clone,
    {
        let layout = Layout::c_contiguous(shape).expect("C-contiguous layout must construct");
        let size = layout.size();
        let storage = VecStorage::fill(size, T::default());
        Self::new(layout, storage).expect("Valid layout bounds")
    }

    /// Create a new Array of a given shape filled with a clone of `value`.
    pub fn from_elem(shape: [usize; N], value: T) -> Self
    where
        T: Clone,
    {
        let layout = Layout::c_contiguous(shape).expect("C-contiguous layout must construct");
        let size = layout.size();
        let storage = VecStorage::fill(size, value);
        Self::new(layout, storage).expect("Valid layout bounds")
    }

    /// Create a new Array of a given shape filled with one.
    pub fn ones(shape: [usize; N]) -> Self
    where
        T: num_traits::One + Clone,
    {
        let layout = Layout::c_contiguous(shape).expect("C-contiguous layout must construct");
        let size = layout.size();
        let storage = VecStorage::fill(size, T::one());
        Self::new(layout, storage).expect("Valid layout bounds")
    }

    /// Create a new Array from a vector of elements in C-contiguous order.
    pub fn from_vec(shape: [usize; N], vec: Vec<T>) -> Result<Self> {
        let layout = Layout::c_contiguous(shape)?;
        let size = layout.size();
        if vec.len() != size {
            return Err(crate::domain::error::LetoError::StorageError {
                reason: format!(
                    "Vector length {} does not match layout size {}",
                    vec.len(),
                    size
                ),
            });
        }
        let storage = VecStorage::new(vec);
        Self::new(layout, storage)
    }

    /// Create a new Array from a shape and a flat vector (standard C-contiguous).
    pub fn from_shape_vec(shape: [usize; N], vec: Vec<T>) -> Result<Self> {
        Self::from_vec(shape, vec)
    }

    /// Create an Array by calling a generator function for each coordinate.
    pub fn from_shape_fn<F>(shape: [usize; N], mut f: F) -> Self
    where
        F: FnMut([usize; N]) -> T,
    {
        let layout = Layout::c_contiguous(shape).expect("C-contiguous layout must construct");
        let size = layout.size();
        let mut vec = Vec::with_capacity(size);
        for flat_idx in 0..size {
            let mut index = [0usize; N];
            let mut temp = flat_idx;
            for i in (0..N).rev() {
                if shape[i] > 0 {
                    index[i] = temp % shape[i];
                    temp /= shape[i];
                }
            }
            vec.push(f(index));
        }
        let storage = VecStorage::new(vec);
        Self::new(layout, storage).expect("Valid layout bounds")
    }

    /// Consume the array and return its elements as a flat vector.
    /// If the array is contiguous, this is a zero-copy operation.
    /// If the array is not contiguous, the elements are copied in logical order.
    pub fn into_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        if self.layout.is_c_contiguous() {
            self.storage.into_inner()
        } else {
            let size = self.layout.size();
            let mut vec = Vec::with_capacity(size);
            let shape = self.layout.shape;
            for flat_idx in 0..size {
                let mut index = [0usize; N];
                let mut temp = flat_idx;
                for i in (0..N).rev() {
                    if shape[i] > 0 {
                        index[i] = temp % shape[i];
                        temp /= shape[i];
                    }
                }
                let val = self.get(index).unwrap().clone();
                vec.push(val);
            }
            vec
        }
    }
}
