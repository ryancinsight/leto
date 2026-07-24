//! `ArrayD<T, S>`: a runtime-rank array **boundary carrier** (ADR 0007).
//!
//! `ArrayD` holds data whose rank is known only at run time (PyO3 `leto`
//! arrays, generic I/O). It supports the rank-agnostic operations — construct,
//! inspect, index, reshape, materialize — and **nothing else**: all numeric
//! computation is performed after recovering a typed
//! [`Array`](crate::application::array::Array) with
//! [`into_dimensionality`](ArrayD::into_dimensionality) (the bridge module). This
//! keeps every compute kernel single-authored on the const-rank core (SSOT) while
//! still admitting runtime-rank data at the edges.

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use crate::domain::dynamic::LayoutDyn;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::kernels;
use crate::infrastructure::storage::{Storage, StorageMut, VecStorage};

/// A runtime-rank strided array: a [`LayoutDyn`] over a storage backing `S`.
///
/// The rank is a value (`ndim()`), not a type parameter. For all arithmetic,
/// bridge to a const-rank [`Array`](crate::application::array::Array) via
/// [`into_dimensionality`](Self::into_dimensionality).
#[derive(Debug, Clone)]
pub struct ArrayD<T, S> {
    pub(crate) layout: LayoutDyn,
    pub(crate) storage: S,
    pub(crate) _marker: PhantomData<T>,
}

impl<T, S> ArrayD<T, S>
where
    S: Storage<T>,
{
    /// Construct from a runtime-rank layout and storage, validating that the
    /// layout's addressable range fits the storage.
    ///
    /// # Errors
    /// [`LetoError::StorageError`] if the layout exceeds the storage bounds.
    pub fn new(layout: LayoutDyn, storage: S) -> Result<Self> {
        layout.validate_storage_len(storage.len())?;
        Ok(Self {
            layout,
            storage,
            _marker: PhantomData,
        })
    }

    /// The rank (number of axes).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    /// The shape (extent of each axis).
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.layout.shape
    }

    /// The strides (in elements) of each axis.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.layout.strides
    }

    /// The starting offset into the backing storage.
    #[inline]
    pub fn offset(&self) -> usize {
        self.layout.offset
    }

    /// The logical element count `∏ shapeᵢ`.
    #[inline]
    pub fn size(&self) -> usize {
        self.layout.size()
    }

    /// Returns true when the array has no logical elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// The runtime-rank layout.
    #[inline]
    pub fn layout(&self) -> &LayoutDyn {
        &self.layout
    }

    /// A reference to the underlying storage backing.
    #[inline]
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// A mutable reference to the underlying storage backing.
    #[inline]
    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    /// Reference to the element at `index` (length must equal [`ndim`](Self::ndim)).
    ///
    /// # Errors
    /// [`LetoError::OutOfBounds`] on wrong arity or an out-of-range component.
    #[inline]
    pub fn get(&self, index: &[usize]) -> Result<&T> {
        let offset = self.layout.offset_of(index)?;
        self.storage
            .as_slice()
            .get(offset)
            .ok_or(LetoError::StorageError {
                reason: format!(
                    "physical offset {offset} exceeds backing slice length {}",
                    self.storage.len()
                ),
            })
    }
}

impl<T> ArrayD<T, VecStorage<T>> {
    /// Construct a C-contiguous array from a runtime shape and a flat row-major
    /// vector.
    ///
    /// # Errors
    /// [`LetoError`] if the layout overflows or `data.len()` does not equal the
    /// shape's element count.
    pub fn from_shape_vec(shape: &[usize], data: Vec<T>) -> Result<Self> {
        let layout = LayoutDyn::c_contiguous(shape)?;
        let size = layout.size();
        if data.len() != size {
            return Err(LetoError::StorageError {
                reason: format!(
                    "vector length {} does not match layout size {size}",
                    data.len()
                ),
            });
        }
        Self::new(layout, VecStorage::new(data))
    }

    /// Construct a C-contiguous array filled with `T::default()`.
    ///
    /// # Errors
    /// [`LetoError::Overflow`] if the shape's element count overflows.
    pub fn zeros(shape: &[usize]) -> Result<Self>
    where
        T: Default + Clone,
    {
        let layout = LayoutDyn::c_contiguous(shape)?;
        let size = layout.size();
        Self::new(layout, VecStorage::fill(size, T::default()))
    }

    /// Construct a C-contiguous array filled with `value` (leto `from_elem` parity).
    ///
    /// # Errors
    /// [`LetoError::Overflow`] if the shape's element count overflows.
    pub fn from_elem(shape: &[usize], value: T) -> Result<Self>
    where
        T: Clone,
    {
        let layout = LayoutDyn::c_contiguous(shape)?;
        let size = layout.size();
        Self::new(layout, VecStorage::fill(size, value))
    }

    /// Iterator over the logical row-major elements (leto `iter` parity).
    ///
    /// The backing `VecStorage` is always contiguous, so this is a slice
    /// iterator — O(1) construction and no extra allocation.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.storage.as_slice().iter()
    }

    /// Mutable iterator over the logical row-major elements
    /// (leto `iter_mut` parity, contiguous VecStorage only).
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.storage.as_mut_slice().iter_mut()
    }

    /// Materialize the elements in logical row-major order.
    ///
    /// Walks the (possibly strided) layout through the shared offset kernels —
    /// the runtime-rank analogue of the const-rank `into_vec`.
    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Clone,
    {
        if self.layout.is_c_contiguous() {
            return Ok(self.storage.as_slice()
                [self.layout.offset..self.layout.offset + self.size()]
                .to_vec());
        }
        let size = self.size();
        let ndim = self.ndim();
        let mut out: Vec<T> = Vec::with_capacity(size);
        let mut index = vec![0usize; ndim];
        for flat in 0..size {
            kernels::fill_index_from_flat(flat, &self.layout.shape, &mut index);
            out.push(self.get(&index)?.clone());
        }
        Ok(out)
    }

    /// Reshape to a new runtime shape, reusing the storage (zero-copy).
    ///
    /// Requires a C-contiguous source (the same contract as the const-rank
    /// `reshape`); strided arrays must be materialized first.
    ///
    /// # Errors
    /// [`LetoError::ShapeMismatch`] if the element counts differ;
    /// [`LetoError::StorageError`] if the source is not C-contiguous.
    pub fn into_shape(self, shape: &[usize]) -> Result<Self> {
        let target = LayoutDyn::c_contiguous(shape)?;
        if target.checked_size()? != self.layout.checked_size()? {
            return Err(LetoError::ShapeMismatch {
                lhs: self.layout.shape.to_vec(),
                rhs: shape.to_vec(),
            });
        }
        if !self.layout.is_c_contiguous() {
            return Err(LetoError::StorageError {
                reason: "into_shape requires a C-contiguous layout".to_string(),
            });
        }
        Ok(Self {
            layout: target,
            storage: self.storage,
            _marker: PhantomData,
        })
    }
}

impl<T, S> ArrayD<T, S>
where
    S: StorageMut<T>,
{
    /// Mutable reference to the element at `index` (length must equal
    /// [`ndim`](Self::ndim)).
    ///
    /// # Errors
    /// [`LetoError::OutOfBounds`] on wrong arity or an out-of-range component.
    /// [`LetoError::StorageError`] when the physical offset exceeds the backing slice.
    #[inline]
    pub fn get_mut(&mut self, index: &[usize]) -> Result<&mut T> {
        let offset = self.layout.offset_of(index)?;
        let len = self.storage.len();
        self.storage
            .as_mut_slice()
            .get_mut(offset)
            .ok_or(LetoError::StorageError {
                reason: format!("physical offset {offset} exceeds backing slice length {len}"),
            })
    }

    /// Set the element at `index` to `value`.
    ///
    /// Convenience wrapper over [`get_mut`](Self::get_mut).
    ///
    /// # Errors
    /// Propagates errors from [`get_mut`](Self::get_mut).
    #[inline]
    pub fn set(&mut self, index: &[usize], value: T) -> Result<()> {
        *self.get_mut(index)? = value;
        Ok(())
    }
}

impl<T> ArrayD<T, VecStorage<T>>
where
    T: Copy,
{
    /// Fill every element with `value` in logical row-major order.
    pub fn fill(&mut self, value: T) {
        for slot in self.storage.as_mut_slice() {
            *slot = value;
        }
    }
}

// ── Index / IndexMut for dynamic multi-index slices ───────────────────────────

/// Panic-on-error index accessor. `array[&idx[..]]` panics on bounds violations.
///
/// Use [`ArrayD::get`] for fallible access.
impl<T, S: Storage<T>> Index<&[usize]> for ArrayD<T, S> {
    type Output = T;

    #[inline]
    fn index(&self, index: &[usize]) -> &T {
        self.get(index)
            .expect("ArrayD index out of bounds or wrong rank")
    }
}

/// Panic-on-error mutable index accessor. `array[&idx[..]] = v` panics on
/// bounds violations.
///
/// Use [`ArrayD::get_mut`] for fallible access.
impl<T, S: StorageMut<T>> IndexMut<&[usize]> for ArrayD<T, S> {
    #[inline]
    fn index_mut(&mut self, index: &[usize]) -> &mut T {
        self.get_mut(index)
            .expect("ArrayD index_mut out of bounds or wrong rank")
    }
}
