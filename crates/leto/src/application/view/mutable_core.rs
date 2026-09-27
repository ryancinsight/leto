use super::immutable::ArrayView;
use super::raw::{
    assert_exclusive_window, raw_mut_from_ptr, raw_ref_from_ptr, raw_slice_from_ptr,
    raw_slice_from_ptr_mut,
};
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;

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
    #[inline]
    pub(super) fn with_layout<const M: usize>(self, layout: Layout<M>) -> ArrayViewMut<'a, T, M> {
        ArrayViewMut {
            layout,
            ptr: self.ptr,
            len: self.len,
            window_shared: self.window_shared,
            _marker: std::marker::PhantomData,
        }
    }

    #[inline]
    pub(super) fn element(&self, offset: usize) -> &T {
        // SAFETY: callers validate `offset < self.len`.
        unsafe { raw_ref_from_ptr(self.ptr, offset) }
    }

    #[inline]
    pub(super) fn element_mut(&mut self, offset: usize) -> &mut T {
        // SAFETY: callers validate `offset < self.len` and hold `&mut self`.
        unsafe { raw_mut_from_ptr(self.ptr, offset) }
    }

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
        assert_exclusive_window(self.window_shared);
        // SAFETY: `ptr` is valid for `len` elements for the duration of the
        // borrow of `self`, and the assertion above establishes the window is
        // exclusively owned, so no sibling view can mint `&mut` into it.
        let data = unsafe { raw_slice_from_ptr(self.ptr, self.len) };
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
        assert_exclusive_window(self.window_shared);
        // SAFETY: self.ptr is valid for self.len elements, and the assertion
        // above establishes the window is exclusively owned.
        unsafe { raw_slice_from_ptr(self.ptr, self.len) }
    }

    /// Returns the raw mutable data slice.
    ///
    /// # Panics
    ///
    /// Panics when the view was yielded by a mutable lane/axis iterator over an
    /// interleaved layout (see [`as_view`](Self::as_view)).
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        assert_exclusive_window(self.window_shared);
        // SAFETY: self.ptr is valid for self.len elements, and the assertion
        // above establishes the window is exclusively owned.
        unsafe { raw_slice_from_ptr_mut(self.ptr, self.len) }
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
        Ok(self.element(offset))
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
        Ok(self.element_mut(offset))
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
}
