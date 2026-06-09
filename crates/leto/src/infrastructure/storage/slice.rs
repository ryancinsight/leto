use super::traits::{Storage, StorageMut};

/// Zero-copy read-only array storage borrowing an existing slice.
pub struct SliceStorage<'a, T> {
    slice: &'a [T],
}

impl<'a, T> SliceStorage<'a, T> {
    /// Create a new SliceStorage from a borrowed slice.
    #[inline]
    pub const fn new(slice: &'a [T]) -> Self {
        Self { slice }
    }
}

impl<'a, T> Storage<T> for SliceStorage<'a, T> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        self.slice
    }
}

/// Zero-copy mutable array storage borrowing an existing mutable slice.
pub struct SliceStorageMut<'a, T> {
    slice: &'a mut [T],
}

impl<'a, T> SliceStorageMut<'a, T> {
    /// Create a new SliceStorageMut from a borrowed mutable slice.
    #[inline]
    pub fn new(slice: &'a mut [T]) -> Self {
        Self { slice }
    }
}

impl<'a, T> Storage<T> for SliceStorageMut<'a, T> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        self.slice
    }
}

impl<'a, T> StorageMut<T> for SliceStorageMut<'a, T> {
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.slice
    }
}
