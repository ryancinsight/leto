/// A trait representing read-only multidimensional array storage.
pub trait Storage<T> {
    /// Returns a reference to the underlying elements as a slice.
    fn as_slice(&self) -> &[T];

    /// Returns the number of physical elements in the storage.
    #[inline]
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Returns true if the storage contains no elements.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A trait representing mutable multidimensional array storage.
pub trait StorageMut<T>: Storage<T> {
    /// Returns a mutable reference to the underlying elements as a slice.
    fn as_mut_slice(&mut self) -> &mut [T];
}
