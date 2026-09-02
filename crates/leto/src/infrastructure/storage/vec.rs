use super::traits::{Storage, StorageMut};
use serde::{Deserialize, Serialize};

/// Owned array storage backed by a standard heap `Vec`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::unsafe_derive_deserialize)]
pub struct VecStorage<T> {
    data: Vec<T>,
}

impl<T> VecStorage<T> {
    /// Create a new VecStorage of a given length, filled with elements using a generator function.
    #[inline]
    pub fn generate<F>(len: usize, mut f: F) -> Self
    where
        F: FnMut() -> T,
    {
        let mut data: Vec<T> = Vec::with_capacity(len);
        for _ in 0..len {
            data.push(f());
        }
        Self { data }
    }

    /// Create a new VecStorage of a given length, filled with cloneable elements.
    #[inline]
    pub fn fill(len: usize, value: T) -> Self
    where
        T: Clone,
    {
        Self {
            data: vec![value; len],
        }
    }

    /// Create uninitialized storage without zero-filling.
    ///
    /// # Safety
    ///
    /// The returned storage contains uninitialized memory. Every element must
    /// be written before being read, otherwise the behavior is undefined. This
    /// is safe when the caller fully overwrites the storage (e.g. a
    /// keep-dim reduction that writes every output element).
    #[inline]
    #[allow(clippy::uninit_vec)]
    pub fn uninit(len: usize) -> Self {
        let mut data = Vec::with_capacity(len);
        // SAFETY: `Vec::with_capacity` allocates `len` elements, `set_len`
        // exposes them as initialized for the caller to overwrite. The caller
        // must write every element before any read, which `reduce_axis` does.
        unsafe {
            data.set_len(len);
        }
        Self { data }
    }

    /// Wrap an existing Vec.
    #[inline]
    pub const fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Consume the storage and return the inner vector.
    #[inline]
    pub fn into_inner(self) -> Vec<T> {
        self.data
    }
}

impl<T> Storage<T> for VecStorage<T> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        &self.data
    }
}

impl<T> StorageMut<T> for VecStorage<T> {
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}
