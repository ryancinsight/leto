use super::traits::{Storage, StorageMut};
use serde::{Deserialize, Serialize};
use std::mem::MaybeUninit;

/// Owned array storage backed by a standard heap `Vec`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[expect(
    clippy::unsafe_derive_deserialize,
    reason = "VecStorage derive is safe for POD storage"
)]
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

    /// Allocate uninitialized slots without zero-filling or constructing `T`.
    ///
    /// The slots can be safely written or dropped. Reading a `T` requires
    /// initialization and an explicit unsafe conversion through
    /// [`assume_init`](VecStorage::assume_init).
    ///
    /// ```compile_fail
    /// use leto::{Storage, VecStorage};
    /// let storage = VecStorage::<u32>::uninit(1);
    /// let _: u32 = storage.as_slice()[0];
    /// ```
    #[inline]
    pub fn uninit(len: usize) -> VecStorage<MaybeUninit<T>> {
        let mut data = Vec::with_capacity(len);
        // SAFETY: capacity is at least len and every bit pattern, including
        // uninitialized memory, is valid for MaybeUninit<T>.
        unsafe {
            data.set_len(len);
        }
        VecStorage { data }
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

impl<T> VecStorage<MaybeUninit<T>> {
    /// Consume fully initialized slots without copying the allocation.
    ///
    /// # Safety
    ///
    /// Every slot must hold a valid, uniquely owned `T`. Each value becomes
    /// owned by the returned storage and will be dropped normally. Initialized
    /// resource-owning values in abandoned slots must be dropped by the caller
    /// before dropping unconverted storage.
    ///
    /// ```compile_fail
    /// use leto::VecStorage;
    /// let storage = VecStorage::<u32>::uninit(1);
    /// let _ = storage.assume_init();
    /// ```
    #[inline]
    pub unsafe fn assume_init(self) -> VecStorage<T> {
        let mut data = std::mem::ManuallyDrop::new(self.data);
        // SAFETY: the caller establishes initialization and ownership of all
        // len elements. MaybeUninit<T> has T's layout, including alignment;
        // the original pointer/capacity preserve the allocation's layout.
        let data = unsafe {
            Vec::from_raw_parts(data.as_mut_ptr().cast::<T>(), data.len(), data.capacity())
        };
        VecStorage { data }
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

#[cfg(test)]
#[path = "vec_tests.rs"]
mod tests;
