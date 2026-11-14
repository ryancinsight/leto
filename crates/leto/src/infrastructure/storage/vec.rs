use super::traits::{Storage, StorageMut};
use serde::{Deserialize, Serialize};

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

impl<T: Copy> VecStorage<T> {
    /// Create storage of `len` elements without writing any of them.
    ///
    /// The write-only fast path for an allocating kernel that overwrites every
    /// element, such as a keep-dimension reduction: it skips the fill pass
    /// that [`fill`](Self::fill) performs. `T: Copy` excludes drop glue, so
    /// overwriting an unwritten element never destroys a value that was not
    /// constructed. Safe construction is [`generate`](Self::generate),
    /// [`fill`](Self::fill) and [`new`](Self::new).
    ///
    /// ```compile_fail,E0133
    /// use leto::VecStorage;
    ///
    /// let _ = VecStorage::<f64>::uninit(4);
    /// ```
    ///
    /// # Safety
    ///
    /// The caller must uphold all of the following:
    ///
    /// - `T` is valid for every bit pattern (an integer, a float, or a
    ///   composite of them), because the storage exposes `&mut [T]` and
    ///   `&[T]` over the unwritten elements before they are written;
    /// - every element is written before it is read, on every exit of the
    ///   code that holds the storage, including a successful return;
    /// - an element not yet written is never read, copied, compared or
    ///   serialized, and the storage is not cloned or compared before every
    ///   element is written.
    ///
    /// # Examples
    ///
    /// ```
    /// use leto::{Storage, StorageMut, VecStorage};
    ///
    /// // SAFETY: `f64` is valid for every bit pattern, and the copy below
    /// // writes every element before the storage is read.
    /// let mut storage = unsafe { VecStorage::<f64>::uninit(3) };
    /// storage.as_mut_slice().copy_from_slice(&[0.0, 1.0, 2.0]);
    /// assert_eq!(storage.as_slice(), &[0.0, 1.0, 2.0]);
    /// ```
    #[inline]
    #[expect(clippy::uninit_vec, reason = "the caller overwrites every element")]
    pub unsafe fn uninit(len: usize) -> Self {
        let mut data = Vec::with_capacity(len);
        // SAFETY: `with_capacity` allocated room for `len` elements and `T`
        // has no drop glue (`Copy`). The caller's contract (every bit
        // pattern valid, every element written before it is read) makes the
        // exposed elements sound to overwrite and then read.
        unsafe {
            data.set_len(len);
        }
        Self { data }
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
