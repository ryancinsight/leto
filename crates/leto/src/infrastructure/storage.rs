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

// ── SliceStorage ──

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

// ── SliceStorageMut ──

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

// ── VecStorage ──

/// Owned array storage backed by a standard heap `Vec`.
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
        let mut data = Vec::with_capacity(len);
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

// ── MnemosyneStorage ──

/// Aligned, owned array storage allocated via the Mnemosyne systems allocator.
#[cfg(feature = "mnemosyne-alloc")]
pub struct MnemosyneStorage<T> {
    ptr: *mut T,
    len: usize,
    layout: std::alloc::Layout,
}

#[cfg(feature = "mnemosyne-alloc")]
impl<T> MnemosyneStorage<T> {
    fn allocate_raw(len: usize) -> Self {
        use std::alloc::{GlobalAlloc, Layout as AllocLayout};

        let size = len
            .checked_mul(std::mem::size_of::<T>())
            .expect("MnemosyneStorage allocation size overflow");
        let align = std::mem::align_of::<T>();
        let layout = AllocLayout::from_size_align(size, align)
            .expect("Invalid layout construction for MnemosyneStorage");

        if size == 0 {
            return Self {
                ptr: std::ptr::NonNull::<T>::dangling().as_ptr(),
                len,
                layout,
            };
        }

        // SAFETY: `layout` is constructible and verified above.
        let ptr = unsafe { mnemosyne::Mnemosyne.alloc(layout) } as *mut T;
        if ptr.is_null() {
            panic!("Mnemosyne allocation failed");
        }

        Self { ptr, len, layout }
    }
}

#[cfg(feature = "mnemosyne-alloc")]
impl<T: Default> MnemosyneStorage<T> {
    /// Allocate storage via Mnemosyne and initialize each element with `T::default()`.
    pub fn new(len: usize) -> Self {
        let storage = Self::allocate_raw(len);
        if !storage.ptr.is_null() {
            for index in 0..len {
                // SAFETY: `index < len`; allocation holds `len` elements.
                unsafe {
                    storage.ptr.add(index).write(T::default());
                }
            }
        }
        storage
    }
}

#[cfg(feature = "mnemosyne-alloc")]
impl<T> MnemosyneStorage<T> {
    /// Allocate storage and copy elements from a slice.
    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Copy,
    {
        let storage = Self::allocate_raw(slice.len());
        if !storage.ptr.is_null() {
            // SAFETY: source and destination are valid for `slice.len()` non-overlapping elements.
            unsafe {
                std::ptr::copy_nonoverlapping(slice.as_ptr(), storage.ptr, slice.len());
            }
        }
        storage
    }
}

#[cfg(feature = "mnemosyne-alloc")]
impl<T> Drop for MnemosyneStorage<T> {
    #[inline]
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.layout.size() > 0 {
            for index in 0..self.len {
                // SAFETY: all public constructors initialize every element.
                unsafe {
                    std::ptr::drop_in_place(self.ptr.add(index));
                }
            }
            use std::alloc::GlobalAlloc;
            // SAFETY: `self.ptr` was previously allocated with this layout.
            unsafe {
                mnemosyne::Mnemosyne.dealloc(self.ptr as *mut u8, self.layout);
            }
        }
    }
}

#[cfg(feature = "mnemosyne-alloc")]
unsafe impl<T: Send> Send for MnemosyneStorage<T> {}
#[cfg(feature = "mnemosyne-alloc")]
unsafe impl<T: Sync> Sync for MnemosyneStorage<T> {}

#[cfg(feature = "mnemosyne-alloc")]
impl<T> Storage<T> for MnemosyneStorage<T> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: `self.ptr` points to valid memory of `self.len` elements.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

#[cfg(feature = "mnemosyne-alloc")]
impl<T> StorageMut<T> for MnemosyneStorage<T> {
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: `self.ptr` points to valid mutable memory of `self.len` elements.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }
}
