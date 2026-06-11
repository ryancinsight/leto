use super::traits::{Storage, StorageMut};

/// Aligned, owned array storage allocated via the Mnemosyne systems allocator.
pub struct MnemosyneStorage<T> {
    ptr: *mut T,
    len: usize,
    layout: std::alloc::Layout,
}

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

    /// Allocate storage via Mnemosyne and move elements from a vector.
    pub fn from_vec(vec: Vec<T>) -> Self {
        let len = vec.len();
        let capacity = vec.capacity();
        let storage = Self::allocate_raw(len);
        let mut vec = std::mem::ManuallyDrop::new(vec);
        if !storage.ptr.is_null() && len > 0 {
            // SAFETY: source has `len` initialized elements, destination has
            // room for `len` elements, and the regions do not overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(vec.as_ptr(), storage.ptr, len);
            }
        }

        // SAFETY: the elements were moved into Mnemosyne storage above. The
        // reconstructed vector has length zero, so it releases only capacity.
        unsafe {
            let _ = Vec::from_raw_parts(vec.as_mut_ptr(), 0, capacity);
        }
        storage
    }

    /// Move initialized elements from Mnemosyne storage into a vector.
    pub fn into_vec(self) -> Vec<T> {
        let mut vec = Vec::with_capacity(self.len);
        let storage = std::mem::ManuallyDrop::new(self);
        if !storage.ptr.is_null() && storage.len > 0 {
            // SAFETY: source has `len` initialized elements, destination has
            // capacity for `len` elements, and the regions do not overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(storage.ptr, vec.as_mut_ptr(), storage.len);
                vec.set_len(storage.len);
            }
        }

        if storage.layout.size() > 0 {
            use std::alloc::GlobalAlloc;
            // SAFETY: `storage.ptr` was allocated by Mnemosyne with this layout.
            unsafe {
                mnemosyne::Mnemosyne.dealloc(storage.ptr as *mut u8, storage.layout);
            }
        }

        vec
    }
}

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

unsafe impl<T: Send> Send for MnemosyneStorage<T> {}
unsafe impl<T: Sync> Sync for MnemosyneStorage<T> {}

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
