use crate::application::array::Array;
use crate::application::index::index_from_flat;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
#[cfg(feature = "mnemosyne-alloc")]
use crate::infrastructure::storage::MnemosyneStorage;
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
            return Err(LetoError::StorageError {
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

    /// Create a new Array from a shape and a flat vector in C-contiguous order.
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
            vec.push(f(index_from_flat(flat_idx, &shape)));
        }
        let storage = VecStorage::new(vec);
        Self::new(layout, storage).expect("Valid layout bounds")
    }

    /// Consume the array and return its elements as a flat vector.
    ///
    /// C-contiguous arrays return their owned vector without copying. Strided
    /// arrays are copied in logical row-major order.
    pub fn into_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        if self.layout.is_c_contiguous() {
            return self.storage.into_inner();
        }

        let size = self.layout.size();
        let mut vec = Vec::with_capacity(size);
        let shape = self.layout.shape;
        for flat_idx in 0..size {
            let index = index_from_flat(flat_idx, &shape);
            let val = self.get(index).expect("validated layout index").clone();
            vec.push(val);
        }
        vec
    }
}

#[cfg(feature = "mnemosyne-alloc")]
impl<T, const N: usize> Array<T, MnemosyneStorage<T>, N> {
    /// Create a Mnemosyne-backed array of a given shape filled with `T::default()`.
    pub fn zeros_mnemosyne(shape: [usize; N]) -> Self
    where
        T: Default,
    {
        let layout = Layout::c_contiguous(shape).expect("C-contiguous layout must construct");
        let storage = MnemosyneStorage::new(layout.size());
        Self::new(layout, storage).expect("valid Mnemosyne storage bounds")
    }

    /// Create a Mnemosyne-backed array from a vector in C-contiguous order.
    pub fn from_mnemosyne_vec(shape: [usize; N], vec: Vec<T>) -> Result<Self> {
        let layout = Layout::c_contiguous(shape)?;
        let size = layout.size();
        if vec.len() != size {
            return Err(LetoError::StorageError {
                reason: format!(
                    "Vector length {} does not match layout size {}",
                    vec.len(),
                    size
                ),
            });
        }
        let storage = MnemosyneStorage::from_vec(vec);
        Self::new(layout, storage)
    }

    /// Create a Mnemosyne-backed array from a shape and flat vector.
    pub fn from_mnemosyne_shape_vec(shape: [usize; N], vec: Vec<T>) -> Result<Self> {
        Self::from_mnemosyne_vec(shape, vec)
    }

    /// Create a Mnemosyne-backed array by calling a generator for each coordinate.
    pub fn from_mnemosyne_shape_fn<F>(shape: [usize; N], mut f: F) -> Self
    where
        F: FnMut([usize; N]) -> T,
    {
        let layout = Layout::c_contiguous(shape).expect("C-contiguous layout must construct");
        let size = layout.size();
        let mut vec = Vec::with_capacity(size);
        for flat_idx in 0..size {
            vec.push(f(index_from_flat(flat_idx, &shape)));
        }
        let storage = MnemosyneStorage::from_vec(vec);
        Self::new(layout, storage).expect("Valid layout bounds")
    }

    /// Create a Mnemosyne-backed array by copying a C-contiguous source slice.
    pub fn from_mnemosyne_slice(shape: [usize; N], slice: &[T]) -> Result<Self>
    where
        T: Copy,
    {
        let layout = Layout::c_contiguous(shape)?;
        let size = layout.size();
        if slice.len() != size {
            return Err(LetoError::StorageError {
                reason: format!(
                    "Slice length {} does not match layout size {}",
                    slice.len(),
                    size
                ),
            });
        }
        let storage = MnemosyneStorage::from_slice(slice);
        Self::new(layout, storage)
    }

    /// Consume the Mnemosyne-backed array and return its elements as a vector.
    ///
    /// C-contiguous arrays move their owned storage without cloning. Strided
    /// arrays are copied in logical row-major order.
    pub fn into_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        if self.layout.is_c_contiguous() {
            return self.storage.into_vec();
        }

        let size = self.layout.size();
        let mut vec = Vec::with_capacity(size);
        let shape = self.layout.shape;
        for flat_idx in 0..size {
            let index = index_from_flat(flat_idx, &shape);
            let val = self.get(index).expect("validated layout index").clone();
            vec.push(val);
        }
        vec
    }
}
