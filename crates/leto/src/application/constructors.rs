use crate::application::array::Array;
use crate::application::index::index_from_flat;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;
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
