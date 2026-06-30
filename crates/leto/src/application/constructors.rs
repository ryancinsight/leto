use crate::application::array::Array;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;

#[inline]
fn increment_index<const N: usize>(index: &mut [usize; N], shape: &[usize; N]) {
    for i in (0..N).rev() {
        index[i] += 1;
        if index[i] < shape[i] {
            break;
        }
        index[i] = 0;
    }
}
#[cfg(feature = "mnemosyne-alloc")]
use crate::infrastructure::storage::MnemosyneStorage;
use crate::infrastructure::storage::{StackStorage, VecStorage};

impl<T, const CAP: usize, const N: usize> Array<T, StackStorage<T, CAP>, N> {
    /// Create a stack-backed array from a runtime `shape` and an inline
    /// `[T; CAP]` in C-contiguous order — no heap allocation.
    ///
    /// # Errors
    /// [`LetoError`] if the shape's element count does not equal `CAP`.
    pub fn from_stack(shape: [usize; N], data: [T; CAP]) -> Result<Self> {
        let layout = Layout::c_contiguous(shape)?;
        if layout.size() != CAP {
            return Err(LetoError::StorageError {
                reason: format!(
                    "stack capacity {CAP} does not match shape element count {}",
                    layout.size()
                ),
            });
        }
        Self::new(layout, StackStorage::new(data))
    }

    /// Create a stack-backed array of `shape` filled with `value` (no heap).
    ///
    /// # Errors
    /// [`LetoError`] if the shape's element count does not equal `CAP`.
    pub fn from_stack_elem(shape: [usize; N], value: T) -> Result<Self>
    where
        T: Copy,
    {
        Self::from_stack(shape, [value; CAP])
    }
}

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
        T: eunomia::NumericElement + Clone,
    {
        let layout = Layout::c_contiguous(shape).expect("C-contiguous layout must construct");
        let size = layout.size();
        let storage = VecStorage::fill(size, T::ONE);
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
        let mut vec: Vec<T> = Vec::with_capacity(size);
        let mut index = [0usize; N];
        for _ in 0..size {
            vec.push(f(index));
            increment_index(&mut index, &shape);
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
        let mut vec: Vec<T> = Vec::with_capacity(size);
        vec.extend(self.iter().cloned());
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
        let mut vec: Vec<T> = Vec::with_capacity(size);
        let mut index = [0usize; N];
        for _ in 0..size {
            vec.push(f(index));
            increment_index(&mut index, &shape);
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
        let mut vec: Vec<T> = Vec::with_capacity(size);
        vec.extend(self.iter().cloned());
        vec
    }
}

/// Build a 1-D array from a `Vec` (its length becomes the shape), matching
/// `ndarray::Array1::from(vec)` / `from_vec`. The vector's storage is moved in
/// place — no copy.
impl<T> From<Vec<T>> for Array<T, VecStorage<T>, 1> {
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        let len = vec.len();
        Self::from_shape_vec([len], vec).expect("1-D length always matches a [len] shape")
    }
}

#[cfg(test)]
mod from_vec_tests {
    use super::Array;
    use crate::infrastructure::storage::VecStorage;

    #[test]
    fn array1_from_vec_uses_len_as_shape() {
        let a: Array<f64, VecStorage<f64>, 1> = vec![1.0, 2.0, 3.0].into();
        assert_eq!(a.shape(), [3]);
        assert_eq!(a.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }
}
