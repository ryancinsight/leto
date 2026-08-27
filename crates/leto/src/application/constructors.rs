#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use crate::application::array::Array;
use crate::domain::error::{LetoError, Result};
use crate::domain::layout::Layout;

pub trait IntoShape<const N: usize> {
    fn into_shape(self) -> [usize; N];
}

impl<const N: usize> IntoShape<N> for [usize; N] {
    #[inline]
    fn into_shape(self) -> [usize; N] {
        self
    }
}

impl IntoShape<1> for usize {
    #[inline]
    fn into_shape(self) -> [usize; 1] {
        [self]
    }
}

impl IntoShape<2> for (usize, usize) {
    #[inline]
    fn into_shape(self) -> [usize; 2] {
        [self.0, self.1]
    }
}

impl IntoShape<3> for (usize, usize, usize) {
    #[inline]
    fn into_shape(self) -> [usize; 3] {
        [self.0, self.1, self.2]
    }
}

impl IntoShape<4> for (usize, usize, usize, usize) {
    #[inline]
    fn into_shape(self) -> [usize; 4] {
        [self.0, self.1, self.2, self.3]
    }
}

impl IntoShape<5> for (usize, usize, usize, usize, usize) {
    #[inline]
    fn into_shape(self) -> [usize; 5] {
        [self.0, self.1, self.2, self.3, self.4]
    }
}

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
    pub fn from_stack(shape: impl IntoShape<N>, data: [T; CAP]) -> Result<Self> {
        let shape = shape.into_shape();
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
    pub fn from_stack_elem(shape: impl IntoShape<N>, value: T) -> Result<Self>
    where
        T: Copy,
    {
        Self::from_stack(shape, [value; CAP])
    }
}

impl<T, const N: usize> Array<T, VecStorage<T>, N> {
    /// Create a new Array of a given shape filled with the default value of T.
    pub fn zeros(shape: impl IntoShape<N>) -> Self
    where
        T: Default + Clone,
    {
        let shape = shape.into_shape();
        let layout = Layout::c_contiguous(shape).expect("C-contiguous layout must construct");
        let size = layout.size();
        let storage = VecStorage::fill(size, T::default());
        Self::new(layout, storage).expect("Valid layout bounds")
    }

    /// Create a new Array of a given shape filled with a clone of `value`.
    pub fn from_elem(shape: impl IntoShape<N>, value: T) -> Self
    where
        T: Clone,
    {
        let shape = shape.into_shape();
        let layout = Layout::c_contiguous(shape).expect("C-contiguous layout must construct");
        let size = layout.size();
        let storage = VecStorage::fill(size, value);
        Self::new(layout, storage).expect("Valid layout bounds")
    }

    /// Create a new Array of a given shape filled with one.
    pub fn ones(shape: impl IntoShape<N>) -> Self
    where
        T: eunomia::NumericElement + Clone,
    {
        let shape = shape.into_shape();
        let layout = Layout::c_contiguous(shape).expect("C-contiguous layout must construct");
        let size = layout.size();
        let storage = VecStorage::fill(size, T::ONE);
        Self::new(layout, storage).expect("Valid layout bounds")
    }

    /// Create a new Array from a vector of elements in C-contiguous order.
    pub fn from_vec(shape: impl IntoShape<N>, vec: Vec<T>) -> Result<Self> {
        let shape = shape.into_shape();
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
    pub fn from_shape_vec(shape: impl IntoShape<N>, vec: Vec<T>) -> Result<Self> {
        Self::from_vec(shape, vec)
    }

    /// Create an Array by calling a generator function for each coordinate.
    pub fn from_shape_fn<F>(shape: impl IntoShape<N>, mut f: F) -> Self
    where
        F: FnMut([usize; N]) -> T,
    {
        let shape = shape.into_shape();
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

impl<T> FromIterator<T> for Array<T, VecStorage<T>, 1> {
    #[inline]
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let vec: Vec<T> = iter.into_iter().collect();
        Self::from_vec([vec.len()], vec).expect("1-D iterator length must match collected storage")
    }
}

impl<T> Array<T, VecStorage<T>, 2>
where
    T: eunomia::NumericElement,
{
    /// Create an `n × n` identity matrix.
    pub fn eye(n: usize) -> Self {
        let mut array = Self::zeros([n, n]);
        for i in 0..n {
            array[[i, i]] = T::ONE;
        }
        array
    }

    /// Create an `nrows × ncols` matrix by calling `f(i, j)` for each element.
    pub fn from_fn<F>(nrows: usize, ncols: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        Self::from_shape_fn([nrows, ncols], |idx| f(idx[0], idx[1]))
    }
}

impl<T> Array<eunomia::Complex<T>, VecStorage<eunomia::Complex<T>>, 2>
where
    T: eunomia::FloatElement + core::ops::Neg<Output = T>,
{
    /// Hermitian adjoint (conjugate transpose) of a complex matrix.
    ///
    /// Returns a new owned `cols × rows` matrix where element `[i,j]` equals
    /// `conj(self[[j, i]])`.
    pub fn adjoint(&self) -> Self {
        let [rows, cols] = self.shape();
        let mut data = Vec::with_capacity(rows * cols);
        for j in 0..cols {
            for i in 0..rows {
                data.push(self[[i, j]].conj());
            }
        }
        Self::from_shape_vec([cols, rows], data).expect("adjoint preserves element count")
    }
}

#[cfg(feature = "mnemosyne-alloc")]
impl<T, const N: usize> Array<T, MnemosyneStorage<T>, N> {
    /// Create a Mnemosyne-backed array of a given shape filled with `T::default()`.
    pub fn zeros_mnemosyne(shape: impl IntoShape<N>) -> Self
    where
        T: Default,
    {
        let shape = shape.into_shape();
        let layout = Layout::c_contiguous(shape).expect("C-contiguous layout must construct");
        let storage = MnemosyneStorage::new(layout.size());
        Self::new(layout, storage).expect("valid Mnemosyne storage bounds")
    }

    /// Create a Mnemosyne-backed array from a vector in C-contiguous order.
    pub fn from_mnemosyne_vec(shape: impl IntoShape<N>, vec: Vec<T>) -> Result<Self> {
        let shape = shape.into_shape();
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
    pub fn from_mnemosyne_shape_vec(shape: impl IntoShape<N>, vec: Vec<T>) -> Result<Self> {
        Self::from_mnemosyne_vec(shape, vec)
    }

    /// Create a Mnemosyne-backed array by initializing final storage in place.
    ///
    /// The generator is called once per coordinate in C-contiguous logical
    /// order. No intermediate collection is allocated. If the generator panics,
    /// the initialized prefix is dropped and the Mnemosyne allocation is freed.
    ///
    /// # Panics
    ///
    /// Panics when the shape cannot form a C-contiguous layout, Mnemosyne cannot
    /// allocate the result, or the generator panics.
    pub fn from_mnemosyne_shape_fn<F>(shape: impl IntoShape<N>, mut f: F) -> Self
    where
        F: FnMut([usize; N]) -> T,
    {
        let shape = shape.into_shape();
        let layout = Layout::c_contiguous(shape).expect("C-contiguous layout must construct");
        let mut index = [0usize; N];
        let storage = MnemosyneStorage::from_fn(layout.size(), |_| {
            let value = f(index);
            increment_index(&mut index, &shape);
            value
        });
        Self::new(layout, storage).expect("Valid layout bounds")
    }

    /// Create a Mnemosyne-backed array by copying a C-contiguous source slice.
    pub fn from_mnemosyne_slice(shape: impl IntoShape<N>, slice: &[T]) -> Result<Self>
    where
        T: Copy,
    {
        let shape = shape.into_shape();
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
/// `leto::Array1::from(vec)` / `from_vec`. The vector's storage is moved in
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

    #[test]
    fn array1_collects_from_iterator() {
        let a: Array<i32, VecStorage<i32>, 1> = (2..5).collect();
        assert_eq!(a.shape(), [3]);
        assert_eq!(a.as_slice().unwrap(), &[2, 3, 4]);
    }

    #[test]
    fn array2_eye_creates_identity_matrix() {
        let eye = Array::<f64, VecStorage<f64>, 2>::eye(3);
        assert_eq!(eye.shape(), [3, 3]);
        assert_eq!(eye[[0, 0]], 1.0);
        assert_eq!(eye[[0, 1]], 0.0);
        assert_eq!(eye[[1, 0]], 0.0);
        assert_eq!(eye[[1, 1]], 1.0);
        assert_eq!(eye[[2, 2]], 1.0);
    }

    #[test]
    fn array2_from_fn_creates_matrix_from_closure() {
        let m = Array::<f64, VecStorage<f64>, 2>::from_fn(2, 3, |i, j| (i * 10 + j) as f64);
        assert_eq!(m.shape(), [2, 3]);
        assert_eq!(m[[0, 0]], 0.0);
        assert_eq!(m[[0, 1]], 1.0);
        assert_eq!(m[[1, 0]], 10.0);
        assert_eq!(m[[1, 2]], 12.0);
    }

    #[test]
    fn array2_adjoint_of_complex_matrix() {
        use eunomia::Complex64;
        let m = Array::<Complex64, VecStorage<Complex64>, 2>::from_fn(2, 3, |i, j| {
            Complex64::new((i * 10 + j) as f64, (i * 10 + j + 1) as f64)
        });
        let adj = m.adjoint();
        assert_eq!(adj.shape(), [3, 2]);
        for i in 0..2 {
            for j in 0..3 {
                let expected = m[[i, j]].conj();
                assert_eq!(adj[[j, i]].re, expected.re);
                assert_eq!(adj[[j, i]].im, expected.im);
            }
        }
    }

    #[test]
    fn array2_adjoint_twice_is_original() {
        use eunomia::Complex64;
        let m = Array::<Complex64, VecStorage<Complex64>, 2>::from_fn(3, 4, |i, j| {
            Complex64::new((i * j) as f64, (i + j) as f64)
        });
        let adj = m.adjoint();
        let adjadj = adj.adjoint();
        assert_eq!(adjadj.shape(), [3, 4]);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(adjadj[[i, j]].re, m[[i, j]].re);
                assert_eq!(adjadj[[i, j]].im, m[[i, j]].im);
            }
        }
    }
}
