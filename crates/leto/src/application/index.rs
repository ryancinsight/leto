/// Convert a flat row-major logical index into an N-dimensional index.
#[inline(always)]
pub(crate) fn index_from_flat<const N: usize>(flat: usize, shape: &[usize; N]) -> [usize; N] {
    let mut index = [0usize; N];
    crate::domain::layout::kernels::fill_index_from_flat(flat, shape, &mut index);
    index
}

use crate::application::array::Array;
use crate::infrastructure::storage::{Storage, StorageMut};
use core::ops::{Index, IndexMut};

/// `arr[[i, j, …]]` element access by N-dimensional index (ndarray parity).
///
/// Delegates to [`Array::get`](crate::application::array::Array::get); panics on
/// an out-of-bounds index, per the [`Index`] contract.
impl<T, S: Storage<T>, const N: usize> Index<[usize; N]> for Array<T, S, N> {
    type Output = T;
    #[inline]
    fn index(&self, index: [usize; N]) -> &T {
        self.get(index)
            .unwrap_or_else(|_| panic!("index {index:?} out of bounds"))
    }
}

/// Mutable `arr[[i, j, …]]` element access (ndarray parity).
impl<T, S: StorageMut<T>, const N: usize> IndexMut<[usize; N]> for Array<T, S, N> {
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut T {
        self.get_mut(index)
            .unwrap_or_else(|_| panic!("index {index:?} out of bounds"))
    }
}

#[cfg(test)]
mod tests {
    use crate::application::array::Array;
    use crate::infrastructure::storage::VecStorage;

    #[test]
    fn index_and_index_mut_by_array() {
        let mut a =
            Array::<f64, VecStorage<f64>, 2>::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        assert_eq!(a[[0, 0]], 1.0);
        assert_eq!(a[[1, 2]], 6.0);
        a[[1, 1]] = 50.0;
        assert_eq!(a[[1, 1]], 50.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn index_out_of_bounds_panics() {
        let a = Array::<f64, VecStorage<f64>, 1>::from_shape_vec([2], vec![1.0, 2.0]).unwrap();
        let _ = a[[5]];
    }
}
