//! Element-wise transforms over an [`Array`]: by-value map to a new array
//! ([`mapv`](Array::mapv)) and reduction ([`fold`](Array::fold)), matching the
//! `ndarray` methods of the same name.
//!
//! Both iterate in logical row-major order, so they are correct for any
//! layout (a C-contiguous array or an arbitrarily strided view alike), and
//! `mapv` always yields a fresh C-contiguous [`Array`]. Generic and fully
//! monomorphized — the supplied closure is the only cost over a hand-written
//! loop.

use crate::application::array::Array;
use crate::infrastructure::storage::{Storage, VecStorage};

impl<T, S, const N: usize> Array<T, S, N>
where
    S: Storage<T>,
    T: Copy,
{
    /// Apply `f` to each element **by value**, returning a new owned array of
    /// the same shape (ndarray `mapv` parity). The result is C-contiguous.
    #[must_use = "mapv returns a new array; use mapv_inplace to mutate in place"]
    pub fn mapv<U, F>(&self, mut f: F) -> Array<U, VecStorage<U>, N>
    where
        F: FnMut(T) -> U,
    {
        let data: Vec<U> = self.iter().map(|&x| f(x)).collect();
        Array::<U, VecStorage<U>, N>::from_shape_vec(self.shape(), data)
            .expect("invariant: mapv preserves the element count and shape")
    }

    /// Fold the elements in logical row-major order, threading `init` through
    /// `f` (ndarray `fold` parity).
    pub fn fold<B, F>(&self, init: B, mut f: F) -> B
    where
        F: FnMut(B, T) -> B,
    {
        self.iter().fold(init, |acc, &x| f(acc, x))
    }
}

#[cfg(test)]
mod tests {
    use crate::application::array::Array;
    use crate::infrastructure::storage::VecStorage;

    fn arr(shape: [usize; 2], data: Vec<f64>) -> Array<f64, VecStorage<f64>, 2> {
        Array::<f64, VecStorage<f64>, 2>::from_shape_vec(shape, data).unwrap()
    }

    #[test]
    fn mapv_applies_elementwise_and_preserves_shape() {
        let a = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.mapv(|x| x * x);
        assert_eq!(b.shape(), [2, 3]);
        assert_eq!(b.iter().copied().collect::<Vec<_>>(), vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
    }

    #[test]
    fn mapv_can_change_element_type() {
        let a = arr([1, 3], vec![0.4, 1.6, 2.5]);
        let b: Array<i64, VecStorage<i64>, 2> = a.mapv(|x| x as i64);
        assert_eq!(b.iter().copied().collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn fold_reduces_in_row_major_order() {
        let a = arr([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.fold(0.0, |acc, x| acc + x), 10.0);
        // order-sensitive fold: subtract in row-major order 0-1-2-3-4 => -10
        assert_eq!(a.fold(0.0, |acc, x| acc - x), -10.0);
    }
}
