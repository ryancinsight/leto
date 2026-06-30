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
use crate::application::view::ArrayView;
use crate::infrastructure::storage::{Storage, StorageMut, VecStorage};

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

    /// Collect the elements into a `Vec` in logical row-major order (ndarray
    /// `to_vec` parity); correct for any strides.
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        self.iter().copied().collect()
    }

    /// Fold the elements in logical row-major order, threading `init` through
    /// `f` (ndarray `fold` parity).
    pub fn fold<B, F>(&self, init: B, mut f: F) -> B
    where
        F: FnMut(B, T) -> B,
    {
        self.iter().fold(init, |acc, &x| f(acc, x))
    }

    /// Apply `f` to each pair of co-located elements **by value**, returning a
    /// new owned array of the same shape (ndarray `Zip::from(a).and(b).
    /// map_collect(f)` parity). The result is C-contiguous.
    ///
    /// Both arrays are traversed in logical row-major order, so the pairing is
    /// correct regardless of either side's strides. Fully monomorphized — the
    /// supplied closure is the only cost over a hand-written zipped loop.
    ///
    /// # Panics
    /// If `self.shape() != other.shape()`.
    #[must_use]
    pub fn zip_map<U, S2, V, F>(&self, other: &Array<U, S2, N>, mut f: F) -> Array<V, VecStorage<V>, N>
    where
        U: Copy,
        S2: Storage<U>,
        F: FnMut(T, U) -> V,
    {
        assert_eq!(
            self.shape(),
            other.shape(),
            "zip_map requires matching shapes: self {:?} vs other {:?}",
            self.shape(),
            other.shape()
        );
        let data: Vec<V> = self.iter().zip(other.iter()).map(|(&a, &b)| f(a, b)).collect();
        Array::<V, VecStorage<V>, N>::from_shape_vec(self.shape(), data)
            .expect("invariant: zip_map preserves the element count and shape")
    }
}

impl<T, S, const N: usize> Array<T, S, N>
where
    S: StorageMut<T>,
    T: Copy,
{
    /// Replace each element with `f(element)` **in place** (ndarray
    /// `mapv_inplace` parity).
    ///
    /// A C-contiguous array (the result of every owned-array constructor) takes
    /// the contiguous fast path; an arbitrarily strided array falls back to a
    /// logical-order walk that visits each element's physical offset exactly
    /// once, so the result is correct for any layout.
    pub fn mapv_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(T) -> T,
    {
        let layout = self.layout;
        let size = layout.size();
        if size == 0 {
            return;
        }
        if layout.is_c_dense() {
            let start = layout.offset;
            for slot in &mut self.storage.as_mut_slice()[start..start + size] {
                *slot = f(*slot);
            }
        } else {
            let shape = layout.shape;
            let slice = self.storage.as_mut_slice();
            let mut index = [0usize; N];
            for _ in 0..size {
                let off = layout
                    .offset_of(index)
                    .expect("invariant: logical index is in bounds");
                slice[off] = f(slice[off]);
                // row-major odometer increment of the multi-index.
                for d in (0..N).rev() {
                    index[d] += 1;
                    if index[d] < shape[d] {
                        break;
                    }
                    index[d] = 0;
                }
            }
        }
    }

    /// Copy every element of `src` into `self` in logical row-major order
    /// (ndarray `assign` parity). Both arrays are traversed logically, so the
    /// copy is correct regardless of either side's strides.
    ///
    /// # Panics
    /// If `self.shape() != src.shape()`.
    pub fn assign(&mut self, src: &ArrayView<'_, T, N>) {
        assert_eq!(
            self.shape(),
            src.shape(),
            "assign requires matching shapes: self {:?} vs src {:?}",
            self.shape(),
            src.shape()
        );
        let mut it = src.iter();
        self.mapv_inplace(|_| *it.next().expect("invariant: src has self.size() elements"));
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

    #[test]
    fn assign_copies_source_elementwise() {
        let mut dst = arr([2, 3], vec![0.0; 6]);
        let src = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        dst.assign(&src.view());
        assert_eq!(
            dst.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    #[should_panic(expected = "matching shapes")]
    fn assign_shape_mismatch_panics() {
        let mut dst = arr([2, 2], vec![0.0; 4]);
        let src = arr([2, 3], vec![0.0; 6]);
        dst.assign(&src.view());
    }

    #[test]
    fn zip_map_pairs_colocated_elements() {
        let a = arr([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = arr([2, 2], vec![10.0, 20.0, 30.0, 40.0]);
        let c = a.zip_map(&b, |x, y| x + y);
        assert_eq!(c.shape(), [2, 2]);
        assert_eq!(c.iter().copied().collect::<Vec<_>>(), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    #[should_panic(expected = "matching shapes")]
    fn zip_map_shape_mismatch_panics() {
        let a = arr([2, 2], vec![0.0; 4]);
        let b = arr([2, 3], vec![0.0; 6]);
        let _ = a.zip_map(&b, |x, y| x + y);
    }

    #[test]
    fn mapv_inplace_mutates_and_matches_mapv() {
        let mut a = arr([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let expected = a.mapv(|x| 2.0 * x + 1.0);
        a.mapv_inplace(|x| 2.0 * x + 1.0);
        assert_eq!(
            a.iter().copied().collect::<Vec<_>>(),
            expected.iter().copied().collect::<Vec<_>>()
        );
        assert_eq!(a.shape(), [2, 3]);
    }
}
