#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::types::Array;
use crate::infrastructure::storage::Storage;

impl<T, S, const N: usize> Eq for Array<T, S, N>
where
    T: Eq,
    S: Storage<T>,
{
}

/// Logical element-wise equality: two arrays are equal iff they have the same
/// shape and the same elements in logical row-major order (leto parity).
/// Correct across differing strides/storage, unlike a derived `PartialEq`.
impl<T, S1, S2, const N: usize> PartialEq<Array<T, S2, N>> for Array<T, S1, N>
where
    T: PartialEq,
    S1: Storage<T>,
    S2: Storage<T>,
{
    #[inline]
    fn eq(&self, other: &Array<T, S2, N>) -> bool {
        self.shape() == other.shape() && self.iter().eq(other.iter())
    }
}

#[cfg(test)]
mod partial_eq_tests {
    use crate::application::array::Array;
    use crate::infrastructure::storage::VecStorage;

    #[test]
    fn equal_iff_same_shape_and_elements() {
        let a = Array::<f64, VecStorage<f64>, 2>::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0])
            .unwrap();
        let b = Array::<f64, VecStorage<f64>, 2>::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0])
            .unwrap();
        let c = Array::<f64, VecStorage<f64>, 2>::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 9.0])
            .unwrap();
        let d = Array::<f64, VecStorage<f64>, 2>::from_shape_vec([4, 1], vec![1.0, 2.0, 3.0, 4.0])
            .unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d); // same data, different shape
    }
}
