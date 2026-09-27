use super::types::Array;
use crate::infrastructure::storage::{Storage, StorageMut};

impl<T, S, const N: usize> std::ops::Index<[usize; N]> for Array<T, S, N>
where
    S: Storage<T>,
{
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        self.get(index)
            .expect("invariant: array index is within shape and storage bounds")
    }
}

impl<T, S> std::ops::Index<usize> for Array<T, S, 1>
where
    S: Storage<T>,
{
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.get([index])
            .expect("invariant: array index is within shape and storage bounds")
    }
}

impl<T, S, const N: usize> std::ops::IndexMut<[usize; N]> for Array<T, S, N>
where
    S: StorageMut<T>,
{
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        self.get_mut(index)
            .expect("invariant: array index is within shape and storage bounds")
    }
}

impl<T, S> std::ops::IndexMut<usize> for Array<T, S, 1>
where
    S: StorageMut<T>,
{
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut([index])
            .expect("invariant: array index is within shape and storage bounds")
    }
}
