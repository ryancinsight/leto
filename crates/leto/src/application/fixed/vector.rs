//! Stack-backed fixed-size vector primitive.

use core::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub};

/// Stack-backed fixed-size vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedVector<T, const N: usize> {
    data: [T; N],
}

impl<T, const N: usize> FixedVector<T, N> {
    /// Create a vector from components.
    pub const fn new(data: [T; N]) -> Self {
        Self { data }
    }

    /// Return the vector components.
    pub fn into_array(self) -> [T; N] {
        self.data
    }

    /// Borrow the vector components.
    pub const fn as_array(&self) -> &[T; N] {
        &self.data
    }

    /// Iterate over vector components in index order.
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.data.iter()
    }

    #[inline]
    pub(super) fn for_each_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T, usize),
    {
        for i in 0..N {
            f(&mut self.data[i], i);
        }
    }
}

impl<T, const N: usize> FixedVector<T, N>
where
    T: Copy,
{
    #[inline]
    pub(super) fn map<U, F>(self, mut f: F) -> FixedVector<U, N>
    where
        F: FnMut(T) -> U,
    {
        FixedVector::new(std::array::from_fn(|i| f(self.data[i])))
    }

    #[inline]
    pub(super) fn zip_map<U, F>(self, rhs: Self, mut f: F) -> FixedVector<U, N>
    where
        F: FnMut(T, T) -> U,
    {
        FixedVector::new(std::array::from_fn(|i| f(self.data[i], rhs.data[i])))
    }
}

impl<T, const N: usize> FixedVector<T, N>
where
    T: Copy + Default,
{
    /// Create the zero vector.
    pub fn zeros() -> Self {
        Self {
            data: [T::default(); N],
        }
    }
}

impl<T, const N: usize> FixedVector<T, N>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    /// Dot product with another vector.
    pub fn dot(&self, rhs: &Self) -> T {
        let mut acc = T::default();
        for i in 0..N {
            acc = acc + self.data[i] * rhs.data[i];
        }
        acc
    }
}

impl<T, const N: usize> Index<usize> for FixedVector<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for FixedVector<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T, const N: usize> Add for FixedVector<T, N>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.zip_map(rhs, |lhs, rhs| lhs + rhs)
    }
}

impl<T, const N: usize> AddAssign for FixedVector<T, N>
where
    T: Copy + AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.for_each_mut(|value, i| *value += rhs.data[i]);
    }
}

impl<T, const N: usize> Sub for FixedVector<T, N>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.zip_map(rhs, |lhs, rhs| lhs - rhs)
    }
}

impl<T, const N: usize> Mul<T> for FixedVector<T, N>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        self.map(|value| value * rhs)
    }
}

impl<T, const N: usize> Div<T> for FixedVector<T, N>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        self.map(|value| value / rhs)
    }
}

impl<T, const N: usize> Neg for FixedVector<T, N>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map(|value| -value)
    }
}

#[cfg(test)]
mod tests {
    use super::FixedVector;

    #[test]
    fn fixed_vector_dot_matches_inner_product() {
        let lhs = FixedVector::new([1.0, 2.0, 3.0]);
        let rhs = FixedVector::new([4.0, 5.0, 6.0]);

        assert_eq!(lhs.dot(&rhs), 32.0);
    }

    #[test]
    fn fixed_vector_iterates_in_index_order() {
        let vector = FixedVector::new([1.0, 2.0, 3.0]);

        assert_eq!(
            vector.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0]
        );
    }
}
