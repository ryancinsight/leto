use super::super::vector::FixedVector;
use super::FixedMatrix;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

impl<T, const R: usize, const C: usize> FixedMatrix<T, R, C> {
    #[inline]
    fn for_each_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T, usize, usize),
    {
        for row in 0..R {
            for col in 0..C {
                f(&mut self.data[row][col], row, col);
            }
        }
    }
}

impl<T, const R: usize, const C: usize> FixedMatrix<T, R, C>
where
    T: Copy,
{
    #[inline]
    fn map<U, F>(self, mut f: F) -> FixedMatrix<U, R, C>
    where
        F: FnMut(T) -> U,
    {
        FixedMatrix::from_rows(std::array::from_fn(|row| {
            std::array::from_fn(|col| f(self.data[row][col]))
        }))
    }

    #[inline]
    fn zip_map<U, F>(self, rhs: Self, mut f: F) -> FixedMatrix<U, R, C>
    where
        F: FnMut(T, T) -> U,
    {
        FixedMatrix::from_rows(std::array::from_fn(|row| {
            std::array::from_fn(|col| f(self.data[row][col], rhs.data[row][col]))
        }))
    }
}

// ----------------------------------------------------------------------------

impl<T, const R: usize, const C: usize> Index<(usize, usize)> for FixedMatrix<T, R, C> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0][index.1]
    }
}

impl<T, const R: usize, const C: usize> IndexMut<(usize, usize)> for FixedMatrix<T, R, C> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0][index.1]
    }
}

impl<T, const R: usize, const C: usize> AddAssign for FixedMatrix<T, R, C>
where
    T: Copy + AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.for_each_mut(|value, row, col| *value += rhs.data[row][col]);
    }
}

impl<T, const R: usize, const K: usize, const C: usize> Mul<FixedMatrix<T, K, C>>
    for FixedMatrix<T, R, K>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    type Output = FixedMatrix<T, R, C>;

    fn mul(self, rhs: FixedMatrix<T, K, C>) -> Self::Output {
        FixedMatrix::from_rows(std::array::from_fn(|row| {
            std::array::from_fn(|col| {
                let mut acc = T::default();
                for k in 0..K {
                    acc = acc + self[(row, k)] * rhs[(k, col)];
                }
                acc
            })
        }))
    }
}

impl<T, const R: usize, const C: usize> Mul<FixedVector<T, C>> for FixedMatrix<T, R, C>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    type Output = FixedVector<T, R>;

    fn mul(self, rhs: FixedVector<T, C>) -> Self::Output {
        FixedVector::new(std::array::from_fn(|row| {
            let mut acc = T::default();
            for col in 0..C {
                acc = acc + self[(row, col)] * rhs[col];
            }
            acc
        }))
    }
}

// --- Generic FixedMatrix operators ------------------------------------------

impl<T, const R: usize, const C: usize> Add for FixedMatrix<T, R, C>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.zip_map(rhs, |lhs, rhs| lhs + rhs)
    }
}

impl<T, const R: usize, const C: usize> Sub for FixedMatrix<T, R, C>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.zip_map(rhs, |lhs, rhs| lhs - rhs)
    }
}

impl<T, const R: usize, const C: usize> Neg for FixedMatrix<T, R, C>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map(|value| -value)
    }
}

impl<T, const R: usize, const C: usize> Mul<T> for FixedMatrix<T, R, C>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        self.map(|value| value * rhs)
    }
}

impl<T, const R: usize, const C: usize> Div<T> for FixedMatrix<T, R, C>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        self.map(|value| value / rhs)
    }
}

impl<T, const R: usize, const C: usize> SubAssign for FixedMatrix<T, R, C>
where
    T: Copy + SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.for_each_mut(|value, row, col| *value -= rhs.data[row][col]);
    }
}

impl<T, const R: usize, const C: usize> MulAssign<T> for FixedMatrix<T, R, C>
where
    T: Copy + MulAssign,
{
    fn mul_assign(&mut self, rhs: T) {
        self.for_each_mut(|value, _, _| *value *= rhs);
    }
}

impl<T, const R: usize, const C: usize> DivAssign<T> for FixedMatrix<T, R, C>
where
    T: Copy + DivAssign,
{
    fn div_assign(&mut self, rhs: T) {
        self.for_each_mut(|value, _, _| *value /= rhs);
    }
}
