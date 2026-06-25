//! Stack-backed fixed-size vector and matrix primitives.
//!
//! These types cover small linear-algebra values where heap-backed strided
//! arrays would add avoidable allocation and layout metadata. They are plain
//! row-major array wrappers, so indexing and arithmetic stay stack-local.

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
        Self::new(std::array::from_fn(|i| self.data[i] + rhs.data[i]))
    }
}

impl<T, const N: usize> AddAssign for FixedVector<T, N>
where
    T: Copy + AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.data[i] += rhs.data[i];
        }
    }
}

impl<T, const N: usize> Sub for FixedVector<T, N>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(std::array::from_fn(|i| self.data[i] - rhs.data[i]))
    }
}

impl<T, const N: usize> Mul<T> for FixedVector<T, N>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::new(std::array::from_fn(|i| self.data[i] * rhs))
    }
}

impl<T, const N: usize> Div<T> for FixedVector<T, N>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self::new(std::array::from_fn(|i| self.data[i] / rhs))
    }
}

impl<T, const N: usize> Neg for FixedVector<T, N>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(std::array::from_fn(|i| -self.data[i]))
    }
}

/// Stack-backed row-major fixed-size matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedMatrix<T, const R: usize, const C: usize> {
    data: [[T; C]; R],
}

impl<T, const R: usize, const C: usize> FixedMatrix<T, R, C> {
    /// Create a row-major matrix from rows.
    pub const fn from_rows(data: [[T; C]; R]) -> Self {
        Self { data }
    }

    /// Return the row-major matrix storage.
    pub fn into_rows(self) -> [[T; C]; R] {
        self.data
    }

    /// Borrow the row-major matrix storage.
    pub const fn rows(&self) -> &[[T; C]; R] {
        &self.data
    }
}

impl<T, const R: usize, const C: usize> FixedMatrix<T, R, C>
where
    T: Copy + Default,
{
    /// Create a zero matrix.
    pub fn zeros() -> Self {
        Self {
            data: [[T::default(); C]; R],
        }
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> FixedMatrix<T, C, R> {
        FixedMatrix::from_rows(std::array::from_fn(|row| {
            std::array::from_fn(|col| self.data[col][row])
        }))
    }

    /// Create a matrix from column vectors.
    pub fn from_columns(columns: [FixedVector<T, R>; C]) -> Self {
        Self::from_rows(std::array::from_fn(|row| {
            std::array::from_fn(|col| columns[col][row])
        }))
    }

    /// Replace one matrix column.
    pub fn set_column(&mut self, column: usize, values: FixedVector<T, R>) {
        for row in 0..R {
            self.data[row][column] = values[row];
        }
    }
}

impl<T, const N: usize> FixedMatrix<T, N, N>
where
    T: Copy + Default + From<u8>,
{
    /// Create an identity matrix.
    pub fn identity() -> Self {
        let mut matrix = Self::zeros();
        for i in 0..N {
            matrix[(i, i)] = T::from(1);
        }
        matrix
    }
}

impl FixedMatrix<f64, 3, 3> {
    /// Determinant of a 3x3 matrix.
    pub fn determinant(&self) -> f64 {
        let a = self[(0, 0)];
        let b = self[(0, 1)];
        let c = self[(0, 2)];
        let d = self[(1, 0)];
        let e = self[(1, 1)];
        let f = self[(1, 2)];
        let g = self[(2, 0)];
        let h = self[(2, 1)];
        let i = self[(2, 2)];

        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }
}

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
        for row in 0..R {
            for col in 0..C {
                self.data[row][col] += rhs.data[row][col];
            }
        }
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

#[cfg(test)]
mod tests {
    use super::{FixedMatrix, FixedVector};

    #[test]
    fn fixed_matrix_multiplies_on_stack() {
        let lhs = FixedMatrix::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let rhs = FixedMatrix::from_rows([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);

        let product = lhs * rhs;

        assert_eq!(
            product,
            FixedMatrix::from_rows([[58.0, 64.0], [139.0, 154.0]])
        );
    }

    #[test]
    fn fixed_matrix_determinant_matches_known_value() {
        let matrix = FixedMatrix::from_rows([[6.0, 1.0, 1.0], [4.0, -2.0, 5.0], [2.0, 8.0, 7.0]]);

        assert_eq!(matrix.determinant(), -306.0);
    }

    #[test]
    fn fixed_vector_dot_matches_inner_product() {
        let lhs = FixedVector::new([1.0, 2.0, 3.0]);
        let rhs = FixedVector::new([4.0, 5.0, 6.0]);

        assert_eq!(lhs.dot(&rhs), 32.0);
    }
}
