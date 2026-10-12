use super::super::vector::FixedVector;

/// Stack-backed row-major fixed-size matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedMatrix<T, const R: usize, const C: usize> {
    pub(super) data: [[T; C]; R],
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

    /// Iterate over matrix entries in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter().flat_map(|row| row.iter())
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
