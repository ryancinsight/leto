//! Stack-backed row-major fixed-size matrix primitive.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::vector::FixedVector;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

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

    /// Iterate over matrix entries in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter().flat_map(|row| row.iter())
    }

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

// --- 4×4 matrix methods -----------------------------------------------------

impl<T> FixedMatrix<T, 4, 4>
where
    T: Copy
        + Default
        + From<u8>
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialEq
        + Neg<Output = T>,
{
    /// Create a 4×4 matrix from row-major storage.
    pub fn from_row_major(data: [T; 16]) -> Self {
        Self::from_rows([
            [data[0], data[1], data[2], data[3]],
            [data[4], data[5], data[6], data[7]],
            [data[8], data[9], data[10], data[11]],
            [data[12], data[13], data[14], data[15]],
        ])
    }

    /// Create a 4×4 matrix from column-major storage.
    pub fn from_column_major(data: [T; 16]) -> Self {
        Self::from_rows([
            [data[0], data[4], data[8], data[12]],
            [data[1], data[5], data[9], data[13]],
            [data[2], data[6], data[10], data[14]],
            [data[3], data[7], data[11], data[15]],
        ])
    }

    /// Return the matrix entries in row-major order.
    pub fn into_row_major(self) -> [T; 16] {
        [
            self[(0, 0)],
            self[(0, 1)],
            self[(0, 2)],
            self[(0, 3)],
            self[(1, 0)],
            self[(1, 1)],
            self[(1, 2)],
            self[(1, 3)],
            self[(2, 0)],
            self[(2, 1)],
            self[(2, 2)],
            self[(2, 3)],
            self[(3, 0)],
            self[(3, 1)],
            self[(3, 2)],
            self[(3, 3)],
        ]
    }

    /// Return the matrix entries in column-major order.
    pub fn into_column_major(self) -> [T; 16] {
        [
            self[(0, 0)],
            self[(1, 0)],
            self[(2, 0)],
            self[(3, 0)],
            self[(0, 1)],
            self[(1, 1)],
            self[(2, 1)],
            self[(3, 1)],
            self[(0, 2)],
            self[(1, 2)],
            self[(2, 2)],
            self[(3, 2)],
            self[(0, 3)],
            self[(1, 3)],
            self[(2, 3)],
            self[(3, 3)],
        ]
    }

    /// Determinant using cofactor expansion along the first row.
    pub fn determinant(&self) -> T {
        let a = self[(0, 0)];
        let b = self[(0, 1)];
        let c = self[(0, 2)];
        let d = self[(0, 3)];
        a * subdet_3x3(self, 0, 0) - b * subdet_3x3(self, 0, 1) + c * subdet_3x3(self, 0, 2)
            - d * subdet_3x3(self, 0, 3)
    }

    /// Inverse using the adjugate (cofactor matrix transpose) divided by the
    /// determinant. Returns `None` when the matrix is singular.
    pub fn try_inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det == T::default() {
            return None;
        }
        let inv_det = T::from(1) / det;
        Some(Self::from_rows([
            [
                cofactor_4x4(self, 0, 0) * inv_det,
                cofactor_4x4(self, 1, 0) * inv_det,
                cofactor_4x4(self, 2, 0) * inv_det,
                cofactor_4x4(self, 3, 0) * inv_det,
            ],
            [
                cofactor_4x4(self, 0, 1) * inv_det,
                cofactor_4x4(self, 1, 1) * inv_det,
                cofactor_4x4(self, 2, 1) * inv_det,
                cofactor_4x4(self, 3, 1) * inv_det,
            ],
            [
                cofactor_4x4(self, 0, 2) * inv_det,
                cofactor_4x4(self, 1, 2) * inv_det,
                cofactor_4x4(self, 2, 2) * inv_det,
                cofactor_4x4(self, 3, 2) * inv_det,
            ],
            [
                cofactor_4x4(self, 0, 3) * inv_det,
                cofactor_4x4(self, 1, 3) * inv_det,
                cofactor_4x4(self, 2, 3) * inv_det,
                cofactor_4x4(self, 3, 3) * inv_det,
            ],
        ]))
    }
}

// --- 4×4 helper functions ---------------------------------------------------

/// Determinant of the 3×3 submatrix obtained by removing `exclude_row` and
/// `exclude_col` from the 4×4 matrix.
fn subdet_3x3<T>(m: &FixedMatrix<T, 4, 4>, exclude_row: usize, exclude_col: usize) -> T
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    // Collect the 3×3 submatrix entries
    let mut e = [[T::default(); 3]; 3];
    let mut ri = 0;
    for r in 0..4 {
        if r == exclude_row {
            continue;
        }
        let mut ci = 0;
        for c in 0..4 {
            if c == exclude_col {
                continue;
            }
            e[ri][ci] = m[(r, c)];
            ci += 1;
        }
        ri += 1;
    }
    // Sarrus rule for 3×3 determinant
    let a = e[0][0];
    let b = e[0][1];
    let c_ = e[0][2];
    let d = e[1][0];
    let e_ = e[1][1];
    let f = e[1][2];
    let g = e[2][0];
    let h = e[2][1];
    let i = e[2][2];
    a * (e_ * i - f * h) - b * (d * i - f * g) + c_ * (d * h - e_ * g)
}

/// Cofactor of a 4×4 matrix entry: (-1)^(row+col) * det(minor).
fn cofactor_4x4<T>(m: &FixedMatrix<T, 4, 4>, row: usize, col: usize) -> T
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>,
{
    let det = subdet_3x3(m, row, col);
    match (row + col) % 2 {
        0 => det,
        _ => -det,
    }
}

#[cfg(test)]
mod tests {
    use super::FixedMatrix;

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
    fn fixed_3x3_inverse_matches_known_value() {
        let matrix = FixedMatrix::from_rows([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]);
        let inv = matrix.try_inverse().unwrap();
        let expected =
            FixedMatrix::from_rows([[-24.0, 18.0, 5.0], [20.0, -15.0, -4.0], [-5.0, 4.0, 1.0]]);
        assert_eq!(inv, expected);
    }

    #[test]
    fn fixed_3x3_inverse_times_original_is_identity() {
        let m = FixedMatrix::from_rows([[4.0, 7.0, 2.0], [2.0, 6.0, 1.0], [3.0, 5.0, 8.0]]);
        let inv = m.try_inverse().unwrap();
        let product = m * inv;
        let identity: FixedMatrix<f64, 3, 3> = FixedMatrix::identity();
        for row in 0..3 {
            for col in 0..3 {
                assert!((product[(row, col)] - identity[(row, col)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn fixed_3x3_inverse_returns_none_for_singular() {
        let singular = FixedMatrix::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        assert!(singular.try_inverse().is_none());
    }

    #[test]
    fn fixed_3x3_inverse_identity() {
        let identity = FixedMatrix::<f64, 3, 3>::identity();
        let inv = identity.try_inverse().unwrap();
        for row in 0..3 {
            for col in 0..3 {
                assert!((inv[(row, col)] - identity[(row, col)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn fixed_2x2_inverse_matches_known_value() {
        let m = FixedMatrix::from_rows([[1.0, 2.0], [3.0, 4.0]]);
        let inv = m.try_inverse().unwrap();
        let expected = FixedMatrix::from_rows([[-2.0, 1.0], [1.5, -0.5]]);
        for row in 0..2 {
            for col in 0..2 {
                assert!((inv[(row, col)] - expected[(row, col)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn fixed_2x2_inverse_times_original_is_identity() {
        let m = FixedMatrix::from_rows([[5.0, 3.0], [2.0, 1.0]]);
        let inv = m.try_inverse().unwrap();
        let product = m * inv;
        let identity = FixedMatrix::<f64, 2, 2>::identity();
        for row in 0..2 {
            for col in 0..2 {
                assert!((product[(row, col)] - identity[(row, col)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn fixed_2x2_inverse_returns_none_for_singular() {
        let singular = FixedMatrix::from_rows([[1.0, 2.0], [2.0, 4.0]]);
        assert!(singular.try_inverse().is_none());
    }

    #[test]
    fn fixed_matrix_iterates_in_row_major_order() {
        let matrix = FixedMatrix::from_rows([[1.0, 2.0], [3.0, 4.0]]);

        assert_eq!(
            matrix.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn fixed_matrix_converts_row_and_column_major_storage() {
        let row_major = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let column_major = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

        let matrix = FixedMatrix::<f64, 3, 3>::from_row_major(row_major);

        assert_eq!(matrix.into_row_major(), row_major);
        assert_eq!(matrix.into_column_major(), column_major);
        assert_eq!(
            FixedMatrix::<f64, 3, 3>::from_column_major(column_major),
            matrix
        );
    }

    #[test]
    fn fixed_4x4_diagonal_inverse_preserves_value_contract() {
        let matrix = FixedMatrix::<f64, 4, 4>::from_row_major([
            1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        assert_eq!(matrix.determinant(), 8.0);
        assert_eq!(
            matrix.into_column_major(),
            [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );

        let inverse = matrix
            .try_inverse()
            .expect("diagonal matrix is nonsingular");
        let expected = FixedMatrix::<f64, 4, 4>::from_row_major([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        assert_eq!(inverse, expected);
        assert_eq!(
            FixedMatrix::<f64, 4, 4>::from_row_major([
                1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ])
            .try_inverse(),
            None
        );
    }
}
