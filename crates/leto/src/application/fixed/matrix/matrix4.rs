use super::FixedMatrix;
use core::ops::{Add, Div, Mul, Neg, Sub};

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
