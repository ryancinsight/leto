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

    /// Iterate over vector components in index order.
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.data.iter()
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

impl FixedMatrix<f64, 3, 3> {
    /// Create a 3x3 matrix from row-major storage.
    pub const fn from_row_major(data: [f64; 9]) -> Self {
        Self::from_rows([
            [data[0], data[1], data[2]],
            [data[3], data[4], data[5]],
            [data[6], data[7], data[8]],
        ])
    }

    /// Create a 3x3 matrix from column-major storage.
    pub const fn from_column_major(data: [f64; 9]) -> Self {
        Self::from_rows([
            [data[0], data[3], data[6]],
            [data[1], data[4], data[7]],
            [data[2], data[5], data[8]],
        ])
    }

    /// Return the matrix entries in row-major order.
    pub fn into_row_major(self) -> [f64; 9] {
        [
            self[(0, 0)],
            self[(0, 1)],
            self[(0, 2)],
            self[(1, 0)],
            self[(1, 1)],
            self[(1, 2)],
            self[(2, 0)],
            self[(2, 1)],
            self[(2, 2)],
        ]
    }

    /// Return the matrix entries in column-major order.
    pub fn into_column_major(self) -> [f64; 9] {
        [
            self[(0, 0)],
            self[(1, 0)],
            self[(2, 0)],
            self[(0, 1)],
            self[(1, 1)],
            self[(2, 1)],
            self[(0, 2)],
            self[(1, 2)],
            self[(2, 2)],
        ]
    }

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

    /// Eigendecomposition of a real symmetric 3x3 matrix.
    ///
    /// Uses the analytic cubic formula (trigonometric solution of the depressed
    /// cubic) for eigenvalues and the cross-product of `(A - λI)` columns for
    /// eigenvectors. Eigenvalues are sorted descending; eigenvectors are
    /// orthonormal columns of the returned matrix.
    pub fn symmetric_eigen(&self) -> (FixedVector<f64, 3>, Self) {
        let a = self[(0, 0)];
        let b = self[(0, 1)];
        let c = self[(0, 2)];
        let _d = self[(1, 0)];
        let e = self[(1, 1)];
        let f = self[(1, 2)];
        let _g = self[(2, 0)];
        let _h = self[(2, 1)];
        let i = self[(2, 2)];

        // Characteristic polynomial: λ³ - I₁λ² + I₂λ - I₃ = 0
        let i1 = a + e + i;
        let i2 = a * e + a * i + e * i - b * b - c * c - f * f;
        let i3 = a * e * i + 2.0 * b * c * f - a * f * f - e * c * c - i * b * b;

        // Depressed cubic: μ³ + pμ + q = 0,  λ = μ + I₁/3
        let p = i2 - i1 * i1 / 3.0;
        let q = (2.0 * i1 * i1 * i1 - 9.0 * i1 * i2 + 27.0 * i3) / 27.0;
        let shift = i1 / 3.0;

        if p.abs() < 1e-30 {
            // Isotropic / nearly-equal eigenvalues: identity eigenvectors
            let val = shift;
            return (FixedVector::new([val, val, val]), Self::identity());
        }

        // Trigonometric solution:  μₖ = 2√(-p/3) cos(θ/3 + 2πk/3)
        let sqrt_neg_p3 = (-p / 3.0).sqrt();
        let r = 2.0 * sqrt_neg_p3;
        let cos_arg = (-q / 2.0) / (sqrt_neg_p3 * sqrt_neg_p3 * sqrt_neg_p3);
        let theta = cos_arg.clamp(-1.0, 1.0).acos() / 3.0;

        let two_pi_3 = 2.0943951023931953;
        let four_pi_3 = 4.1887902047863905;

        // Unsorted:  k=0 (θ) largest,  k=2 (θ+4π/3) middle,  k=1 (θ+2π/3) smallest
        let mut vals = [
            r * theta.cos() + shift,
            r * (theta + two_pi_3).cos() + shift,
            r * (theta + four_pi_3).cos() + shift,
        ];

        // Sort descending
        if vals[0] < vals[1] {
            vals.swap(0, 1);
        }
        if vals[0] < vals[2] {
            vals.swap(0, 2);
        }
        if vals[1] < vals[2] {
            vals.swap(1, 2);
        }

        let ev0 = eigenvector_3x3(a, b, c, e, f, i, vals[0]);
        let mut ev1 = eigenvector_3x3(a, b, c, e, f, i, vals[1]);

        // Orthogonalize ev1 against ev0 (the most well-conditioned direction)
        let dot01 = ev0[0] * ev1[0] + ev0[1] * ev1[1] + ev0[2] * ev1[2];
        ev1 = [
            ev1[0] - dot01 * ev0[0],
            ev1[1] - dot01 * ev0[1],
            ev1[2] - dot01 * ev0[2],
        ];
        normalize3(&mut ev1);

        // Third eigenvector = cross of first two → guaranteed orthonormal + right-handed
        let ev2 = cross3(ev0, ev1);

        let eigenvectors = Self::from_rows([
            [ev0[0], ev1[0], ev2[0]],
            [ev0[1], ev1[1], ev2[1]],
            [ev0[2], ev1[2], ev2[2]],
        ]);

        (FixedVector::new(vals), eigenvectors)
    }

    /// Inverse of a 3x3 matrix using the analytic cofactor formula.
    ///
    /// Returns `None` when the matrix is singular (determinant ≈ 0).
    pub fn try_inverse(&self) -> Option<Self> {
        let a = self[(0, 0)];
        let b = self[(0, 1)];
        let c = self[(0, 2)];
        let d = self[(1, 0)];
        let e = self[(1, 1)];
        let f = self[(1, 2)];
        let g = self[(2, 0)];
        let h = self[(2, 1)];
        let i = self[(2, 2)];

        let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        if det == 0.0 {
            return None;
        }

        let inv_det = 1.0 / det;
        Some(Self::from_rows([
            [
                (e * i - f * h) * inv_det,
                (c * h - b * i) * inv_det,
                (b * f - c * e) * inv_det,
            ],
            [
                (f * g - d * i) * inv_det,
                (a * i - c * g) * inv_det,
                (c * d - a * f) * inv_det,
            ],
            [
                (d * h - e * g) * inv_det,
                (b * g - a * h) * inv_det,
                (a * e - b * d) * inv_det,
            ],
        ]))
    }
}

impl FixedMatrix<f64, 2, 2> {
    /// Determinant of a 2x2 matrix.
    pub fn determinant(&self) -> f64 {
        self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]
    }

    /// Eigendecomposition of a real symmetric 2x2 matrix.
    ///
    /// Uses the closed-form quadratic formula. Eigenvalues are sorted
    /// descending; eigenvectors are orthonormal columns.
    pub fn symmetric_eigen(&self) -> (FixedVector<f64, 2>, Self) {
        let a = self[(0, 0)];
        let b = self[(0, 1)];
        let _c = self[(1, 0)];
        let d = self[(1, 1)];

        // Eigenvalues of [a b; b d]:  λ = ½[(a+d) ± √((a-d)² + 4b²)]
        let trace = a + d;
        let disc = ((a - d) * (a - d) + 4.0 * b * b).sqrt();
        let lambda1 = (trace + disc) / 2.0;
        let lambda2 = (trace - disc) / 2.0;

        // Eigenvector for λ₁ (larger eigenvalue)
        let (ev0, ev1) = if b.abs() > 1e-30 {
            let v0 = [b, lambda1 - a];
            let norm = (v0[0] * v0[0] + v0[1] * v0[1]).sqrt();
            ([v0[0] / norm, v0[1] / norm], [v0[1] / norm, -v0[0] / norm])
        } else {
            // Diagonal: eigenvectors are standard basis, sorted by value
            if a >= d {
                ([1.0, 0.0], [0.0, 1.0])
            } else {
                ([0.0, 1.0], [1.0, 0.0])
            }
        };

        let eigenvectors = Self::from_rows([[ev0[0], ev1[0]], [ev0[1], ev1[1]]]);

        (FixedVector::new([lambda1, lambda2]), eigenvectors)
    }

    /// Inverse of a 2x2 matrix using cofactor formula.
    ///
    /// Returns `None` when the matrix is singular (determinant ≈ 0).
    pub fn try_inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det == 0.0 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self::from_rows([
            [self[(1, 1)] * inv_det, -self[(0, 1)] * inv_det],
            [-self[(1, 0)] * inv_det, self[(0, 0)] * inv_det],
        ]))
    }
}

// --- 3x3 symmetric eigendecomposition helpers --------------------------------

/// Cross-product of two 3-vectors.
#[inline]
fn cross3(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

/// Normalize a 3-vector in place; leaves zero vectors unchanged.
#[inline]
fn normalize3(v: &mut [f64; 3]) {
    let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if n2 > 0.0 {
        let inv = 1.0 / n2.sqrt();
        v[0] *= inv;
        v[1] *= inv;
        v[2] *= inv;
    } else {
        *v = [1.0, 0.0, 0.0];
    }
}

/// Compute the unit eigenvector of the 3x3 symmetric matrix `[a b c; b e f; c f i]`
/// for eigenvalue `λ` using the cross-product of two columns of `(A - λI)`.
///
/// Picks the pair whose cross-product has the largest norm for numerical
/// stability. Degenerate (repeated-eigenvalue) cases fall back to `[1,0,0]`.
#[inline]
fn eigenvector_3x3(a: f64, b: f64, c: f64, e: f64, f: f64, i: f64, lambda: f64) -> [f64; 3] {
    let col0 = [a - lambda, b, c];
    let col1 = [b, e - lambda, f];
    let col2 = [c, f, i - lambda];

    let v01 = cross3(col0, col1);
    let v12 = cross3(col1, col2);
    let v20 = cross3(col2, col0);

    let n01 = v01[0] * v01[0] + v01[1] * v01[1] + v01[2] * v01[2];
    let n12 = v12[0] * v12[0] + v12[1] * v12[1] + v12[2] * v12[2];
    let n20 = v20[0] * v20[0] + v20[1] * v20[1] + v20[2] * v20[2];

    let mut best = if n01 >= n12 && n01 >= n20 {
        v01
    } else if n12 >= n01 && n12 >= n20 {
        v12
    } else {
        v20
    };
    normalize3(&mut best);
    best
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

    #[test]
    fn fixed_matrix_iterates_in_row_major_order() {
        let matrix = FixedMatrix::from_rows([[1.0, 2.0], [3.0, 4.0]]);

        assert_eq!(
            matrix.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn fixed_3x3_symmetric_eigen_matches_known_values() {
        let m = FixedMatrix::from_rows([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]);
        let (vals, vecs) = m.symmetric_eigen();

        // Known eigenvalues: 2±√2, 2+√2≈3.4142, 2, 2-√2≈0.5858
        assert!((vals[0] - (2.0 + 2.0_f64.sqrt())).abs() < 1e-14);
        assert!((vals[1] - 2.0).abs() < 1e-14);
        assert!((vals[2] - (2.0 - 2.0_f64.sqrt())).abs() < 1e-14);

        // A*V = V*Λ (reconstruct and check)
        let lambda = FixedMatrix::from_rows([
            [vals[0], 0.0, 0.0],
            [0.0, vals[1], 0.0],
            [0.0, 0.0, vals[2]],
        ]);
        let av = m * vecs;
        let vd = vecs * lambda;
        for row in 0..3 {
            for col in 0..3 {
                assert!((av[(row, col)] - vd[(row, col)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn fixed_3x3_symmetric_eigen_eigenvectors_are_orthonormal() {
        let m = FixedMatrix::from_rows([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]]);
        let (_, vecs) = m.symmetric_eigen();

        // Columns should be orthonormal: V^T V = I
        let vt = vecs.transpose();
        let product = vt * vecs;
        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!((product[(row, col)] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn fixed_3x3_symmetric_eigen_identity_matrix() {
        let m: FixedMatrix<f64, 3, 3> = FixedMatrix::identity();
        let (vals, vecs) = m.symmetric_eigen();

        assert!((vals[0] - 1.0).abs() < 1e-15);
        assert!((vals[1] - 1.0).abs() < 1e-15);
        assert!((vals[2] - 1.0).abs() < 1e-15);

        let i: FixedMatrix<f64, 3, 3> = FixedMatrix::identity();
        for row in 0..3 {
            for col in 0..3 {
                assert!((vecs[(row, col)] - i[(row, col)]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn fixed_2x2_symmetric_eigen_matches_quadratic_formula() {
        let m = FixedMatrix::from_rows([[2.0, 1.0], [1.0, 2.0]]);
        let (vals, vecs) = m.symmetric_eigen();

        assert!((vals[0] - 3.0).abs() < 1e-15);
        assert!((vals[1] - 1.0).abs() < 1e-15);

        // A*V = V*Λ
        let lambda = FixedMatrix::from_rows([[vals[0], 0.0], [0.0, vals[1]]]);
        let av = m * vecs;
        let vd = vecs * lambda;
        for row in 0..2 {
            for col in 0..2 {
                assert!((av[(row, col)] - vd[(row, col)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn fixed_2x2_symmetric_eigen_eigenvectors_are_orthonormal() {
        let m = FixedMatrix::from_rows([[3.0, 2.0], [2.0, 6.0]]);
        let (_, vecs) = m.symmetric_eigen();

        let vt = vecs.transpose();
        let product = vt * vecs;
        for row in 0..2 {
            for col in 0..2 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!((product[(row, col)] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn fixed_matrix_converts_row_and_column_major_storage() {
        let row_major = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let column_major = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

        let matrix = FixedMatrix::from_row_major(row_major);

        assert_eq!(matrix.into_row_major(), row_major);
        assert_eq!(matrix.into_column_major(), column_major);
        assert_eq!(FixedMatrix::from_column_major(column_major), matrix);
    }
}
