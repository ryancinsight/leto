//! Fixed-size f64 linear algebra: the 2x2 and 3x3 `FixedMatrix`
//! symmetric eigendecompositions and inverses.
//!
//! Split from the parent module so the stack-backed types and their
//! elementwise operators stay one concern and the analytic linear algebra
//! another; the inherent methods remain on the same public types.

use super::{FixedMatrix, FixedVector};
use crate::geometry::Vector3;

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

        // Depressed cubic: μ³ + pμ + q = 0,  λ = μ + I₁/3, obtained by
        // substituting λ = μ + I₁/3 into λ³ − I₁λ² + I₂λ − I₃ = 0.
        let p = i2 - i1 * i1 / 3.0;
        let q = (-2.0 * i1 * i1 * i1 + 9.0 * i1 * i2 - 27.0 * i3) / 27.0;
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

/// Cross-product of two 3-vectors, routed through [`Vector3::cross`].
#[inline]
fn cross3(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    Vector3::new(u[0], u[1], u[2])
        .cross(Vector3::new(v[0], v[1], v[2]))
        .data
}

/// Normalize a 3-vector in place, routed through [`Vector3::normalize`].
///
/// A zero vector keeps the canonical `+x` direction the analytic eigenvector
/// construction relies on for its fully degenerate fallback, rather than the
/// `0 · ∞` a bare division by the zero norm would produce.
#[inline]
fn normalize3(v: &mut [f64; 3]) {
    let vector = Vector3::new(v[0], v[1], v[2]);
    let norm = vector.norm();
    if norm > 0.0 {
        *v = (vector * norm.recip()).data;
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

#[cfg(test)]
mod tests {
    use super::FixedMatrix;

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
    fn fixed_3x3_symmetric_eigen_nonzero_depressed_constant() {
        // Regression: the `matches_known_values` matrix has eigenvalues symmetric
        // about their mean (q = 0 in the depressed cubic), which masks the sign of
        // the cubic's constant term. These matrices have q ≠ 0 and expose it.

        // Distinct eigenvalues (oracle from leto eigh, descending).
        let m = FixedMatrix::from_rows([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]]);
        let (vals, _) = m.symmetric_eigen();
        assert!((vals[0] - 9.418_832_675_97).abs() < 1e-9);
        assert!((vals[1] - 3.386_770_156_608).abs() < 1e-9);
        assert!((vals[2] - 2.194_397_167_422).abs() < 1e-9);

        // Degenerate double root (isotropic-like): eigenvalues {3, 1, 1}, q ≠ 0.
        // At a double root the closed-form trig method is ill-conditioned:
        // cos_arg → ±1, and acos(1 − ε) ≈ √(2ε), so a rounding error ε ≈ machine
        // epsilon in cos_arg amplifies to O(√ε) ≈ 1e-8 in the repeated eigenvalue.
        let iso = FixedMatrix::from_rows([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]]);
        let (vals, _) = iso.symmetric_eigen();
        assert!((vals[0] - 3.0).abs() < 1e-9);
        assert!((vals[1] - 1.0).abs() < 1e-7);
        assert!((vals[2] - 1.0).abs() < 1e-7);
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
}
