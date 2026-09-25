//! The Bauer–Fike condition factor of a small real matrix, computed from its
//! left and right eigenvectors in `f64` — independently of the Schur routine
//! under test — for eigenvalue bounds that need no tuned constant.
//!
//! For a diagonalizable `A = V·Λ·V⁻¹` and any perturbation `E`, every
//! eigenvalue of `A + E` lies within `κ(V)·‖E‖₂` of an eigenvalue of `A`
//! (Bauer & Fike 1960; Golub & Van Loan, *Matrix Computations*, 4th ed.,
//! Theorem 7.2.2), and `κ₂(V) ≤ ‖V‖_F·‖V⁻¹‖_F`. With distinct eigenvalues,
//! the right eigenvectors `xᵢ` (columns of `V`, normalized to unit length) and
//! the left eigenvectors `yᵢ` (`yᵢᵀA = λᵢyᵢᵀ`) satisfy `yᵢᵀxⱼ = 0` for `i ≠ j`,
//! so the rows of `V⁻¹` are `yᵢᵀ/(yᵢᵀxᵢ)` exactly and
//! `‖V‖_F·‖V⁻¹‖_F = √n·(Σᵢ ‖yᵢ‖²/|yᵢᵀxᵢ|²)^½`. Complex eigenvalues are
//! handled in complex arithmetic (the transposes are unconjugated, as the
//! biorthogonality `yᵢᵀxⱼ = 0` requires).

/// A complex number `re + i·im` in `f64`.
#[derive(Clone, Copy)]
struct Complex(f64, f64);

impl Complex {
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0, self.1 + other.1)
    }
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0, self.1 - other.1)
    }
    fn mul(self, other: Self) -> Self {
        Self(
            self.0 * other.0 - self.1 * other.1,
            self.0 * other.1 + self.1 * other.0,
        )
    }
    fn norm_sq(self) -> f64 {
        self.0 * self.0 + self.1 * self.1
    }
}

/// `u × v` in ℂ³ (bilinear: `(u × v)·u = (u × v)·v = 0` unconjugated).
fn cross(u: [Complex; 3], v: [Complex; 3]) -> [Complex; 3] {
    [
        u[1].mul(v[2]).sub(u[2].mul(v[1])),
        u[2].mul(v[0]).sub(u[0].mul(v[2])),
        u[0].mul(v[1]).sub(u[1].mul(v[0])),
    ]
}

fn norm_sq(v: [Complex; 3]) -> f64 {
    v.iter().map(|z| z.norm_sq()).sum()
}

/// A null vector of the rank-2 matrix with rows `rows`: the largest of the
/// three pairwise cross products (the best-conditioned pair).
fn null_vector(rows: [[Complex; 3]; 3]) -> [Complex; 3] {
    [
        cross(rows[0], rows[1]),
        cross(rows[0], rows[2]),
        cross(rows[1], rows[2]),
    ]
    .into_iter()
    .max_by(|a, b| norm_sq(*a).total_cmp(&norm_sq(*b)))
    .expect("three candidates")
}

/// `‖V‖_F·‖V⁻¹‖_F ≥ κ₂(V)` for the row-major 3×3 `a` with the distinct
/// eigenvalues `eigenvalues`, each `(re, im)`.
pub fn bauer_fike_factor(a: &[f64; 9], eigenvalues: &[(f64, f64); 3]) -> f64 {
    let mut sum = 0.0;
    for &(re, im) in eigenvalues {
        let lambda = Complex(re, im);
        let shifted = |i: usize, j: usize| {
            let entry = Complex(a[i * 3 + j], 0.0);
            if i == j {
                entry.sub(lambda)
            } else {
                entry
            }
        };
        let rows = [0, 1, 2].map(|i| [0, 1, 2].map(|j| shifted(i, j)));
        let columns = [0, 1, 2].map(|j| [0, 1, 2].map(|i| shifted(i, j)));
        let right = null_vector(rows);
        let right_norm = norm_sq(right).sqrt();
        let right = right.map(|z| Complex(z.0 / right_norm, z.1 / right_norm));
        let left = null_vector(columns);
        let alignment = left
            .iter()
            .zip(&right)
            .fold(Complex(0.0, 0.0), |acc, (l, r)| acc.add(l.mul(*r)));
        sum += norm_sq(left) / alignment.norm_sq();
    }
    3.0_f64.sqrt() * sum.sqrt()
}

#[test]
fn bauer_fike_factor_of_a_normal_matrix_is_n() {
    // Symmetric ⇒ V orthogonal ⇒ ‖V‖_F·‖V⁻¹‖_F = √3·√3 = 3 (κ₂ = 1).
    let a = [2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
    let root2 = 2.0_f64.sqrt();
    let factor = bauer_fike_factor(&a, &[(2.0 - root2, 0.0), (2.0, 0.0), (2.0 + root2, 0.0)]);
    // Relative error: the rounded eigenvalues (`|δλ| ≤ ε/2·3.42`) tilt each
    // null vector by at most `2·|δλ|/gap`, `gap = √2`, i.e. `≤ 2.5ε`, and the
    // cross products, norms, dot and quotient round at most 12 times
    // (`γ₁₂ ≈ 6ε`, Higham 2002 §3.1): `≲ 9ε` per term, `≤ 12ε` after the
    // sum and root — `36ε` absolute at `factor = 3`.
    assert!((factor - 3.0).abs() <= 36.0 * f64::EPSILON, "{factor}");
}

#[test]
fn bauer_fike_factor_of_a_normal_complex_spectrum_is_n() {
    // The rotation-plus-scalar [[0, −1, 0], [1, 0, 0], [0, 0, 2]] is normal,
    // with eigenvalues ±i and 2: κ₂ = 1, so the factor is exactly √3·√3 = 3
    // up to the same rounding as above (every eigenvalue exact here).
    let a = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0];
    let factor = bauer_fike_factor(&a, &[(0.0, 1.0), (0.0, -1.0), (2.0, 0.0)]);
    assert!((factor - 3.0).abs() <= 36.0 * f64::EPSILON, "{factor}");
}
