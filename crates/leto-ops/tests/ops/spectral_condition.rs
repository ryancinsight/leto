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
    fn reciprocal(self) -> Self {
        let d = self.norm_sq();
        Self(self.0 / d, -self.1 / d)
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

/// Solve `(M − λI)·x = b` in complex arithmetic by Gaussian elimination with
/// partial pivoting, a zero pivot replaced by `ε·‖M‖` (inverse iteration's
/// convention: `λ` is an eigenvalue, so the shifted matrix is singular to
/// rounding and the solution is dominated by the eigenvector).
fn shifted_solve(
    m: &[f64],
    n: usize,
    lambda: Complex,
    transpose: bool,
    b: &[Complex],
) -> Vec<Complex> {
    let entry = |i: usize, j: usize| {
        let value = if transpose {
            m[j * n + i]
        } else {
            m[i * n + j]
        };
        if i == j {
            Complex(value, 0.0).sub(lambda)
        } else {
            Complex(value, 0.0)
        }
    };
    let mut lu: Vec<Complex> = (0..n * n).map(|k| entry(k / n, k % n)).collect();
    let mut x: Vec<Complex> = b.to_vec();
    let scale = m
        .iter()
        .map(|v| v.abs())
        .sum::<f64>()
        .max(lambda.norm_sq().sqrt());
    let tiny = f64::EPSILON * scale.max(f64::MIN_POSITIVE);
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&a, &b| {
                lu[a * n + col]
                    .norm_sq()
                    .total_cmp(&lu[b * n + col].norm_sq())
            })
            .expect("non-empty");
        if pivot != col {
            for j in 0..n {
                lu.swap(col * n + j, pivot * n + j);
            }
            x.swap(col, pivot);
        }
        if lu[col * n + col].norm_sq() == 0.0 {
            lu[col * n + col] = Complex(tiny, 0.0);
        }
        let inverse = lu[col * n + col].reciprocal();
        for row in (col + 1)..n {
            let factor = lu[row * n + col].mul(inverse);
            for j in col..n {
                lu[row * n + j] = lu[row * n + j].sub(factor.mul(lu[col * n + j]));
            }
            x[row] = x[row].sub(factor.mul(x[col]));
        }
    }
    for col in (0..n).rev() {
        let mut value = x[col];
        for j in (col + 1)..n {
            value = value.sub(lu[col * n + j].mul(x[j]));
        }
        x[col] = value.mul(lu[col * n + col].reciprocal());
    }
    x
}

/// A unit eigenvector of `m` (or of `mᵀ`) for the eigenvalue `λ`, by two
/// steps of inverse iteration from a fixed start.
fn inverse_iteration(m: &[f64], n: usize, lambda: Complex, transpose: bool) -> Vec<Complex> {
    let mut x: Vec<Complex> = (0..n)
        .map(|i| Complex(1.0 + 0.125 * i as f64, 0.25 * ((i * 7) % 5) as f64))
        .collect();
    for _ in 0..2 {
        x = shifted_solve(m, n, lambda, transpose, &x);
        let norm = x.iter().map(|z| z.norm_sq()).sum::<f64>().sqrt();
        x = x
            .into_iter()
            .map(|z| Complex(z.0 / norm, z.1 / norm))
            .collect();
    }
    x
}

/// `‖V‖_F·‖V⁻¹‖_F ≥ κ₂(V)` for the row-major `n × n` matrix `m` with the
/// distinct eigenvalues `eigenvalues`, each `(re, im)` — the right and left
/// eigenvectors found by inverse iteration in complex `f64`, combined as in
/// [`bauer_fike_factor`] (rows of `V⁻¹` are `yᵢᵀ/(yᵢᵀxᵢ)` for unit `xᵢ`).
/// A (numerically) repeated eigenvalue makes `yᵢᵀxᵢ` small and the factor
/// correspondingly large, as the Bauer–Fike bound requires.
pub fn condition_by_inverse_iteration(m: &[f64], n: usize, eigenvalues: &[(f64, f64)]) -> f64 {
    let mut sum = 0.0;
    for &(re, im) in eigenvalues {
        let lambda = Complex(re, im);
        let right = inverse_iteration(m, n, lambda, false);
        let left = inverse_iteration(m, n, lambda, true);
        let alignment = left
            .iter()
            .zip(&right)
            .fold(Complex(0.0, 0.0), |acc, (l, r)| acc.add(l.mul(*r)));
        sum += 1.0 / alignment.norm_sq();
    }
    (n as f64).sqrt() * sum.sqrt()
}

#[test]
fn condition_by_inverse_iteration_matches_the_closed_form() {
    // SIMILAR = S·diag(1, 2, 4)·S⁻¹ of `scale_range.rs`: the factor from the
    // unit-normalized eigenvectors equals the cross-product route's.
    let a = [1.5, 0.5, -0.5, -1.0, 3.0, 1.0, -1.5, 1.5, 2.5];
    let spectrum = [(1.0, 0.0), (2.0, 0.0), (4.0, 0.0)];
    let by_iteration = condition_by_inverse_iteration(&a, 3, &spectrum);
    let by_cross = bauer_fike_factor(&a, &spectrum);
    assert!(
        (by_iteration - by_cross).abs() <= 1e-10 * by_cross,
        "{by_iteration} vs {by_cross}"
    );
}
