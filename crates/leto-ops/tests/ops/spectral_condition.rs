//! The Bauer–Fike condition factor of a small real matrix, computed from its
//! left and right eigenvectors in `f64` — independently of the Schur routine
//! under test — for eigenvalue bounds that need no tuned constant.
//!
//! For a diagonalizable `A = V·Λ·V⁻¹` and any perturbation `E`, every
//! eigenvalue of `A + E` lies within `κ(V)·‖E‖₂` of an eigenvalue of `A`
//! (Bauer & Fike 1960; Golub & Van Loan, *Matrix Computations*, 4th ed.,
//! Theorem 7.2.2), and `κ₂(V) ≤ ‖V‖_F·‖V⁻¹‖_F`. With distinct eigenvalues,
//! the right eigenvectors `xᵢ` (columns of `V`, normalized to unit length) and
//! the left eigenvectors `yᵢ` satisfy `yᵢᵀxⱼ = 0` for `i ≠ j`, so the rows of
//! `V⁻¹` are `yᵢᵀ/(yᵢᵀxᵢ)` exactly and
//! `‖V‖_F·‖V⁻¹‖_F = √n·(Σᵢ ‖yᵢ‖²/(yᵢᵀxᵢ)²)^½`.

/// `u × v` in ℝ³.
fn cross(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// A null vector of the rank-2 matrix with rows `rows`: the largest of the
/// three pairwise cross products (the best-conditioned pair).
fn null_vector(rows: [[f64; 3]; 3]) -> [f64; 3] {
    [
        cross(rows[0], rows[1]),
        cross(rows[0], rows[2]),
        cross(rows[1], rows[2]),
    ]
    .into_iter()
    .max_by(|a, b| norm(*a).total_cmp(&norm(*b)))
    .expect("three candidates")
}

/// `‖V‖_F·‖V⁻¹‖_F ≥ κ₂(V)` for the row-major 3×3 `a` with the real, distinct
/// eigenvalues `eigenvalues`.
pub fn bauer_fike_factor(a: &[f64; 9], eigenvalues: &[f64; 3]) -> f64 {
    let mut sum = 0.0;
    for &lambda in eigenvalues {
        let shifted = |i: usize, j: usize| a[i * 3 + j] - if i == j { lambda } else { 0.0 };
        let rows = [0, 1, 2].map(|i| [0, 1, 2].map(|j| shifted(i, j)));
        let columns = [0, 1, 2].map(|j| [0, 1, 2].map(|i| shifted(i, j)));
        let right = null_vector(rows);
        let right_norm = norm(right);
        let right = right.map(|x| x / right_norm);
        let left = null_vector(columns);
        let alignment: f64 = left.iter().zip(&right).map(|(l, r)| l * r).sum();
        sum += norm(left).powi(2) / alignment.powi(2);
    }
    3.0_f64.sqrt() * sum.sqrt()
}

#[test]
fn bauer_fike_factor_of_a_normal_matrix_is_n() {
    // Symmetric ⇒ V orthogonal ⇒ ‖V‖_F·‖V⁻¹‖_F = √3·√3 = 3 (κ₂ = 1).
    let a = [2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
    let root2 = 2.0_f64.sqrt();
    let factor = bauer_fike_factor(&a, &[2.0 - root2, 2.0, 2.0 + root2]);
    // Relative error: the rounded eigenvalues (`|δλ| ≤ ε/2·3.42`) tilt each
    // null vector by at most `2·|δλ|/gap`, `gap = √2`, i.e. `≤ 2.5ε`, and the
    // cross products, norms, dot and quotient round at most 12 times
    // (`γ₁₂ ≈ 6ε`, Higham 2002 §3.1): `≲ 9ε` per term, `≤ 12ε` after the
    // sum and root — `36ε` absolute at `factor = 3`.
    assert!((factor - 3.0).abs() <= 36.0 * f64::EPSILON, "{factor}");
}
