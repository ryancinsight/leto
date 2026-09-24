//! Symmetric tridiagonal-QL eigensolver: closed forms, backward-error
//! residuals, and differential agreement with the Jacobi solver.
//!
//! # Tolerance derivation
//!
//! Householder tridiagonalization and QL apply `O(n)` orthogonal
//! transformations, each with backward error `O(n)·ε·‖A‖`, so the computed
//! decomposition is exact for `A + E` with `‖E‖₂ ≤ ‖E‖_F ≤ n²·ε·‖A‖_F`
//! (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Lemma
//! 19.3 with `r = n` transformations of `γ̃_n` each). Weyl's inequality turns
//! that into `|λ̂ᵢ − λᵢ| ≤ n²·ε·‖A‖_F`, the residual `‖A v̂ − λ̂ v̂‖₂` obeys the
//! same bound, and the accumulated eigenvector matrix is orthonormal to
//! `n²·ε`. The Jacobi reference stops once every off-diagonal entry is below
//! its tolerance `τ`, adding at most `n·τ` (the Frobenius norm of the
//! remainder) to its own `n²·ε·‖A‖_F`.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array2, SliceArg, Storage};
use leto_ops::{
    symmetric_eigen_jacobi_with_tolerance, symmetric_eigen_qr, SymmetricEigenWorkspace, Xorshift64,
};

/// Seeded symmetric matrix with entries uniform in `[-1, 1)`.
fn random_symmetric(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Xorshift64::new(seed);
    let mut values = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let value = 2.0 * rng.next_unit_f64() - 1.0;
            values[i * n + j] = value;
            values[j * n + i] = value;
        }
    }
    values
}

/// Gram matrix `XᵀX` of a seeded `rows × n` matrix: rank `min(rows, n)`.
fn random_gram(rows: usize, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Xorshift64::new(seed);
    let x: Vec<f64> = (0..rows * n).map(|_| rng.next_unit_f64() - 0.5).collect();
    let mut gram = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            gram[i * n + j] = (0..rows).map(|r| x[r * n + i] * x[r * n + j]).sum();
        }
    }
    gram
}

fn frobenius(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// `n²·ε·‖A‖_F`, the derived backward-error bound.
fn backward_bound(values: &[f64], n: usize) -> f64 {
    (n * n) as f64 * f64::EPSILON * frobenius(values)
}

/// Every eigenpair satisfies `‖A v − λ v‖₂ ≤ bound` and the eigenvectors are
/// orthonormal to `n²·ε`.
fn assert_backward_stable(values: &[f64], n: usize, workspace: &SymmetricEigenWorkspace<f64>) {
    let bound = backward_bound(values, n);
    let vectors: Vec<&[f64]> = workspace.eigenvectors().collect();
    assert_eq!(vectors.len(), n);
    for (&lambda, v) in workspace.eigenvalues().iter().zip(&vectors) {
        let residual = (0..n)
            .map(|i| {
                let av: f64 = (0..n).map(|j| values[i * n + j] * v[j]).sum();
                (av - lambda * v[i]).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        assert!(residual <= bound, "residual {residual} exceeds {bound}");
    }
    let orthogonality = (n * n) as f64 * f64::EPSILON;
    for (a, u) in vectors.iter().enumerate() {
        for (b, w) in vectors.iter().enumerate() {
            let dot: f64 = u.iter().zip(*w).map(|(x, y)| x * y).sum();
            let expected = if a == b { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() <= orthogonality,
                "v{a}·v{b} = {dot}, expected {expected}"
            );
        }
    }
}

#[test]
fn symmetric_eigen_qr_matches_closed_forms() {
    // [[2,1],[1,2]]: 1, 3.
    let pair = Array2::from_shape_vec([2, 2], vec![2.0_f64, 1.0, 1.0, 2.0]).unwrap();
    let eigen = symmetric_eigen_qr(&pair.view()).unwrap();
    assert!((eigen.eigenvalues[0] - 1.0).abs() <= 4.0 * f64::EPSILON);
    assert!((eigen.eigenvalues[1] - 3.0).abs() <= 4.0 * 3.0 * f64::EPSILON);
    // Eigenvectors as columns: column 1 is ±(1, 1)/√2.
    let columns = eigen.eigenvectors.storage().as_slice();
    assert!((columns[1].abs() - std::f64::consts::FRAC_1_SQRT_2).abs() <= 4.0 * f64::EPSILON);
    assert!((columns[1] - columns[3]).abs() <= 4.0 * f64::EPSILON);

    // Path-graph Laplacian P_n: λ_k = 2 − 2 cos(kπ/n), k = 0 … n−1.
    let n = 12;
    let mut laplacian = vec![0.0; n * n];
    for i in 0..n {
        let degree = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
        laplacian[i * n + i] = degree;
        if i + 1 < n {
            laplacian[i * n + i + 1] = -1.0;
            laplacian[(i + 1) * n + i] = -1.0;
        }
    }
    let matrix = Array2::from_shape_vec([n, n], laplacian.clone()).unwrap();
    let mut workspace = SymmetricEigenWorkspace::new();
    workspace.decompose(&matrix.view()).unwrap();
    let bound = backward_bound(&laplacian, n);
    for (k, &value) in workspace.eigenvalues().iter().enumerate() {
        let expected = 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / n as f64).cos();
        assert!(
            (value - expected).abs() <= bound,
            "λ{k} = {value}, expected {expected}"
        );
    }
    assert_backward_stable(&laplacian, n, &workspace);
}

#[test]
fn symmetric_eigen_qr_agrees_with_jacobi_within_backward_error() {
    let tolerance = 1.0e-12;
    for (n, seed) in [(3, 7_u64), (17, 11), (60, 13)] {
        let values = random_symmetric(n, seed);
        let matrix = Array2::from_shape_vec([n, n], values.clone()).unwrap();
        let mut workspace = SymmetricEigenWorkspace::new();
        workspace.decompose(&matrix.view()).unwrap();
        assert_backward_stable(&values, n, &workspace);

        let reference = symmetric_eigen_jacobi_with_tolerance(&matrix.view(), tolerance).unwrap();
        let bound = 2.0 * backward_bound(&values, n) + n as f64 * tolerance;
        for (qr, jacobi) in workspace.eigenvalues().iter().zip(&reference.eigenvalues) {
            assert!(
                (qr - jacobi).abs() <= bound,
                "n={n}: {qr} vs {jacobi}, bound {bound}"
            );
        }
    }
}

#[test]
fn symmetric_eigen_qr_resolves_rank_deficient_gram_matrices() {
    // A rank-10 Gram matrix of order 60: fifty zero eigenvalues, the exact
    // geometry of a noise-free MP-PCA window.
    let (rank, n) = (10, 60);
    let gram = random_gram(rank, n, 29);
    let matrix = Array2::from_shape_vec([n, n], gram.clone()).unwrap();
    let mut workspace = SymmetricEigenWorkspace::new();
    workspace.decompose(&matrix.view()).unwrap();
    let bound = backward_bound(&gram, n);
    for &value in &workspace.eigenvalues()[..n - rank] {
        assert!(
            value.abs() <= bound,
            "null eigenvalue {value} exceeds {bound}"
        );
    }
    let trace: f64 = (0..n).map(|i| gram[i * n + i]).sum();
    let signal: f64 = workspace.eigenvalues()[n - rank..].iter().sum();
    assert!((signal - trace).abs() <= n as f64 * bound);
    assert_backward_stable(&gram, n, &workspace);
}

#[test]
fn symmetric_eigen_qr_sorts_a_diagonal_matrix_exactly() {
    let diagonal = [4.0, -2.0, 7.0, -2.0, 0.0];
    let n = diagonal.len();
    let mut values = vec![0.0; n * n];
    for (i, &d) in diagonal.iter().enumerate() {
        values[i * n + i] = d;
    }
    let eigen =
        symmetric_eigen_qr(&Array2::from_shape_vec([n, n], values).unwrap().view()).unwrap();
    assert_eq!(eigen.eigenvalues, vec![-2.0, -2.0, 0.0, 4.0, 7.0]);
}

#[test]
fn symmetric_eigen_workspace_reuse_is_bitwise_identical_to_a_fresh_solve() {
    let large = random_symmetric(9, 3);
    let small = random_symmetric(4, 5);
    let large_matrix = Array2::from_shape_vec([9, 9], large).unwrap();
    let small_matrix = Array2::from_shape_vec([4, 4], small).unwrap();

    let mut fresh = SymmetricEigenWorkspace::new();
    fresh.decompose(&small_matrix.view()).unwrap();
    let mut reused = SymmetricEigenWorkspace::new();
    reused.decompose(&large_matrix.view()).unwrap();
    reused.decompose(&small_matrix.view()).unwrap();

    assert_eq!(reused.order(), 4);
    assert_eq!(reused.eigenvalues(), fresh.eigenvalues());
    assert!(reused.eigenvectors().eq(fresh.eigenvectors()));
}

#[test]
fn symmetric_eigen_qr_reads_only_the_lower_triangle_of_strided_views() {
    // The even rows and columns form [[4, 1], [1, 4]] below the diagonal; the
    // upper entry is garbage the solver must not read.
    let matrix = Array2::from_shape_vec(
        [4, 4],
        vec![
            4.0_f64, 0.0, 99.0, 0.0, 0.0, 9.0, 0.0, 8.0, 1.0, 0.0, 4.0, 0.0, 0.0, 8.0, 0.0, 9.0,
        ],
    )
    .unwrap();
    let view = matrix
        .view()
        .slice_with::<2>(&[
            SliceArg::range(Some(0), None, 2),
            SliceArg::range(Some(0), None, 2),
        ])
        .unwrap();
    let eigen = symmetric_eigen_qr(&view).unwrap();
    assert!((eigen.eigenvalues[0] - 3.0).abs() <= 8.0 * f64::EPSILON);
    assert!((eigen.eigenvalues[1] - 5.0).abs() <= 8.0 * 5.0 * f64::EPSILON);
}

#[test]
fn symmetric_eigen_qr_is_generic_over_f32() {
    let n = 60;
    let values = random_symmetric(n, 17);
    let narrow: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    let reference =
        symmetric_eigen_qr(&Array2::from_shape_vec([n, n], values).unwrap().view()).unwrap();
    let mut workspace = SymmetricEigenWorkspace::<f32>::new();
    workspace
        .decompose(
            &Array2::from_shape_vec([n, n], narrow.clone())
                .unwrap()
                .view(),
        )
        .unwrap();
    // The f64 reference solves the rounded f32 input to f64 accuracy, so the
    // f32 bound n²·ε₃₂·‖A‖_F dominates.
    let norm = narrow
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt();
    let bound = (n * n) as f64 * f64::from(f32::EPSILON) * norm;
    for (narrow, wide) in workspace.eigenvalues().iter().zip(&reference.eigenvalues) {
        assert!((f64::from(*narrow) - wide).abs() <= bound);
    }
}

#[test]
fn symmetric_eigen_qr_handles_degenerate_orders() {
    let empty = Array2::<f64>::from_shape_vec([0, 0], vec![]).unwrap();
    let eigen = symmetric_eigen_qr(&empty.view()).unwrap();
    assert!(eigen.eigenvalues.is_empty());

    let single = Array2::from_shape_vec([1, 1], vec![-3.5]).unwrap();
    let mut workspace = SymmetricEigenWorkspace::new();
    workspace.decompose(&single.view()).unwrap();
    assert_eq!(workspace.eigenvalues(), &[-3.5]);
    assert_eq!(workspace.eigenvectors().next().unwrap(), &[1.0]);
}

#[test]
fn symmetric_eigen_qr_rejects_invalid_inputs() {
    let rectangular = Array2::from_shape_vec([2, 3], vec![1.0_f64; 6]).unwrap();
    assert!(matches!(
        symmetric_eigen_qr(&rectangular.view()),
        Err(leto::LetoError::ShapeMismatch { .. })
    ));

    let non_finite = Array2::from_shape_vec([2, 2], vec![1.0, 0.0, f64::NAN, 1.0]).unwrap();
    let mut workspace = SymmetricEigenWorkspace::new();
    workspace
        .decompose(&Array2::from_shape_vec([1, 1], vec![2.0]).unwrap().view())
        .unwrap();
    let error = workspace.decompose(&non_finite.view()).unwrap_err();
    assert!(error.to_string().contains("(1, 0)"), "{error}");
    assert_eq!(workspace.order(), 0);
    assert!(workspace.eigenvalues().is_empty());
}
