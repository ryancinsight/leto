//! Symmetric tridiagonal-QL eigensolver: closed forms, backward-error
//! residuals over every supported scalar, clusters, dynamic range, the full
//! exponent range of each format, and differential agreement with Jacobi.
//!
//! # Tolerance derivation
//!
//! The computed decomposition is exact for `A + E`, and `‖E‖_F` composes the
//! per-transformation backward errors of Higham, *Accuracy and Stability of
//! Numerical Algorithms*, 2nd ed. (2002), over the transformations actually
//! applied:
//!
//! - Householder tridiagonalization applies exactly `n − 2` reflectors (Golub
//!   & Van Loan, *Matrix Computations*, 4th ed., Algorithm 8.3.1), each
//!   committing `γ̃_n` per column (§19.3, Lemmas 19.2–19.3: `r` reflectors
//!   commit `r·γ̃_m`, `γ̃_k = c·k·u/(1 − c·k·u)`, `c` a small integer constant
//!   Higham leaves unspecified, `u = ε/2`).
//! - The implicit QL chase applies, per sweep over an active block of order
//!   `k`, `k − 1` Givens rotations, each committing `γ₆` (§19.6, Lemmas
//!   19.7–19.8). The sweep count is bounded by LAPACK `dsteqr`'s `30·n` cap
//!   (`ql.rs`'s `SWEEPS_PER_EIGENVALUE`; exhausting it is a typed error), so
//!   the worst case is `(n − 2)·γ̃_n + 30n·(n − 1)·γ₆`.
//!
//! Observed, the Wilkinson-shifted chase deflates in under two sweeps per
//! eigenvalue (cubic convergence, Golub & Van Loan §8.3), so `r ≈ n`
//! transformations of order-`n` vectors are applied and the first-order total
//! is `‖E‖_F ≤ n²·ε·‖A‖_F` at `c·u ≈ ε` — the bound asserted below, `ε` the
//! machine epsilon of the scalar the solver runs in. Two assumptions carry
//! it and are named rather than hidden: Higham's unspecified `c`, and the
//! observed (not worst-case) sweep count; a run that needed the `30·n` cap
//! would fail these assertions, which is the falsifiable form of the second.
//! Weyl's inequality turns that into `|λ̂ᵢ − λᵢ| ≤ n²·ε·‖A‖_F`; the residual `‖A v̂ − λ̂ v̂‖₂` obeys the same bound
//! and the accumulated eigenvectors are orthonormal to `n²·ε`. Where the
//! reference spectrum belongs to the `f64` matrix before rounding it into `T`,
//! the rounding adds `ε/2·‖A‖_F` (entrywise relative `ε/2`). Residuals and
//! orthogonality are evaluated in `f64` on the exact `f64` images of the `T`
//! values, whose own error (`n·ε₆₄`) is negligible against every bound here.
//! The Jacobi reference stops once every off-diagonal entry is below
//! `τ·‖A‖_F`, adding at most `n·τ·‖A‖_F` to its own `n²·ε·‖A‖_F`.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::format::epsilon;
use eunomia::{Bf16, F16};
use leto::{Array2, LetoError, SliceArg, Storage};
use leto_ops::{
    symmetric_eigen_jacobi_with_tolerance, symmetric_eigen_qr, RealScalar, SymmetricEigenWorkspace,
    Xorshift64,
};

mod scale;

/// Round `values` into `T`, returning the `T` matrix and the exact `f64`
/// image of what `T` holds.
pub(super) fn round_into<T: RealScalar>(values: &[f64], n: usize) -> (Array2<T>, Vec<f64>) {
    let narrow: Vec<T> = values.iter().map(|&v| T::from_f64(v)).collect();
    let image = narrow.iter().map(|v| v.to_f64()).collect();
    (Array2::from_shape_vec([n, n], narrow).unwrap(), image)
}

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

/// `Q·diag(spectrum)·Qᵀ` with `Q` the Householder reflector of a seeded
/// vector: an exact orthogonal similarity in real arithmetic, so the spectrum
/// is known.
pub(super) fn with_spectrum(spectrum: &[f64], seed: u64) -> Vec<f64> {
    let n = spectrum.len();
    let mut rng = Xorshift64::new(seed);
    let v: Vec<f64> = (0..n).map(|_| rng.next_unit_f64() - 0.5).collect();
    let scale = 2.0 / v.iter().map(|x| x * x).sum::<f64>();
    let q = |i: usize, j: usize| f64::from(u8::from(i == j)) - scale * v[i] * v[j];
    let mut values = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            values[i * n + j] = (0..n).map(|k| q(i, k) * spectrum[k] * q(j, k)).sum();
        }
    }
    // Exact symmetry: the sums above differ in rounding between (i, j) and (j, i).
    for i in 0..n {
        for j in 0..i {
            values[j * n + i] = values[i * n + j];
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
    let largest = values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if largest == 0.0 {
        return 0.0;
    }
    largest
        * values
            .iter()
            .map(|v| (v / largest).powi(2))
            .sum::<f64>()
            .sqrt()
}

/// `n²·ε(T)·‖A‖_F`, the derived backward-error bound.
pub(super) fn backward_bound<T: RealScalar>(values: &[f64], n: usize) -> f64 {
    (n * n) as f64 * epsilon::<T>() * frobenius(values)
}

/// Decompose `values` in `T` and assert every eigenpair's residual and the
/// eigenvectors' orthonormality against the derived bounds; returns the
/// eigenvalues as `f64`.
fn assert_backward_stable<T: RealScalar>(values: &[f64], n: usize) -> Vec<f64> {
    let (matrix, image) = round_into::<T>(values, n);
    let mut workspace = SymmetricEigenWorkspace::new();
    workspace.decompose(&matrix.view()).unwrap();
    let bound = backward_bound::<T>(&image, n);
    let vectors: Vec<Vec<f64>> = workspace
        .eigenvectors()
        .map(|v| v.iter().map(|x| x.to_f64()).collect())
        .collect();
    let lambdas: Vec<f64> = workspace.eigenvalues().iter().map(|x| x.to_f64()).collect();
    assert_eq!(vectors.len(), n);
    for (&lambda, v) in lambdas.iter().zip(&vectors) {
        let residual = (0..n)
            .map(|i| {
                let av: f64 = (0..n).map(|j| image[i * n + j] * v[j]).sum();
                (av - lambda * v[i]).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        assert!(residual <= bound, "residual {residual:e} exceeds {bound:e}");
    }
    let orthogonality = (n * n) as f64 * epsilon::<T>();
    for (a, u) in vectors.iter().enumerate() {
        for (b, w) in vectors.iter().enumerate() {
            let dot: f64 = u.iter().zip(w).map(|(x, y)| x * y).sum();
            let expected = if a == b { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() <= orthogonality,
                "v{a}·v{b} = {dot}, expected {expected}"
            );
        }
    }
    assert!(lambdas.windows(2).all(|pair| pair[0] <= pair[1]));
    lambdas
}

/// Every computed eigenvalue matches `spectrum` (ascending) within the
/// backward bound plus the rounding of the `f64` matrix into `T`.
pub(super) fn assert_spectrum<T: RealScalar>(values: &[f64], spectrum: &[f64]) {
    let n = spectrum.len();
    let computed = assert_backward_stable::<T>(values, n);
    let bound = backward_bound::<T>(values, n) + epsilon::<T>() / 2.0 * frobenius(values);
    for (value, expected) in computed.iter().zip(spectrum) {
        assert!(
            (value - expected).abs() <= bound,
            "{value} vs {expected}, bound {bound:e}"
        );
    }
}

fn check_random_matrices<T: RealScalar>(n: usize) {
    for seed in [3_u64, 5, 7] {
        assert_backward_stable::<T>(&random_symmetric(n, seed), n);
    }
}

fn check_clustered_spectrum<T: RealScalar>() {
    // Three clusters, each of three eigenvalues 10⁻⁶ apart — far below the
    // resolution of every format but f64, where they separate.
    let spectrum = [
        -2.0,
        -2.0 + 1e-6,
        -2.0 + 2e-6,
        0.5,
        0.5 + 1e-6,
        0.5 + 2e-6,
        3.0,
        3.0 + 1e-6,
        3.0 + 2e-6,
    ];
    assert_spectrum::<T>(&with_spectrum(&spectrum, 19), &spectrum);
}

#[test]
fn symmetric_eigen_qr_is_backward_stable_for_every_scalar() {
    // n = 6 keeps n²·ε(T)·‖A‖_F informative for Bf16 (36·7.8e-3 ≈ 0.28).
    check_random_matrices::<f64>(6);
    check_random_matrices::<f32>(6);
    check_random_matrices::<F16>(6);
    check_random_matrices::<Bf16>(6);
    check_random_matrices::<f64>(60);
    check_random_matrices::<f32>(60);
}

#[test]
fn symmetric_eigen_qr_resolves_clusters_for_every_scalar() {
    check_clustered_spectrum::<f64>();
    check_clustered_spectrum::<f32>();
    check_clustered_spectrum::<F16>();
    check_clustered_spectrum::<Bf16>();
}

/// [`symmetric_eigen_qr_matches_closed_forms`]'s body, instantiated over
/// every shipped scalar so a known-spectrum regression in `T`'s own
/// arithmetic (not just `f64`'s) fails this test (finding
/// `LETO-DENSE-SCALE-RANGE-2026-09-24`, item D).
fn check_matches_closed_forms<T: RealScalar>() {
    // [[2,1],[1,2]]: 1, 3.
    let pair_values = [2.0_f64, 1.0, 1.0, 2.0];
    let (pair, image) = round_into::<T>(&pair_values, 2);
    let eigen = symmetric_eigen_qr(&pair.view()).unwrap();
    // n²·ε(T)·‖A‖_F (`backward_bound`), n = 2: the derived backward-error
    // bound for a 2×2 closed form (see the module derivation), on the exact
    // image `T` holds.
    let bound = backward_bound::<T>(&image, 2);
    // Eigenvector orthonormality bound: n²·ε(T) (see `assert_backward_stable`).
    let orthogonality = 4.0 * epsilon::<T>();
    let values: Vec<f64> = eigen.eigenvalues.iter().map(|v| v.to_f64()).collect();
    assert!((values[0] - 1.0).abs() <= bound, "{}", values[0]);
    assert!((values[1] - 3.0).abs() <= bound, "{}", values[1]);
    // Eigenvectors as columns: column 1 is ±(1, 1)/√2.
    let columns: Vec<f64> = eigen
        .eigenvectors
        .storage()
        .as_slice()
        .iter()
        .map(|v| v.to_f64())
        .collect();
    assert!(
        (columns[1].abs() - std::f64::consts::FRAC_1_SQRT_2).abs() <= orthogonality,
        "{}",
        columns[1]
    );
    assert!((columns[1] - columns[3]).abs() <= orthogonality);

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
    let spectrum: Vec<f64> = (0..n)
        .map(|k| 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / n as f64).cos())
        .collect();
    assert_spectrum::<T>(&laplacian, &spectrum);
}

#[test]
fn symmetric_eigen_qr_matches_closed_forms() {
    check_matches_closed_forms::<f64>();
    check_matches_closed_forms::<f32>();
    check_matches_closed_forms::<F16>();
    check_matches_closed_forms::<Bf16>();
}

#[test]
fn symmetric_eigen_qr_agrees_with_jacobi_within_backward_error() {
    let tolerance = 1.0e-12;
    for (n, seed) in [(3, 7_u64), (17, 11), (60, 13)] {
        let values = random_symmetric(n, seed);
        let computed = assert_backward_stable::<f64>(&values, n);
        let matrix = Array2::from_shape_vec([n, n], values.clone()).unwrap();
        let reference = symmetric_eigen_jacobi_with_tolerance(&matrix.view(), tolerance).unwrap();
        let bound =
            2.0 * backward_bound::<f64>(&values, n) + n as f64 * tolerance * frobenius(&values);
        for (qr, jacobi) in computed.iter().zip(&reference.eigenvalues) {
            assert!(
                (qr - jacobi).abs() <= bound,
                "n={n}: {qr} vs {jacobi}, bound {bound}"
            );
        }
    }
}

fn check_resolves_rank_deficient_gram_matrices<T: RealScalar>() {
    // A rank-10 Gram matrix of order 60: fifty zero eigenvalues, the exact
    // geometry of a noise-free MP-PCA window.
    let (rank, n) = (10, 60);
    let gram = random_gram(rank, n, 29);
    let computed = assert_backward_stable::<T>(&gram, n);
    let bound = backward_bound::<T>(&gram, n);
    for &value in &computed[..n - rank] {
        assert!(
            value.abs() <= bound,
            "null eigenvalue {value} exceeds {bound}"
        );
    }
    let trace: f64 = (0..n).map(|i| gram[i * n + i]).sum();
    let signal: f64 = computed[n - rank..].iter().sum();
    assert!((signal - trace).abs() <= n as f64 * bound);
}

#[test]
fn symmetric_eigen_qr_resolves_rank_deficient_gram_matrices() {
    check_resolves_rank_deficient_gram_matrices::<f64>();
    check_resolves_rank_deficient_gram_matrices::<f32>();
    check_resolves_rank_deficient_gram_matrices::<F16>();
    check_resolves_rank_deficient_gram_matrices::<Bf16>();
}

fn check_sorts_a_diagonal_matrix_exactly<T: RealScalar>() {
    let diagonal = [4.0_f64, -2.0, 7.0, -2.0, 0.0];
    let n = diagonal.len();
    let mut values = vec![0.0; n * n];
    for (i, &d) in diagonal.iter().enumerate() {
        values[i * n + i] = d;
    }
    let (matrix, _) = round_into::<T>(&values, n);
    let eigen = symmetric_eigen_qr(&matrix.view()).unwrap();
    let expected = [-2.0_f64, -2.0, 0.0, 4.0, 7.0];
    // A diagonal input needs no reduction (its reflectors are all identity)
    // and the QL sweep never runs (every off-diagonal is already zero), so
    // sorting a set of exactly representable values is exact in every
    // format, not merely close.
    for (value, &expected) in eigen.eigenvalues.iter().zip(&expected) {
        assert_eq!(value.to_f64(), expected);
    }
}

#[test]
fn symmetric_eigen_qr_sorts_a_diagonal_matrix_exactly() {
    check_sorts_a_diagonal_matrix_exactly::<f64>();
    check_sorts_a_diagonal_matrix_exactly::<f32>();
    check_sorts_a_diagonal_matrix_exactly::<F16>();
    check_sorts_a_diagonal_matrix_exactly::<Bf16>();
}

#[test]
fn symmetric_eigen_workspace_reuse_is_bitwise_identical_to_a_fresh_solve() {
    let large_matrix = Array2::from_shape_vec([9, 9], random_symmetric(9, 3)).unwrap();
    let small_matrix = Array2::from_shape_vec([4, 4], random_symmetric(4, 5)).unwrap();

    let mut fresh = SymmetricEigenWorkspace::new();
    fresh.decompose(&small_matrix.view()).unwrap();
    let mut reused = SymmetricEigenWorkspace::new();
    reused.decompose(&large_matrix.view()).unwrap();
    reused.decompose(&small_matrix.view()).unwrap();

    assert_eq!(reused.order(), 4);
    assert_eq!(reused.eigenvalues(), fresh.eigenvalues());
    assert!(reused.eigenvectors().eq(fresh.eigenvectors()));
}

fn check_reads_only_the_lower_triangle_of_strided_views<T: RealScalar>() {
    // The even rows and columns form [[4, 1], [1, 4]] below the diagonal; the
    // upper entry is garbage the solver must not read.
    let raw = [
        4.0_f64, 0.0, 99.0, 0.0, 0.0, 9.0, 0.0, 8.0, 1.0, 0.0, 4.0, 0.0, 0.0, 8.0, 0.0, 9.0,
    ];
    let (matrix, _) = round_into::<T>(&raw, 4);
    let view = matrix
        .view()
        .slice_with::<2>(&[
            SliceArg::range(Some(0), None, 2),
            SliceArg::range(Some(0), None, 2),
        ])
        .unwrap();
    let eigen = symmetric_eigen_qr(&view).unwrap();
    // n²·ε(T)·‖A‖_F (`backward_bound`), n = 2, on the read [[4,1],[1,4]] block.
    let block = [4.0_f64, 1.0, 1.0, 4.0];
    let bound = backward_bound::<T>(&block, 2);
    let values: Vec<f64> = eigen.eigenvalues.iter().map(|v| v.to_f64()).collect();
    assert!((values[0] - 3.0).abs() <= bound, "{}", values[0]);
    assert!((values[1] - 5.0).abs() <= bound, "{}", values[1]);
}

#[test]
fn symmetric_eigen_qr_reads_only_the_lower_triangle_of_strided_views() {
    check_reads_only_the_lower_triangle_of_strided_views::<f64>();
    check_reads_only_the_lower_triangle_of_strided_views::<f32>();
    check_reads_only_the_lower_triangle_of_strided_views::<F16>();
    check_reads_only_the_lower_triangle_of_strided_views::<Bf16>();
}

fn check_handles_degenerate_orders<T: RealScalar>() {
    let empty = Array2::<T>::from_shape_vec([0, 0], vec![]).unwrap();
    let eigen = symmetric_eigen_qr(&empty.view()).unwrap();
    assert!(eigen.eigenvalues.is_empty());

    let zero = Array2::from_shape_vec([2, 2], vec![T::ZERO; 4]).unwrap();
    let zero_eigen = symmetric_eigen_qr(&zero.view()).unwrap();
    assert_eq!(zero_eigen.eigenvalues.len(), 2);
    assert!(zero_eigen.eigenvalues.iter().all(|&v| v == T::ZERO));

    let single = Array2::from_shape_vec([1, 1], vec![T::from_f64(-3.5)]).unwrap();
    let mut workspace = SymmetricEigenWorkspace::new();
    workspace.decompose(&single.view()).unwrap();
    assert_eq!(workspace.eigenvalues(), &[T::from_f64(-3.5)]);
    assert_eq!(workspace.eigenvectors().next().unwrap(), &[T::ONE]);
}

#[test]
fn symmetric_eigen_qr_handles_degenerate_orders() {
    check_handles_degenerate_orders::<f64>();
    check_handles_degenerate_orders::<f32>();
    check_handles_degenerate_orders::<F16>();
    check_handles_degenerate_orders::<Bf16>();
}

#[test]
fn symmetric_eigen_qr_rejects_a_non_square_matrix() {
    let rectangular = Array2::from_shape_vec([2, 3], vec![1.0_f64; 6]).unwrap();
    match symmetric_eigen_qr(&rectangular.view()) {
        Err(LetoError::ShapeMismatch { lhs, rhs }) => {
            assert_eq!((lhs, rhs), (vec![2, 3], vec![2, 2]));
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn symmetric_eigen_qr_resolves_an_asymmetric_input_by_the_lower_triangle() {
    let lower = [2.0_f64, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 4.0];
    let mut asymmetric = lower;
    asymmetric[1] = 7.0; // upper entry (0, 1): never read
    let from_lower = symmetric_eigen_qr(
        &Array2::from_shape_vec([3, 3], lower.to_vec())
            .unwrap()
            .view(),
    )
    .unwrap();
    let resolved = symmetric_eigen_qr(
        &Array2::from_shape_vec([3, 3], asymmetric.to_vec())
            .unwrap()
            .view(),
    )
    .unwrap();
    assert_eq!(resolved.eigenvalues, from_lower.eigenvalues);
}
