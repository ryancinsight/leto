//! Symmetric tridiagonal-QL eigensolver: closed forms, backward-error
//! residuals over every supported scalar, clusters, dynamic range, the full
//! exponent range of each format, and differential agreement with Jacobi.
//!
//! # Error bounds
//!
//! The computed decomposition is exact for `A + E` with `‖E‖_F ≤ η·‖A‖_F`,
//! and each computed eigenvector is within `η_Q` of the corresponding column
//! of that exact orthogonal basis — `η` and `η_Q` derived in
//! `backward_error.rs` ([`ql`](super::backward_error::ql),
//! [`ql_vectors`](super::backward_error::ql_vectors)) from Higham's `γ`
//! bounds over the enumerated reflectors, rotations, shifts and deflations,
//! at the code's sweep cap (`30·n`), `ε` the machine epsilon of the scalar
//! the solver runs in. Then:
//!
//! - eigenvalues (Weyl): `|λ̂ᵢ − λᵢ| ≤ η·‖A‖_F`;
//! - residuals: `‖A·v̂ⱼ − λ̂ⱼ·v̂ⱼ‖₂ ≤ ‖E‖₂ + (‖A‖₂ + |λⱼ|)·η_Q ≤
//!   (η + 2η_Q(1 + η))·‖A‖_F`, plus the `f64` evaluation's
//!   `γ_{n+2}·2‖A‖_F`;
//! - orthonormality: `|v̂ᵢᵀv̂ⱼ − δᵢⱼ| ≤ 2η_Q + η_Q²`, plus `γ_n`.
//!
//! Where the reference spectrum belongs to the `f64` matrix before rounding
//! it into `T`, the rounding adds `ε/2·‖A‖_F` (entrywise relative `ε/2`).
//!
//! These a-priori bounds are worst cases at the cap and are asserted only
//! where informative (`η < 1`, [`informative`](super::backward_error::informative));
//! they are vacuous for `F16` and `Bf16`. Every format's eigenvalues are also
//! checked a posteriori: the returned eigenpairs certify, through the
//! measured residual and orthogonality
//! ([`symmetric_certificate`](super::backward_error::symmetric_certificate)),
//! that each `λ̂ᵢ` is within a computed radius of `λᵢ(Â)`; the Jacobi
//! reference is certified the same way.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::backward_error::{ql_vectors, symmetric_certificate};
use super::format::epsilon;
use eunomia::{Bf16, F16};
use fixtures::{
    assert_backward_stable, assert_spectrum, backward_bound, certified, random_gram,
    random_symmetric, round_into, with_spectrum,
};
use leto::{Array2, LetoError, SliceArg, Storage};
use leto_ops::{
    symmetric_eigen_jacobi_with_tolerance, symmetric_eigen_qr, RealScalar, SymmetricEigenWorkspace,
};

mod fixtures;
mod scale;

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
    // The derived eigenvalue bound for n = 2 (module documentation), on the
    // exact image `T` holds.
    // The a-posteriori radius, and the a-priori bound where informative.
    let bounds = [
        Some(certified(&image, &eigen)),
        backward_bound::<T>(&image, 2),
    ];
    // Each eigenvector is within η_Q of the exact one, ±(1, 1)/√2 here.
    let orthogonality = ql_vectors(2, epsilon::<T>());
    let values: Vec<f64> = eigen.eigenvalues.iter().map(|v| v.to_f64()).collect();
    for bound in bounds.into_iter().flatten() {
        assert!((values[0] - 1.0).abs() <= bound, "{}", values[0]);
        assert!((values[1] - 3.0).abs() <= bound, "{}", values[1]);
    }
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
        let (computed, radius) = assert_backward_stable::<f64>(&values, n);
        let matrix = Array2::from_shape_vec([n, n], values.clone()).unwrap();
        let reference = symmetric_eigen_jacobi_with_tolerance(&matrix.view(), tolerance).unwrap();
        // The Jacobi reference is certified from its own eigenbasis.
        let basis = reference.eigenvectors.storage().as_slice();
        let bound = radius + symmetric_certificate(&values, &reference.eigenvalues, basis, n);
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
    let (computed, bound) = assert_backward_stable::<T>(&gram, n);
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
    // On the read [[4,1],[1,4]] block: the a-posteriori radius, and the
    // a-priori bound where informative.
    let block = [4.0_f64, 1.0, 1.0, 4.0];
    let bounds = [
        Some(certified(&block, &eigen)),
        backward_bound::<T>(&block, 2),
    ];
    let values: Vec<f64> = eigen.eigenvalues.iter().map(|v| v.to_f64()).collect();
    for bound in bounds.into_iter().flatten() {
        assert!((values[0] - 3.0).abs() <= bound, "{}", values[0]);
        assert!((values[1] - 5.0).abs() <= bound, "{}", values[1]);
    }
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
