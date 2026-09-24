//! Symmetric tridiagonal-QL eigensolver: closed forms, backward-error
//! residuals over every supported scalar, clusters, dynamic range, the full
//! exponent range of each format, and differential agreement with Jacobi.
//!
//! # Tolerance derivation
//!
//! Householder tridiagonalization and QL apply `O(n)` orthogonal
//! transformations, each with backward error `O(n)·ε·‖A‖`, so the computed
//! decomposition is exact for `A + E` with `‖E‖₂ ≤ ‖E‖_F ≤ n²·ε·‖A‖_F`
//! (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Lemma
//! 19.3 with `r = n` transformations of `γ̃_n` each), `ε` the machine epsilon
//! of the scalar the solver runs in. Weyl's inequality turns that into
//! `|λ̂ᵢ − λᵢ| ≤ n²·ε·‖A‖_F`; the residual `‖A v̂ − λ̂ v̂‖₂` obeys the same bound
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

use eunomia::{Bf16, F16};
use leto::{Array2, LetoError, SliceArg, Storage};
use leto_ops::{
    symmetric_eigen_jacobi_with_tolerance, symmetric_eigen_qr, RealScalar, SymmetricEigenWorkspace,
    Xorshift64,
};

/// A supported scalar with the exponent range of its normal numbers.
trait Format: RealScalar {
    /// Binary exponent of the smallest positive normal value.
    const MIN_EXPONENT: i32;
    /// Binary exponent of the largest finite value.
    const MAX_EXPONENT: i32;
}

impl Format for f64 {
    const MIN_EXPONENT: i32 = -1022;
    const MAX_EXPONENT: i32 = 1023;
}
impl Format for f32 {
    const MIN_EXPONENT: i32 = -126;
    const MAX_EXPONENT: i32 = 127;
}
impl Format for F16 {
    const MIN_EXPONENT: i32 = -14;
    const MAX_EXPONENT: i32 = 15;
}
impl Format for Bf16 {
    const MIN_EXPONENT: i32 = -126;
    const MAX_EXPONENT: i32 = 127;
}

/// Machine epsilon of `T`, found through `T`'s own addition.
fn epsilon<T: RealScalar>() -> f64 {
    let half = T::from_f64(0.5);
    let mut e = T::ONE;
    while T::ONE.add(e.mul(half)) > T::ONE {
        e = e.mul(half);
    }
    e.to_f64()
}

/// Round `values` into `T`, returning the `T` matrix and the exact `f64`
/// image of what `T` holds.
fn round_into<T: RealScalar>(values: &[f64], n: usize) -> (Array2<T>, Vec<f64>) {
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
fn with_spectrum(spectrum: &[f64], seed: u64) -> Vec<f64> {
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
fn backward_bound<T: RealScalar>(values: &[f64], n: usize) -> f64 {
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
fn assert_spectrum<T: RealScalar>(values: &[f64], spectrum: &[f64]) {
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

fn check_dynamic_range<T: RealScalar>() {
    // Eigenvalues across twelve decades: the bound is normwise, so the small
    // ones are resolved to n²·ε·‖A‖_F absolutely, not relatively.
    let spectrum = [1e-12, 1e-9, 1e-6, 1e-3, 1e-1, 1.0];
    assert_spectrum::<T>(&with_spectrum(&spectrum, 23), &spectrum);
}

/// `s·[[2,1,1],[1,2,1],[1,1,2]]` (eigenvalues `s·{1, 1, 4}`) at every
/// power-of-two scale `T` represents: a normal-range scale must decompose
/// correctly; a scale whose entries or eigenvalues leave the normal range
/// must decompose correctly or fail with a typed error, never a wrong `Ok`.
fn check_scale_sweep<T: Format>() {
    let base = [2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0];
    let n = 3;
    // Relative bound: the backward bound of the unscaled matrix over ‖A‖_F
    // scales with s exactly; entries 2s and s are exact in every format.
    let relative = backward_bound::<T>(&base, n);
    // From seven binades into the subnormals (Bf16 has seven below its
    // smallest normal, the fewest of the four) to the largest scale whose
    // entries 2s stay finite; 4s then overflows at the top, which must be a
    // typed error.
    let (low, high) = (T::MIN_EXPONENT - 7, T::MAX_EXPONENT);
    for exponent in low..high {
        let scale = T::ONE.scale_binary(exponent);
        let values: Vec<T> = base.iter().map(|&v| T::from_f64(v).mul(scale)).collect();
        let matrix = Array2::from_shape_vec([n, n], values).unwrap();
        // Normal range: 2s at most the largest value's exponent less the
        // headroom for λ = 4s (two binades), s at least the smallest normal.
        let normal = exponent >= T::MIN_EXPONENT && exponent + 2 <= T::MAX_EXPONENT;
        match symmetric_eigen_qr(&matrix.view()) {
            Ok(eigen) => {
                let s = scale.to_f64();
                for (value, expected) in eigen.eigenvalues.iter().zip([1.0, 1.0, 4.0]) {
                    let error = (value.to_f64() / s - expected).abs();
                    // The entries are powers of two and exact even when
                    // subnormal; scaling is exact, so the only extra error
                    // is rounding each computed eigenvalue back onto the
                    // subnormal grid, whose spacing is 2^MIN_EXPONENT·ε: half
                    // of it, relative to s = 2^exponent.
                    let lost = if exponent < T::MIN_EXPONENT {
                        0.5 * 2.0_f64.powi(T::MIN_EXPONENT - exponent) * epsilon::<T>()
                    } else {
                        0.0
                    };
                    assert!(
                        error <= relative + lost,
                        "2^{exponent}: {:e} vs {expected}·s",
                        value.to_f64()
                    );
                }
            }
            Err(error) => {
                assert!(!normal, "2^{exponent}: normal-range scale failed: {error}");
                assert!(
                    matches!(
                        error,
                        LetoError::Overflow { .. } | LetoError::ConvergenceError { .. }
                    ),
                    "2^{exponent}: untyped failure {error:?}"
                );
            }
        }
    }
}

fn check_non_finite_input<T: RealScalar>() {
    for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let values: Vec<T> = [1.0, 0.0, bad, 1.0]
            .iter()
            .map(|&v| T::from_f64(v))
            .collect();
        let matrix = Array2::from_shape_vec([2, 2], values).unwrap();
        let mut workspace = SymmetricEigenWorkspace::new();
        match workspace.decompose(&matrix.view()) {
            Err(LetoError::InvalidInput(reason)) => assert!(reason.contains("(1, 0)"), "{reason}"),
            other => panic!("{bad}: expected InvalidInput, got {other:?}"),
        }
        assert_eq!(workspace.order(), 0);
    }
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

#[test]
fn symmetric_eigen_qr_spans_a_wide_dynamic_range_for_every_scalar() {
    check_dynamic_range::<f64>();
    check_dynamic_range::<f32>();
    check_dynamic_range::<F16>();
    check_dynamic_range::<Bf16>();
}

#[test]
fn symmetric_eigen_qr_is_correct_or_typed_across_each_exponent_range() {
    check_scale_sweep::<f64>();
    check_scale_sweep::<f32>();
    check_scale_sweep::<F16>();
    check_scale_sweep::<Bf16>();
}

#[test]
fn symmetric_eigen_qr_rejects_non_finite_input_for_every_scalar() {
    check_non_finite_input::<f64>();
    check_non_finite_input::<f32>();
    check_non_finite_input::<F16>();
    check_non_finite_input::<Bf16>();
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
    let spectrum: Vec<f64> = (0..n)
        .map(|k| 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / n as f64).cos())
        .collect();
    assert_spectrum::<f64>(&laplacian, &spectrum);
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

#[test]
fn symmetric_eigen_qr_resolves_rank_deficient_gram_matrices() {
    // A rank-10 Gram matrix of order 60: fifty zero eigenvalues, the exact
    // geometry of a noise-free MP-PCA window.
    let (rank, n) = (10, 60);
    let gram = random_gram(rank, n, 29);
    let computed = assert_backward_stable::<f64>(&gram, n);
    let bound = backward_bound::<f64>(&gram, n);
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
fn symmetric_eigen_qr_handles_degenerate_orders() {
    let empty = Array2::<f64>::from_shape_vec([0, 0], vec![]).unwrap();
    let eigen = symmetric_eigen_qr(&empty.view()).unwrap();
    assert!(eigen.eigenvalues.is_empty());

    let zero = Array2::from_shape_vec([2, 2], vec![0.0_f64; 4]).unwrap();
    assert_eq!(
        symmetric_eigen_qr(&zero.view()).unwrap().eigenvalues,
        vec![0.0, 0.0]
    );

    let single = Array2::from_shape_vec([1, 1], vec![-3.5]).unwrap();
    let mut workspace = SymmetricEigenWorkspace::new();
    workspace.decompose(&single.view()).unwrap();
    assert_eq!(workspace.eigenvalues(), &[-3.5]);
    assert_eq!(workspace.eigenvectors().next().unwrap(), &[1.0]);
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
