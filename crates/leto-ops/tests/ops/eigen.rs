#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::LetoError;
use leto::{Array2, SliceArg, Storage};
use leto_ops::{symmetric_eigen_jacobi, symmetric_eigenvalues_jacobi};

fn assert_close(lhs: f64, rhs: f64, epsilon: f64) {
    assert!(
        (lhs - rhs).abs() <= epsilon,
        "left {lhs} differs from right {rhs}"
    );
}

fn column_norm(values: &[f64], n: usize, col: usize) -> f64 {
    (0..n)
        .map(|row| values[row * n + col] * values[row * n + col])
        .sum::<f64>()
        .sqrt()
}

fn column_dot(values: &[f64], n: usize, lhs: usize, rhs: usize) -> f64 {
    (0..n)
        .map(|row| values[row * n + lhs] * values[row * n + rhs])
        .sum()
}

#[test]
fn symmetric_eigen_jacobi_solves_known_two_by_two_matrix() {
    let matrix = Array2::from_shape_vec([2, 2], vec![2.0, 1.0, 1.0, 2.0]).unwrap();
    let decomposition = symmetric_eigen_jacobi(&matrix.view()).unwrap();

    assert_close(decomposition.eigenvalues[0], 1.0, 1.0e-12);
    assert_close(decomposition.eigenvalues[1], 3.0, 1.0e-12);
    let eigenvectors = decomposition.eigenvectors.storage().as_slice();
    assert_close(column_norm(eigenvectors, 2, 0), 1.0, 1.0e-12);
    assert_close(column_norm(eigenvectors, 2, 1), 1.0, 1.0e-12);
    assert_close(column_dot(eigenvectors, 2, 0, 1), 0.0, 1.0e-12);
}

#[test]
fn symmetric_eigen_jacobi_accepts_strided_symmetric_view() {
    let matrix = Array2::from_shape_vec(
        [4, 4],
        vec![
            4.0, 0.0, 1.0, 0.0, 0.0, 9.0, 0.0, 8.0, 1.0, 0.0, 4.0, 0.0, 0.0, 8.0, 0.0, 9.0,
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

    let decomposition = symmetric_eigen_jacobi(&view).unwrap();
    assert_close(decomposition.eigenvalues[0], 3.0, 1.0e-12);
    assert_close(decomposition.eigenvalues[1], 5.0, 1.0e-12);
}

#[test]
fn symmetric_eigenvalues_jacobi_matches_full_decomposition_without_vectors() {
    let matrix =
        Array2::from_shape_vec([3, 3], vec![3.0, 2.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 7.0]).unwrap();

    let eigenvalues = symmetric_eigenvalues_jacobi(&matrix.view()).unwrap();
    let decomposition = symmetric_eigen_jacobi(&matrix.view()).unwrap();

    assert_eq!(eigenvalues.len(), 3);
    for (actual, expected) in eigenvalues.iter().zip(decomposition.eigenvalues.iter()) {
        assert_close(*actual, *expected, 1.0e-12);
    }
    assert_close(eigenvalues[0], 1.0, 1.0e-12);
    assert_close(eigenvalues[1], 5.0, 1.0e-12);
    assert_close(eigenvalues[2], 7.0, 1.0e-12);
}

#[test]
fn symmetric_eigenvalues_jacobi_accepts_strided_symmetric_view() {
    let matrix = Array2::from_shape_vec(
        [4, 4],
        vec![
            4.0, 0.0, 1.0, 0.0, 0.0, 9.0, 0.0, 8.0, 1.0, 0.0, 4.0, 0.0, 0.0, 8.0, 0.0, 9.0,
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

    let eigenvalues = symmetric_eigenvalues_jacobi(&view).unwrap();
    assert_close(eigenvalues[0], 3.0, 1.0e-12);
    assert_close(eigenvalues[1], 5.0, 1.0e-12);
}

#[test]
fn symmetric_eigen_jacobi_matches_path_graph_laplacian_closed_form() {
    let matrix = Array2::from_shape_vec(
        [3, 3],
        vec![1.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 1.0],
    )
    .unwrap();
    let decomposition = symmetric_eigen_jacobi(&matrix.view()).unwrap();
    let expected = [0.0, 1.0, 3.0];

    for (actual, expected) in decomposition.eigenvalues.iter().zip(expected.iter()) {
        assert_close(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn symmetric_eigen_jacobi_is_generic_over_f32() {
    // Same 2x2 as the f64 case, exercising the generic path at f32 precision.
    let matrix = Array2::from_shape_vec([2, 2], vec![2.0f32, 1.0, 1.0, 2.0]).unwrap();
    let decomposition = symmetric_eigen_jacobi(&matrix.view()).unwrap();

    assert!((decomposition.eigenvalues[0] - 1.0f32).abs() <= 1.0e-5);
    assert!((decomposition.eigenvalues[1] - 3.0f32).abs() <= 1.0e-5);
    let eigenvectors = decomposition.eigenvectors.storage().as_slice();
    let norm0 = column_norm32(eigenvectors, 2, 0);
    assert!((norm0 - 1.0f32).abs() <= 1.0e-5);
}

fn column_norm32(values: &[f32], n: usize, col: usize) -> f32 {
    (0..n)
        .map(|row| values[row * n + col] * values[row * n + col])
        .sum::<f32>()
        .sqrt()
}

#[test]
fn symmetric_eigen_jacobi_rejects_invalid_inputs() {
    let rectangular = Array2::from_shape_vec([2, 3], vec![1.0f64; 6]).unwrap();
    assert!(symmetric_eigen_jacobi(&rectangular.view()).is_err());
    assert!(symmetric_eigenvalues_jacobi(&rectangular.view()).is_err());

    let asymmetric = Array2::from_shape_vec([2, 2], vec![1.0f64, 2.0, 3.0, 4.0]).unwrap();
    assert!(symmetric_eigen_jacobi(&asymmetric.view()).is_err());
    assert!(symmetric_eigenvalues_jacobi(&asymmetric.view()).is_err());

    let non_finite = Array2::from_shape_vec([2, 2], vec![1.0, f64::NAN, f64::NAN, 1.0]).unwrap();
    assert!(matches!(
        symmetric_eigen_jacobi(&non_finite.view()),
        Err(LetoError::InvalidInput(_))
    ));
    assert!(matches!(
        symmetric_eigenvalues_jacobi(&non_finite.view()),
        Err(LetoError::InvalidInput(_))
    ));
}

#[test]
fn symmetric_eigen_jacobi_resolves_small_magnitude_matrices() {
    // Regression: an absolute 1e-12 tolerance accepted this matrix unrotated
    // and returned {2e-13, 2e-13}. Eigenvalues 1e-13 and 3e-13.
    let matrix = Array2::from_shape_vec([2, 2], vec![2e-13_f64, 1e-13, 1e-13, 2e-13]).unwrap();
    let decomposition = symmetric_eigen_jacobi(&matrix.view()).unwrap();
    // One rotation of a 2×2: a few roundings of each entry, 4ε relative.
    assert!((decomposition.eigenvalues[0] / 1e-13 - 1.0).abs() <= 4.0 * f64::EPSILON);
    assert!((decomposition.eigenvalues[1] / 3e-13 - 1.0).abs() <= 4.0 * f64::EPSILON);
}

#[test]
fn symmetric_eigen_jacobi_accuracy_is_invariant_under_scaling() {
    // [[2,1,1],[1,2,1],[1,1,2]] has eigenvalues {1, 1, 4}; scaled by s they
    // are s·{1, 1, 4} to the same relative accuracy at every magnitude the
    // format holds with its squares (Jacobi forms none beyond ‖A‖_F, which is
    // computed scaled). Bound: 32·n²·ε relative, the rotation cap times one
    // rounding each.
    let base = [2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0];
    let bound = 32.0 * 9.0 * f64::EPSILON;
    for s in [1e-300_f64, 1e-170, 1e-13, 1.0, 1e13, 1e170, 1e300] {
        let matrix = Array2::from_shape_vec([3, 3], base.iter().map(|v| v * s).collect()).unwrap();
        let values = symmetric_eigenvalues_jacobi(&matrix.view()).unwrap();
        for (value, expected) in values.iter().zip([1.0, 1.0, 4.0]) {
            assert!(
                (value / s - expected).abs() <= bound,
                "s = {s:e}: {value:e}"
            );
        }
    }
}

/// Seeded `n × n` symmetric matrix with entries uniform in `[-1, 1)`, plus the
/// same shape rank-deficient: `XᵀX` of a `rank × n` seeded `X`.
fn seeded_pair(n: usize, rank: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = leto_ops::Xorshift64::new(seed);
    let mut dense = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let value = 2.0 * rng.next_unit_f64() - 1.0;
            dense[i * n + j] = value;
            dense[j * n + i] = value;
        }
    }
    let x: Vec<f64> = (0..rank * n).map(|_| rng.next_unit_f64() - 0.5).collect();
    let mut gram = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            gram[i * n + j] = (0..rank).map(|r| x[r * n + i] * x[r * n + j]).sum();
        }
    }
    (dense, gram)
}

#[test]
fn symmetric_eigen_jacobi_default_tolerance_is_reachable() {
    // The ε default must converge inside the 32·n² budget on dense and
    // rank-deficient 60 × 60 inputs, in both native formats.
    let n = 60;
    let (dense, gram) = seeded_pair(n, 10, 41);
    for values in [&dense, &gram] {
        let wide = Array2::from_shape_vec([n, n], values.clone()).unwrap();
        assert_eq!(symmetric_eigenvalues_jacobi(&wide.view()).unwrap().len(), n);
        let narrow: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        let narrow = Array2::from_shape_vec([n, n], narrow).unwrap();
        assert_eq!(
            symmetric_eigenvalues_jacobi(&narrow.view()).unwrap().len(),
            n
        );
    }
}

#[test]
fn symmetric_eigen_jacobi_accepts_rounding_level_asymmetry() {
    // Regression: apollo-gft builds adjacency weights along two rounding
    // paths; 0.1 + 0.2 and 0.3 differ by one ulp (5.55e-17). The weighted
    // triangle [[0, w, 1], [w, 0, 1], [1, 1, 0]] with w = 0.3 has eigenvalues
    // −w and (w ± √(w² + 8))/2.
    let w = 0.3_f64;
    let matrix = Array2::from_shape_vec(
        [3, 3],
        vec![0.0, 0.1 + 0.2, 1.0, 0.3, 0.0, 1.0, 1.0, 1.0, 0.0],
    )
    .unwrap();
    let values = symmetric_eigenvalues_jacobi(&matrix.view()).unwrap();
    let root = (w * w + 8.0).sqrt();
    let expected = [(w - root) / 2.0, -w, (w + root) / 2.0];
    // Weyl: the one-ulp asymmetry, the rotations' backward error n²·ε·‖A‖_F,
    // and the stopping remainder n·ε·‖A‖_F, with ‖A‖_F = √(2w² + 4) < 2.1.
    let bound = (9.0 + 3.0) * f64::EPSILON * 2.1 + 5.6e-17;
    for (value, expected) in values.iter().zip(expected) {
        assert!((value - expected).abs() <= bound, "{value} vs {expected}");
    }

    // An asymmetry far beyond rounding is still rejected.
    let skewed = Array2::from_shape_vec([2, 2], vec![1.0_f64, 0.5, 0.5 + 1e-9, 1.0]).unwrap();
    assert!(matches!(
        symmetric_eigenvalues_jacobi(&skewed.view()),
        Err(LetoError::InvalidInput(_))
    ));
}

/// `Q·D·Qᵀ` assembled entry by entry, each entry an `n`-term sum evaluated
/// along its own rounding path: symmetric in exact arithmetic only. `Q` is the
/// product of two seeded Householder reflectors; `D` is seeded in `[-1, 1)`
/// times `scale`. Returns the matrix and `D` sorted ascending.
fn accumulated_qdqt(n: usize, seed: u64, scale: f64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = leto_ops::Xorshift64::new(seed);
    let mut reflector = || {
        let v: Vec<f64> = (0..n).map(|_| rng.next_unit_f64() - 0.5).collect();
        let beta = 2.0 / v.iter().map(|x| x * x).sum::<f64>();
        move |i: usize, j: usize| f64::from(u8::from(i == j)) - beta * v[i] * v[j]
    };
    let (h1, h2) = (reflector(), reflector());
    let mut q = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            q[i * n + j] = (0..n).map(|k| h1(i, k) * h2(k, j)).sum();
        }
    }
    let d: Vec<f64> = (0..n)
        .map(|_| scale * (2.0 * rng.next_unit_f64() - 1.0))
        .collect();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = (0..n).map(|k| q[i * n + k] * d[k] * q[j * n + k]).sum();
        }
    }
    let mut sorted = d;
    sorted.sort_by(f64::total_cmp);
    (a, sorted)
}

#[test]
fn symmetric_eigen_jacobi_accepts_accumulated_rounding_asymmetry() {
    // Regression: the 2ε·max(|aᵢⱼ|, |aⱼᵢ|, ‖A‖_F/n) bound rejected 2–8 of 25
    // such matrices at n = 30 and 60. Each entry's two rounding paths differ by
    // at most 2γ_{n+2}·‖A‖_F, inside the accepted (n + 2)·ε·‖A‖_F.
    for n in [30_usize, 60] {
        for scale in [1.0, 1e-8] {
            for seed in 0..25_u64 {
                let (a, spectrum) = accumulated_qdqt(n, 1000 + seed, scale);
                let norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
                let matrix = Array2::from_shape_vec([n, n], a).unwrap();
                let values = symmetric_eigenvalues_jacobi(&matrix.view())
                    .unwrap_or_else(|error| panic!("n={n}, scale={scale:e}, seed={seed}: {error}"));
                // Weyl: accepted asymmetry n(n+2)·ε·‖A‖_F, rotations n²·ε·‖A‖_F,
                // stopping remainder n·ε·‖A‖_F, and the construction's own
                // 2γ_{n+2}·‖A‖_F per entry (n(n+2)·ε·‖A‖_F in Frobenius norm).
                let nf = n as f64;
                let bound = (2.0 * nf * (nf + 2.0) + nf * nf + nf) * f64::EPSILON * norm;
                for (value, expected) in values.iter().zip(&spectrum) {
                    assert!(
                        (value - expected).abs() <= bound,
                        "n={n}, seed={seed}: {value} vs {expected}"
                    );
                }
            }
        }
    }

    // apollo-gft adjacency with a 1e-15 (≈ 4.5 ulp) asymmetric weight, which
    // the 2ε·max bound rejected: (n + 2)·ε·‖A‖_F = 5·ε·√6 ≈ 2.7e-15.
    let adjacency = Array2::from_shape_vec(
        [3, 3],
        vec![0.0_f64, 1.0, 1.0, 1.0 + 1e-15, 0.0, 1.0, 1.0, 1.0, 0.0],
    )
    .unwrap();
    let values = symmetric_eigenvalues_jacobi(&adjacency.view()).unwrap();
    // The triangle graph's adjacency has eigenvalues {−1, −1, 2}. Weyl: the
    // 1e-15 asymmetry, rotations n²·ε·‖A‖_F and remainder n·ε·‖A‖_F.
    let bound = 1e-15 + 12.0 * f64::EPSILON * 6.0_f64.sqrt();
    for (value, expected) in values.iter().zip([-1.0, -1.0, 2.0]) {
        assert!((value - expected).abs() <= bound, "{value} vs {expected}");
    }
}
