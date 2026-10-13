#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::seeded_pair;
use leto::{Array2, LetoError, Storage};
use leto_ops::{symmetric_eigen_jacobi, symmetric_eigenvalues_jacobi};

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

#[test]
fn symmetric_eigen_jacobi_resolves_an_accepted_asymmetry_by_the_upper_triangle() {
    // 0.5 + 1e-15 at (2, 0) is within the symmetry bound
    // (n + 2)·ε·‖A‖_F ≈ 6.5e-15. The first rotation, on the largest pair (0, 1),
    // reads column entries such as (2, 0); the solver works on the upper
    // triangle mirrored, so the result is bitwise that of the exactly
    // symmetric matrix built from the upper triangle.
    let upper = [2.0_f64, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 4.0];
    let mut asymmetric = upper;
    asymmetric[6] = 0.5 + 1e-15;
    let from_upper = symmetric_eigen_jacobi(
        &Array2::from_shape_vec([3, 3], upper.to_vec())
            .unwrap()
            .view(),
    )
    .unwrap();
    let resolved = symmetric_eigen_jacobi(
        &Array2::from_shape_vec([3, 3], asymmetric.to_vec())
            .unwrap()
            .view(),
    )
    .unwrap();
    assert_eq!(resolved.eigenvalues, from_upper.eigenvalues);
    assert_eq!(
        resolved.eigenvectors.storage().as_slice(),
        from_upper.eigenvectors.storage().as_slice()
    );
}
