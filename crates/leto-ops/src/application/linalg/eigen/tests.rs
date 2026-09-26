use super::rotation::{diagonalize_within, NoEigenvectors};
use super::{symmetric_eigen_jacobi, symmetric_eigenvalues_jacobi};
use leto::{Array2, LetoError};

#[test]
fn jacobi_typed_overflow_replaces_a_wrong_ok_near_max() {
    // M = 0.75·MAX: [[M, M], [M, M]] has true eigenvalues {0, 2M}. Since
    // M > MAX/2, `2M` itself is not representable in f64 — the correct
    // outcome is a typed `Overflow`, never a finite answer. Before
    // balancing (finding LETO-DENSE-SCALE-RANGE-2026-09-24, item J) the
    // unscaled rotation overflowed internally and returned the wrong
    // `Ok([M, M])` instead (neither eigenvalue is `M`).
    let m = 0.75 * f64::MAX;
    let a = Array2::from_shape_vec([2, 2], vec![m, m, m, m]).expect("2x2");
    assert!(matches!(
        symmetric_eigen_jacobi(&a.view()),
        Err(LetoError::Overflow { .. })
    ));
    // The eigenvalues-only path (`symmetric_eigenvalues_jacobi`) scales
    // independently of the full-decomposition path: this kills the
    // mutant that drops its own balancing call.
    assert!(matches!(
        symmetric_eigenvalues_jacobi(&a.view()),
        Err(LetoError::Overflow { .. })
    ));
}

#[test]
fn exhausted_rotation_budget_is_a_typed_convergence_error() {
    // [[2,1,1],[1,2,1],[1,1,2]]: the first rotation (pivot (0,1), equal
    // diagonals, θ = π/4) zeroes a₀₁ and a₀₂ = (1 − 1)/√2 but leaves
    // a₁₂ = (1 + 1)/√2 = √2, so one rotation stops at residual √2/‖A‖_F
    // = √2/√18 = 1/3 (a few roundings, 8ε relative).
    let mut a = [2.0_f64, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0];
    let result = diagonalize_within(&mut a, 3, 1e-12, 1, &mut NoEigenvectors);
    let Err(LetoError::ConvergenceError {
        max_iters,
        residual,
        tol,
    }) = result
    else {
        panic!("expected ConvergenceError, got {result:?}");
    };
    assert_eq!(max_iters, 1);
    assert_eq!(tol, 1e-12);
    let expected = 1.0 / 3.0;
    assert!(
        (residual / expected - 1.0).abs() <= 8.0 * f64::EPSILON,
        "{residual}"
    );
}
