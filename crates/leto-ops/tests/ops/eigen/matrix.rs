#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::{column_dot, column_norm};
use leto::{Array2, LetoError};
use leto_ops::{symmetric_eigen_jacobi, MatrixDecompose};

#[test]
fn matrix_symmetric_eigen_routes_through_ql() {
    // Path-graph Laplacian: eigenvalues 0, 1, 3.
    let matrix = Array2::from_shape_vec(
        [3, 3],
        vec![1.0_f64, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 1.0],
    )
    .unwrap();
    let decomposition = matrix.symmetric_eigen().unwrap();
    // Backward-error envelope n²·ε·‖A‖_F, n = 3, ‖A‖_F = 3.
    let bound = 9.0 * f64::EPSILON * 3.0;
    for (value, expected) in decomposition.eigenvalues.iter().zip([0.0, 1.0, 3.0]) {
        assert!((value - expected).abs() <= bound, "{value} vs {expected}");
    }
    assert!(decomposition
        .eigenvalues
        .windows(2)
        .all(|pair| pair[0] <= pair[1]));
    let vectors = decomposition.eigenvectors.as_slice().unwrap();
    for col in 0..3 {
        assert!((column_norm(vectors, 3, col) - 1.0).abs() <= 1.0e-12);
    }
    assert!(column_dot(vectors, 3, 0, 1).abs() <= 1.0e-12);
    assert!(column_dot(vectors, 3, 0, 2).abs() <= 1.0e-12);
    assert!(column_dot(vectors, 3, 1, 2).abs() <= 1.0e-12);
}

#[test]
fn matrix_symmetric_eigen_rejects_invalid_inputs() {
    let rectangular = Array2::from_shape_vec([2, 3], vec![1.0f64; 6]).unwrap();
    assert!(rectangular.symmetric_eigen().is_err());

    let asymmetric = Array2::from_shape_vec([2, 2], vec![1.0f64, 2.0, 3.0, 4.0]).unwrap();
    assert!(matches!(
        asymmetric.symmetric_eigen(),
        Err(LetoError::InvalidInput(_))
    ));

    let non_finite = Array2::from_shape_vec([2, 2], vec![1.0, f64::NAN, f64::NAN, 1.0]).unwrap();
    assert!(matches!(
        non_finite.symmetric_eigen(),
        Err(LetoError::InvalidInput(_))
    ));
}

#[test]
fn matrix_symmetric_eigen_matches_jacobi_on_accepted_asymmetry() {
    // Same one-ulp-asymmetry triangle as the Jacobi acceptance test: the
    // routed method must resolve it by the upper triangle identically.
    let matrix = Array2::from_shape_vec(
        [3, 3],
        vec![0.0_f64, 0.1 + 0.2, 1.0, 0.3, 0.0, 1.0, 1.0, 1.0, 0.0],
    )
    .unwrap();
    let via_method = matrix.symmetric_eigen().unwrap();
    let via_jacobi = symmetric_eigen_jacobi(&matrix.view()).unwrap();
    // Same resolved matrix, two backward-stable kernels: Weyl plus both
    // n²·ε·‖A‖_F envelopes with ‖A‖_F < 2.1.
    let bound = 2.0 * 9.0 * f64::EPSILON * 2.1;
    for (method, jacobi) in via_method
        .eigenvalues
        .iter()
        .zip(via_jacobi.eigenvalues.iter())
    {
        assert!((method - jacobi).abs() <= bound, "{method} vs {jacobi}");
    }
}
