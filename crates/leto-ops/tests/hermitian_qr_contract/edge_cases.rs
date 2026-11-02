use super::{errors, toeplitz, C};
use leto::{Array2, LetoError};
use leto_ops::{hermitian_eigen_qr, HermitianEigenConfig};

#[test]
fn exhausted_diagnostics_use_global_norm_and_original_units() {
    for scale in [2.0_f64.powi(-600), 1.0, 2.0_f64.powi(600)] {
        let mut a = toeplitz(33);
        for i in 0..33 {
            for j in 0..33 {
                a[[i, j]] *= if i == j { scale } else { 0.2 * scale };
            }
        }
        let config = HermitianEigenConfig {
            tolerance: scale,
            max_iterations: 0,
            ..Default::default()
        };
        match hermitian_eigen_qr(&a, config) {
            Err(LetoError::ConvergenceError { residual, tol, .. }) => {
                assert_eq!(tol, scale, "diagnostic tolerance must retain input units");
                assert!((residual / scale - 1.6).abs() < 1e-13);
                assert!(residual > tol);
            }
            other => panic!("global off-diagonal norm 1.6 exceeds tolerance 1: {other:?}"),
        }
    }
}

#[test]
fn absolute_policy_handles_an_overflowing_shift_ratio() {
    let n = 33;
    let edge = 1e-310;
    let mut a = Array2::from_elem([n, n], C::new(0.0, 0.0));
    for i in 0..n {
        a[[i, i]] = C::new(if i < 2 { (i + 1) as f64 } else { 4.0 }, 0.0);
    }
    a[[0, 1]] = C::new(0.0, edge);
    a[[1, 0]] = C::new(0.0, -edge);
    for tolerance in [0.0, f64::from_bits(1), 1e-320] {
        let result = hermitian_eigen_qr(
            &a,
            HermitianEigenConfig {
                tolerance,
                ..Default::default()
            },
        )
        .expect("overflowing ratio has a finite limiting shift");
        assert_eq!(result.eigenvalues[n - 1], 1.0);
        assert_eq!(result.eigenvalues[n - 2], 2.0);
        // This nonzero component verifies the tiny coupling was rotated,
        // rather than being discarded by a substituted relative threshold.
        let component = result.eigenvectors[[1, n - 1]];
        assert!((component.re.hypot(component.im) / edge - 1.0).abs() < 1e-12);
        assert!(result.off_diagonal_norm <= tolerance);
        assert!(result.iterations > 0);
        let (residual, orthogonality) = errors(&a, &result);
        assert!(residual < 1e-13 && orthogonality < 1e-13);
    }
}

#[test]
fn dense_rank_two_gram_has_analytic_spectrum_and_eigenpairs() {
    let n = 120;
    let phases = [
        C::new(1.0, 0.0),
        C::new(0.0, 1.0),
        C::new(-1.0, 0.0),
        C::new(0.0, -1.0),
    ];
    // u_i=i^i, v_i=(-1)^i u_i, A=uu^H+4vv^H. u and v are
    // orthogonal with squared norm 120: spectrum 480, 120, 118 zeros.
    let mut a = Array2::from_elem([n, n], C::new(0.0, 0.0));
    for i in 0..n {
        for j in 0..n {
            let coefficient = if (i + j) % 2 == 0 { 5.0 } else { -3.0 };
            a[[i, j]] = phases[i % 4] * phases[j % 4].conj() * coefficient;
        }
    }
    let config = HermitianEigenConfig::default();
    let result = hermitian_eigen_qr(&a, config).expect("rank-two Gram");
    let value_bound = config.tolerance + 64.0 * n as f64 * f64::EPSILON * 480.0;
    assert!((result.eigenvalues[0] - 480.0).abs() < value_bound);
    assert!((result.eigenvalues[1] - 120.0).abs() < value_bound);
    assert!(result
        .eigenvalues
        .iter()
        .skip(2)
        .all(|v| v.abs() < value_bound));
    let (residual, orthogonality) = errors(&a, &result);
    assert!(residual < 64.0 * n as f64 * f64::EPSILON);
    assert!(orthogonality < 64.0 * n as f64 * f64::EPSILON);
    assert!(result.off_diagonal_norm <= config.tolerance);
}
