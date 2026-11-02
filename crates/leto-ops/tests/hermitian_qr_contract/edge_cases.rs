use super::{errors, toeplitz, C};
use eunomia::FloatElement;
use leto::{Array2, LetoError};
use leto_ops::{hermitian_eigen_qr, HermitianEigenConfig};

#[test]
fn zero_leading_columns_with_complex_tails_retain_eigenpairs() {
    let n = 33;
    let mut a = Array2::from_elem([n, n], C::new(0.0, 0.0));
    for i in 0..n {
        a[[i, i]] = C::new(4.0, 0.0);
    }
    for (i, j, z) in [(2, 0, C::new(0.0, 3.0)), (4, 1, C::new(2.0, 2.0))] {
        a[[i, j]] = z;
        a[[j, i]] = z.conj();
    }
    let result =
        hermitian_eigen_qr(&a, HermitianEigenConfig::default()).expect("zero-leading reflectors");
    let expected = [7.0, 4.0 + 8.0_f64.sqrt(), 4.0 - 8.0_f64.sqrt(), 1.0];
    for (index, value) in [0, 1, n - 2, n - 1].into_iter().zip(expected) {
        assert!((result.eigenvalues[index] - value).abs() < 1e-12);
    }
    assert!(result
        .eigenvalues
        .iter()
        .skip(2)
        .take(n - 4)
        .all(|v| (v - 4.0).abs() < 1e-12));
    let (residual, orthogonality) = errors(&a, &result);
    assert!(residual < 1e-12 && orthogonality < 1e-12);
}

#[test]
fn subnormal_global_norm_does_not_round_each_edge_away() {
    let n = 33;
    let tiny = f64::from_bits(1);
    let mut a = Array2::from_elem([n, n], C::new(0.0, 0.0));
    for i in 0..n {
        a[[i, i]] = C::new(4.0, 0.0);
        if i + 1 < n {
            a[[i, i + 1]] = C::new(tiny, 0.0);
            a[[i + 1, i]] = C::new(tiny, 0.0);
        }
    }
    match hermitian_eigen_qr(
        &a,
        HermitianEigenConfig {
            tolerance: tiny,
            max_iterations: 0,
            ..Default::default()
        },
    ) {
        Err(LetoError::ConvergenceError { residual, tol, .. }) => {
            assert_eq!(residual, 8.0 * tiny);
            assert_eq!(tol, tiny);
        }
        Ok(result) => panic!(
            "64 subnormal components have norm 8*min_subnormal, returned Ok with norm {}",
            result.off_diagonal_norm
        ),
        Err(error) => panic!("expected convergence diagnostics, got {error}"),
    }
}

#[test]
fn subnormal_columns_and_final_complex_edges_preserve_unitarity() {
    for final_edge in [false, true] {
        let n = 33;
        let tiny = f64::from_bits(1);
        let mut a = Array2::from_elem([n, n], C::new(0.0, 0.0));
        for i in 0..n {
            a[[i, i]] = C::new(4.0, 0.0);
        }
        let edges = if final_edge {
            vec![(n - 1, n - 2, C::new(tiny, tiny))]
        } else {
            vec![(1, 0, C::new(tiny, 0.0)), (2, 0, C::new(tiny, 0.0))]
        };
        for (i, j, z) in edges {
            a[[i, j]] = z;
            a[[j, i]] = z.conj();
        }
        let result = hermitian_eigen_qr(
            &a,
            HermitianEigenConfig {
                tolerance: 0.0,
                ..Default::default()
            },
        )
        .expect("subnormal coupling must not create a non-unitary transformation");
        let (residual, orthogonality) = errors(&a, &result);
        assert!(
            result.eigenvalues.iter().all(|v| (v - 4.0).abs() < 1e-12),
            "values={:?}",
            result.eigenvalues
        );
        assert!(residual < 1e-12, "relative residual={residual}");
        assert!(orthogonality < 1e-12, "orthogonality={orthogonality}");
        assert_eq!(result.off_diagonal_norm, 0.0);
    }
}

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
        a[[i, i]] = C::new(if i < 2 { f64::from_count(i + 1) } else { 4.0 }, 0.0);
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
    let value_bound = config.tolerance + 64.0 * f64::from_count(n) * f64::EPSILON * 480.0;
    assert!((result.eigenvalues[0] - 480.0).abs() < value_bound);
    assert!((result.eigenvalues[1] - 120.0).abs() < value_bound);
    assert!(result
        .eigenvalues
        .iter()
        .skip(2)
        .all(|v| v.abs() < value_bound));
    let (residual, orthogonality) = errors(&a, &result);
    assert!(residual < 64.0 * f64::from_count(n) * f64::EPSILON);
    assert!(orthogonality < 64.0 * f64::from_count(n) * f64::EPSILON);
    assert!(result.off_diagonal_norm <= config.tolerance);
}
