//! Hermitian QR contracts above the Jacobi delegation boundary.
use eunomia::Complex;
use leto::{Array2, LetoError};
use leto_ops::{
    hermitian_eigen_jacobi, hermitian_eigen_qr, HermitianEigenConfig, HermitianEigenResult,
};

type C = Complex<f64>;

#[path = "hermitian_qr_contract/edge_cases.rs"]
mod edge_cases;

fn toeplitz(n: usize) -> Array2<C> {
    // Unitarily similar, by diagonal phases, to tridiag(1, 4, 1).
    let mut a = Array2::from_elem([n, n], C::new(0.0, 0.0));
    for i in 0..n {
        a[[i, i]] = C::new(4.0, 0.0);
        if i + 1 < n {
            a[[i, i + 1]] = C::new(0.0, 1.0);
            a[[i + 1, i]] = C::new(0.0, -1.0);
        }
    }
    a
}

fn errors(a: &Array2<C>, result: &HermitianEigenResult) -> (f64, f64) {
    let n = a.shape()[0];
    let scale = a
        .iter()
        .map(|z| z.re.abs().max(z.im.abs()))
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    let mut residual = 0.0;
    let mut orthogonality = 0.0;
    let mut norm = 0.0;
    for i in 0..n {
        for j in 0..n {
            norm += (a[[i, j]] / scale).norm_sqr();
            let mut av = C::new(0.0, 0.0);
            let mut gram = C::new(0.0, 0.0);
            for k in 0..n {
                av += (a[[i, k]] / scale) * result.eigenvectors[[k, j]];
                gram += result.eigenvectors[[k, i]].conj() * result.eigenvectors[[k, j]];
            }
            residual +=
                (av - result.eigenvectors[[i, j]] * (result.eigenvalues[j] / scale)).norm_sqr();
            orthogonality += (gram - C::new(f64::from(i == j), 0.0)).norm_sqr();
        }
    }
    (residual.sqrt() / norm.sqrt(), orthogonality.sqrt())
}

#[test]
fn exhausted_budget_returns_error_above_jacobi_boundary() {
    let config = HermitianEigenConfig {
        max_iterations: 0,
        ..Default::default()
    };
    match hermitian_eigen_qr(&toeplitz(33), config) {
        Err(LetoError::ConvergenceError {
            max_iters,
            residual,
            tol,
        }) => {
            assert_eq!(max_iters, 0);
            assert!(
                residual.is_finite() && residual > tol,
                "{residual} <= {tol}"
            );
        }
        other => panic!("non-diagonal matrix with no iteration budget must fail: {other:?}"),
    }
}

#[test]
fn complex_toeplitz_above_jacobi_boundary_has_analytic_eigenpairs() {
    let n = 33;
    let a = toeplitz(n);
    let config = HermitianEigenConfig {
        max_iterations: 128,
        ..Default::default()
    };
    let result =
        hermitian_eigen_qr(&a, config).expect("128 shifts suffice for this Toeplitz matrix");
    let (residual, orthogonality) = errors(&a, &result);
    let max_value_error = (0..n)
        .map(|i| {
            let angle = std::f64::consts::PI * (i + 1) as f64 / (n + 1) as f64;
            (result.eigenvalues[i] - (4.0 + 2.0 * angle.cos())).abs()
        })
        .fold(0.0_f64, f64::max);
    eprintln!("n={n} shifts={} offdiag={} value_error={max_value_error:e} residual={residual:e} orthogonality={orthogonality:e}", result.iterations, result.off_diagonal_norm);
    // Absolute stopping tolerance plus a dimension-scaled roundoff allowance.
    let bound = config.tolerance + 64.0 * n as f64 * f64::EPSILON;
    assert!(
        max_value_error <= bound,
        "eigenvalue error {max_value_error:e} > {bound:e}"
    );
    assert!(
        residual <= bound,
        "relative residual {residual:e} > {bound:e}"
    );
    assert!(
        orthogonality <= bound,
        "orthogonality error {orthogonality:e} > {bound:e}"
    );
    assert!(result.off_diagonal_norm <= config.tolerance);
}

#[test]
fn diagonal_matrix_needs_no_qr_steps() {
    let n = 33;
    let mut a = Array2::from_elem([n, n], C::new(0.0, 0.0));
    for i in 0..n {
        a[[i, i]] = C::new(i as f64, 0.0);
    }
    let result = hermitian_eigen_qr(
        &a,
        HermitianEigenConfig {
            max_iterations: 0,
            ..Default::default()
        },
    )
    .expect("already diagonal");
    assert_eq!(result.iterations, 0);
    assert_eq!(result.off_diagonal_norm, 0.0);
    for i in 0..n {
        assert_eq!(result.eigenvalues[i], (n - i - 1) as f64);
    }
    assert_eq!(errors(&a, &result), (0.0, 0.0));
}

#[test]
fn small_qr_preserves_jacobi_delegation() {
    let a = toeplitz(8);
    let config = HermitianEigenConfig::default();
    let qr = hermitian_eigen_qr(&a, config).expect("small QR");
    let jacobi = hermitian_eigen_jacobi(&a, config).expect("Jacobi");
    assert_eq!(qr.eigenvalues.as_slice(), jacobi.eigenvalues.as_slice());
    assert_eq!(qr.eigenvectors.as_slice(), jacobi.eigenvectors.as_slice());
    assert_eq!(qr.iterations, jacobi.iterations);
    let (residual, orthogonality) = errors(&a, &qr);
    assert!(residual < 1e-10 && orthogonality < 1e-10);
}

#[test]
fn dense_repeated_spectrum_survives_complex_phases_and_extreme_scales() {
    let n = 33;
    let phases = [
        C::new(1.0, 0.0),
        C::new(0.0, 1.0),
        C::new(-1.0, 0.0),
        C::new(0.0, -1.0),
    ];
    for scale in [2.0_f64.powi(-600), 1.0, 2.0_f64.powi(600)] {
        // -4I + 8 u uᴴ / n: eigenvalues 4 once, -4 with multiplicity n-1.
        let mut a = Array2::from_elem([n, n], C::new(0.0, 0.0));
        for i in 0..n {
            for j in 0..n {
                a[[i, j]] = phases[i % 4] * phases[j % 4].conj() * (8.0 / n as f64);
                if i == j {
                    a[[i, j]] -= C::new(4.0, 0.0);
                }
                a[[i, j]] *= scale;
            }
        }
        let config = HermitianEigenConfig {
            tolerance: 1e-10 * scale,
            ..Default::default()
        };
        let result = hermitian_eigen_qr(&a, config).expect("scaled dense Hermitian spectrum");
        for i in 0..n {
            let expected = if i == 0 { 4.0 } else { -4.0 };
            assert!((result.eigenvalues[i] / scale - expected).abs() < 1e-10);
        }
        let (residual, orthogonality) = errors(&a, &result);
        assert!(
            residual < 1e-10 && orthogonality < 1e-10,
            "scale={scale:e}, residual={residual:e}, orthogonality={orthogonality:e}"
        );
        assert!(result.off_diagonal_norm <= config.tolerance);
    }
}

#[test]
fn last_allowed_shift_is_accepted_and_one_fewer_reports_nonconvergence() {
    let a = toeplitz(33);
    let config = HermitianEigenConfig::default();
    let first = hermitian_eigen_qr(&a, config).expect("converges");
    let exact = hermitian_eigen_qr(
        &a,
        HermitianEigenConfig {
            max_iterations: first.iterations,
            ..config
        },
    )
    .expect("last allowed shift");
    assert_eq!(exact.iterations, first.iterations);
    assert_eq!(exact.eigenvalues.as_slice(), first.eigenvalues.as_slice());
    let short = hermitian_eigen_qr(
        &a,
        HermitianEigenConfig {
            max_iterations: first.iterations - 1,
            ..config
        },
    );
    match short {
        Err(LetoError::ConvergenceError {
            max_iters,
            residual,
            tol,
        }) => {
            assert_eq!(max_iters, first.iterations - 1);
            assert!(residual.is_finite() && residual > tol);
        }
        _ => panic!("one fewer shift must not claim convergence"),
    }
}

#[test]
fn invalid_qr_configuration_and_nonhermitian_data_are_rejected() {
    for tolerance in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(matches!(
            hermitian_eigen_qr(
                &toeplitz(33),
                HermitianEigenConfig {
                    tolerance,
                    ..Default::default()
                }
            ),
            Err(LetoError::InvalidInput(_))
        ));
    }
    for value in [
        C::new(f64::NAN, 0.0),
        C::new(0.0, f64::INFINITY),
        C::new(4.0, 1.0),
    ] {
        let mut a = toeplitz(33);
        a[[0, 0]] = value;
        assert!(matches!(
            hermitian_eigen_qr(&a, HermitianEigenConfig::default()),
            Err(LetoError::InvalidInput(_))
        ));
    }
}

#[test]
fn indefinite_diagonal_condition_uses_absolute_extrema_without_sorting() {
    let n = 33;
    let mut a = Array2::from_elem([n, n], C::new(0.0, 0.0));
    for i in 0..n {
        a[[i, i]] = C::new(2.0, 0.0);
    }
    a[[0, 0]] = C::new(-100.0, 0.0);
    a[[n - 1, n - 1]] = C::new(4.0, 0.0);
    let result = hermitian_eigen_qr(
        &a,
        HermitianEigenConfig {
            sort_descending: false,
            ..Default::default()
        },
    )
    .expect("diagonal");
    assert_eq!(result.condition_number, Some(50.0));
    assert_eq!(errors(&a, &result), (0.0, 0.0));
}

#[test]
fn zero_diagonal_indefinite_toeplitz_retains_its_zero_eigenvalue() {
    let n = 33;
    let mut a = toeplitz(n);
    for i in 0..n {
        a[[i, i]] = C::new(0.0, 0.0);
    }
    let result =
        hermitian_eigen_qr(&a, HermitianEigenConfig::default()).expect("indefinite Toeplitz");
    for i in 0..n {
        let expected = 2.0 * (std::f64::consts::PI * (i + 1) as f64 / (n + 1) as f64).cos();
        assert!((result.eigenvalues[i] - expected).abs() < 1e-10);
    }
    let (residual, orthogonality) = errors(&a, &result);
    assert!(residual < 1e-10 && orthogonality < 1e-10);
    assert_eq!(result.condition_number, None);
}

#[test]
fn absolute_tolerance_resolves_a_tiny_block_beside_unit_scale() {
    let n = 33;
    let tiny = 1e-280;
    let mut a = Array2::from_elem([n, n], C::new(0.0, 0.0));
    for i in 0..n {
        a[[i, i]] = C::new(if i < n - 3 { 1.0 } else { 4.0 * tiny }, 0.0);
    }
    for i in n - 3..n - 1 {
        a[[i, i + 1]] = C::new(0.0, tiny);
        a[[i + 1, i]] = C::new(0.0, -tiny);
    }
    let config = HermitianEigenConfig {
        tolerance: 1e-290,
        ..Default::default()
    };
    let result = hermitian_eigen_qr(&a, config).expect("tiny block needs local safe arithmetic");
    for i in 0..n - 3 {
        assert_eq!(result.eigenvalues[i], 1.0);
    }
    for i in 0..3 {
        let expected = 4.0 + 2.0 * (std::f64::consts::PI * (i + 1) as f64 / 4.0).cos();
        let actual = result.eigenvalues[n - 3 + i] / tiny;
        assert!(
            (actual - expected).abs() < 1e-10,
            "tiny eigenvalue {i}: {actual} != {expected}"
        );
    }
    assert!(result.off_diagonal_norm <= config.tolerance);
}

#[test]
fn kwavers_120_gram_fixture_converges_with_correct_eigenpairs() {
    // Exact construction from Kwavers #1012, merge e3e0f2aae79d89e56f370ec986ccb9667ae1114c,
    // crates/kwavers/benches/subspace_eigen.rs::hermitian_covariance.
    let n = 120;
    let mut factor = Array2::from_elem([n, n], C::new(0.0, 0.0));
    let mut row_phase = 0.0_f64;
    for i in 0..n {
        let mut col_phase = 0.0_f64;
        for j in 0..n {
            factor[[i, j]] = C::new(
                (row_phase * 0.37 + col_phase * 0.11).sin(),
                (row_phase * 0.13 - col_phase * 0.29).cos(),
            );
            col_phase += 1.0;
        }
        row_phase += 1.0;
    }
    let mut a = Array2::from_elem([n, n], C::new(0.0, 0.0));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                a[[i, j]] += factor[[i, k]] * factor[[j, k]].conj();
            }
        }
        a[[i, i]] += C::new(0.5, 0.0);
    }
    let config = HermitianEigenConfig::default();
    let result = hermitian_eigen_qr(&a, config).expect("120x120 Gram QR converges");
    let jacobi = hermitian_eigen_jacobi(&a, config).expect("independent Jacobi comparison");
    let (residual, orthogonality) = errors(&a, &result);
    let largest = result.eigenvalues[0];
    let value_difference = (0..n)
        .map(|i| (result.eigenvalues[i] - jacobi.eigenvalues[i]).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("Gram n={n} shifts={} offdiag={} max_jacobi_difference={value_difference:e} residual={residual:e} orthogonality={orthogonality:e}", result.iterations, result.off_diagonal_norm);
    let roundoff = 64.0 * n as f64 * f64::EPSILON;
    assert!(value_difference <= config.tolerance + roundoff * largest);
    assert!(result.eigenvalues[n - 1] >= 0.5 - roundoff * largest);
    assert!(residual <= config.tolerance / largest + roundoff);
    assert!(orthogonality <= roundoff);
    assert!(result.off_diagonal_norm <= config.tolerance);
}
