//! GMRES(m) conformance tests.
//!
//! Reference values are the published solutions of the two systems used by the
//! `faer-gmres` and `gmres` (RLado) crates, so the recurrence here is checked
//! against independently produced results rather than against itself.

use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, Array2, LetoError};
use leto_ops::{
    Configurable, ConvergenceMonitor, CsrMatrix, IterativeLinearSolver, IterativeSolverConfig,
    JacobiPreconditioner, LinearOperator, Preconditioner, GMRES,
};

/// Dense 5x5 non-symmetric system shared by both reference crates.
const REFERENCE_MATRIX: [[f64; 5]; 5] = [
    [0.888_641, 0.477_151, 0.764_081, 0.244_348, 0.662_542],
    [0.695_741, 0.991_383, 0.800_932, 0.089_616, 0.250_400],
    [0.149_974, 0.584_978, 0.937_576, 0.870_798, 0.990_016],
    [0.429_292, 0.459_984, 0.056_629, 0.567_589, 0.048_561],
    [0.454_428, 0.253_192, 0.173_598, 0.321_640, 0.632_031],
];
const REFERENCE_RHS: [f64; 5] = [0.104_594, 0.437_549, 0.040_264, 0.298_842, 0.254_451];
/// Published solution of `REFERENCE_MATRIX · x = REFERENCE_RHS`.
const REFERENCE_SOLUTION: [f64; 5] = [0.037_919, 0.888_551, -0.657_575, -0.181_680, 0.292_447];

fn scalar<T: FloatElement>(value: f64) -> T {
    <T as FloatElement>::from_f64(value)
}

fn vector<T: FloatElement + RealField + Copy>(values: &[f64]) -> Array1<T> {
    let mut v = Array1::zeros([values.len()]);
    for (index, &value) in values.iter().enumerate() {
        v[index] = scalar(value);
    }
    v
}

fn reference_operator<T>() -> CsrMatrix<T>
where
    T: FloatElement + RealField + Copy + leto_ops::Scalar,
{
    let mut dense = Array2::zeros([5, 5]);
    for (row, values) in REFERENCE_MATRIX.iter().enumerate() {
        for (column, &value) in values.iter().enumerate() {
            dense[[row, column]] = scalar::<T>(value);
        }
    }
    CsrMatrix::from_dense(&dense.view())
}

fn diagonal_operator<T>(diagonal: &[f64]) -> CsrMatrix<T>
where
    T: FloatElement + RealField + Copy + leto_ops::Scalar,
{
    let n = diagonal.len();
    let mut dense = Array2::zeros([n, n]);
    for (index, &value) in diagonal.iter().enumerate() {
        dense[[index, index]] = scalar::<T>(value);
    }
    CsrMatrix::from_dense(&dense.view())
}

fn assert_close<T: NumericElement>(actual: &Array1<T>, expected: &[f64], tolerance: f64) {
    for (index, &want) in expected.iter().enumerate() {
        let got = NumericElement::to_f64(actual[index]);
        assert!(
            (got - want).abs() <= tolerance,
            "component {index}: got {got}, want {want} (tolerance {tolerance})"
        );
    }
}

fn residual_norm<T, Op>(a: &Op, b: &Array1<T>, x: &Array1<T>) -> f64
where
    T: RealField + Copy,
    Op: LinearOperator<T> + ?Sized,
{
    let mut ax = Array1::zeros(b.shape());
    a.apply(x, &mut ax).expect("operator application");
    let mut sum = 0.0;
    for i in 0..b.shape()[0] {
        let residual = NumericElement::to_f64(b[i] - ax[i]);
        sum += residual * residual;
    }
    sum.sqrt()
}

/// Preconditioner `M⁻¹ = scale · I`.
///
/// A uniform scaling leaves the Krylov subspace, and therefore the true
/// solution, untouched but multiplies every preconditioned residual estimate
/// by `scale`. It is the minimal construction that separates "the recurrence
/// believes it has converged" from "the system is solved".
struct ScaledIdentity<T> {
    scale: T,
}

impl<T: RealField + Copy> Preconditioner<T> for ScaledIdentity<T> {
    fn apply_to(&self, r: &Array1<T>, z: &mut Array1<T>) -> leto::Result<()> {
        for i in 0..r.shape()[0] {
            z[i] = r[i] * self.scale;
        }
        Ok(())
    }
}

/// Solve the reference system and check it two ways.
///
/// `REFERENCE_SOLUTION` is quoted to six decimals by both reference crates and
/// is itself accurate to about five significant figures, so it can only carry a
/// `1e-4` comparison — the tolerance those crates use.  The rigorous oracle is
/// the residual, asserted at `residual_tolerance`.
fn solve_reference<T>(restart: usize, tolerance: f64, residual_tolerance: f64)
where
    T: FloatElement + RealField + Copy + core::fmt::Debug + leto_ops::Scalar,
{
    let a = reference_operator::<T>();
    let b = vector::<T>(&REFERENCE_RHS);
    let mut x = Array1::zeros([5]);
    let solver = GMRES::new(
        IterativeSolverConfig::new(scalar::<T>(tolerance)).with_max_iterations(200),
        restart,
    );
    solver
        .solve_unpreconditioned(&a, &b, &mut x)
        .expect("reference system converges");
    assert_close(&x, &REFERENCE_SOLUTION, 1e-4);
    let achieved = residual_norm(&a, &b, &x);
    assert!(
        achieved <= residual_tolerance,
        "residual {achieved} exceeds {residual_tolerance}"
    );
}

#[test]
fn reference_system_matches_published_solution_f64() {
    solve_reference::<f64>(30, 1e-10, 1e-12);
}

#[test]
fn reference_system_matches_published_solution_f32() {
    // f32 carries about 7 decimal digits; the threshold is set from the
    // published 6-digit reference solution, not from machine epsilon.
    solve_reference::<f32>(30, 1e-6, 1e-5);
}

#[test]
fn restarts_reach_the_same_solution() {
    // Restart widths below the system dimension force several outer cycles,
    // exercising the restart path that a single full cycle would skip.
    solve_reference::<f64>(3, 1e-10, 1e-10);
    solve_reference::<f64>(2, 1e-10, 1e-10);

    // GMRES(1) has no convergence guarantee on a non-symmetric operator and
    // stagnates on this system (Saad 2003, §6.5.5). The contract is that it
    // reports the stagnation instead of returning a false success.
    let a = reference_operator::<f64>();
    let b = vector::<f64>(&REFERENCE_RHS);
    let mut x = Array1::zeros([5]);
    let stalled = GMRES::new(
        IterativeSolverConfig::new(1e-10).with_max_iterations(200),
        1,
    )
    .solve_unpreconditioned(&a, &b, &mut x);
    assert!(stalled.is_err(), "GMRES(1) is expected to stagnate here");
    assert!(residual_norm(&a, &b, &x) > 1e-10);
}

#[test]
fn diagonal_system_matches_closed_form() {
    // diag(1, 2, 3) · x = [2, 2, 2]  =>  x = [2, 1, 2/3].
    let a = diagonal_operator::<f64>(&[1.0, 2.0, 3.0]);
    let b = vector::<f64>(&[2.0, 2.0, 2.0]);
    let mut x = Array1::zeros([3]);
    let solver = GMRES::new(IterativeSolverConfig::new(1e-12), 10);
    solver
        .solve_unpreconditioned(&a, &b, &mut x)
        .expect("diagonal system converges");
    assert_close(&x, &[2.0, 1.0, 2.0 / 3.0], 1e-10);
}

#[test]
fn distinct_eigenvalue_count_bounds_the_iteration_count() {
    // GMRES terminates in at most d iterations for a diagonalisable operator
    // with d distinct eigenvalues (minimal-polynomial argument). Two distinct
    // eigenvalues over six unknowns must close the Krylov subspace by step 2.
    let a = diagonal_operator::<f64>(&[2.0, 2.0, 2.0, 5.0, 5.0, 5.0]);
    let b = vector::<f64>(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
    let mut x = Array1::zeros([6]);
    let solver = GMRES::new(IterativeSolverConfig::new(1e-12), 20);
    let monitor: ConvergenceMonitor<f64> = solver
        .solve_unpreconditioned(&a, &b, &mut x)
        .expect("two-eigenvalue system converges");
    assert!(
        monitor.iteration <= 2,
        "expected termination within 2 iterations, took {}",
        monitor.iteration
    );
    for i in 0..6 {
        let expected = NumericElement::to_f64(b[i]) / if i < 3 { 2.0 } else { 5.0 };
        assert!((NumericElement::to_f64(x[i]) - expected).abs() <= 1e-10);
    }
}

#[test]
fn preconditioner_reduces_the_iteration_count() {
    // Jacobi scaling on a strongly row-scaled system; the reference crates
    // assert the same inequality for their Jacobi operator.
    let mut dense = Array2::zeros([4, 4]);
    let scales = [1.0, 1e3, 1e-2, 1e2];
    for row in 0..4 {
        for column in 0..4 {
            let base = if row == column { 4.0 } else { 0.6 };
            dense[[row, column]] = base * scales[row];
        }
    }
    let a = CsrMatrix::from_dense(&dense.view());
    let b = vector::<f64>(&[1.0, 1.0, 1.0, 1.0]);
    let config = IterativeSolverConfig::new(1e-10).with_max_iterations(200);

    let mut x_plain = Array1::zeros([4]);
    let plain = GMRES::new(config, 4)
        .solve_unpreconditioned(&a, &b, &mut x_plain)
        .expect("unpreconditioned solve converges");

    let mut x_jacobi = Array1::zeros([4]);
    let jacobi = JacobiPreconditioner::from_matrix(&a);
    let preconditioned = GMRES::new(config, 4)
        .solve_preconditioned(&a, &b, &jacobi, &mut x_jacobi)
        .expect("preconditioned solve converges");

    assert!(
        preconditioned.iteration < plain.iteration,
        "Jacobi took {} iterations, unpreconditioned took {}",
        preconditioned.iteration,
        plain.iteration
    );
    for i in 0..4 {
        let plain_value = NumericElement::to_f64(x_plain[i]);
        let jacobi_value = NumericElement::to_f64(x_jacobi[i]);
        assert!((plain_value - jacobi_value).abs() <= 1e-6);
    }
}

#[test]
fn preconditioned_estimate_alone_never_reports_convergence() {
    // Regression. The Arnoldi recurrence measures the norm of M⁻¹(b − Ax).
    // Under M⁻¹ = scale·I that estimate is `scale` times the true residual, so
    // for a small scale it meets the threshold while the system is still
    // unsolved. Reporting Ok on that estimate returned a badly wrong x; the
    // invariant asserted here is that Ok always implies a solved system.
    let a = reference_operator::<f64>();
    let b = vector::<f64>(&REFERENCE_RHS);
    let tolerance = 1e-10;

    for scale in [1.0_f64, 1e-6, 1e-12] {
        let mut x = Array1::zeros([5]);
        let solver = GMRES::new(
            IterativeSolverConfig::new(tolerance).with_max_iterations(200),
            30,
        );
        let outcome = solver.solve_preconditioned(&a, &b, &ScaledIdentity { scale }, &mut x);
        let achieved = residual_norm(&a, &b, &x);
        match outcome {
            Ok(_) => assert!(
                achieved <= tolerance,
                "scale {scale} reported convergence at true residual {achieved}"
            ),
            Err(_) => assert!(
                achieved > tolerance,
                "scale {scale} reported failure at converged residual {achieved}"
            ),
        }
    }

    // The unscaled case must additionally reach the reference solution.
    let mut x = Array1::zeros([5]);
    GMRES::new(
        IterativeSolverConfig::new(tolerance).with_max_iterations(200),
        30,
    )
    .solve_preconditioned(&a, &b, &ScaledIdentity { scale: 1.0_f64 }, &mut x)
    .expect("identity scaling converges");
    assert_close(&x, &REFERENCE_SOLUTION, 1e-4);
}

#[test]
fn relative_tolerance_makes_convergence_scale_invariant() {
    // Scaling A and b by alpha scales every residual by alpha, so a fixed
    // absolute tolerance is unreachable for large alpha. The relative test is
    // invariant under that rescaling.
    let alpha = 1e8;
    let mut dense = Array2::zeros([5, 5]);
    for (row, values) in REFERENCE_MATRIX.iter().enumerate() {
        for (column, &value) in values.iter().enumerate() {
            dense[[row, column]] = value * alpha;
        }
    }
    let scaled = CsrMatrix::from_dense(&dense.view());
    let mut b: Array1<f64> = Array1::zeros([5]);
    for (i, &value) in REFERENCE_RHS.iter().enumerate() {
        b[i] = value * alpha;
    }

    let absolute = IterativeSolverConfig::new(1e-10).with_max_iterations(60);
    let mut x_absolute = Array1::zeros([5]);
    let absolute_result =
        GMRES::new(absolute, 30).solve_unpreconditioned(&scaled, &b, &mut x_absolute);
    // At this scale the smallest attainable residual is above 1e-10, so the
    // absolute test cannot be met however the solver terminates.
    assert!(
        absolute_result.is_err(),
        "absolute-only tolerance must be unreachable at alpha = {alpha}"
    );
    assert!(residual_norm(&scaled, &b, &x_absolute) > 1e-10);

    let relative = absolute.with_relative_tolerance(1e-12);
    let mut x_relative = Array1::zeros([5]);
    GMRES::new(relative, 30)
        .solve_unpreconditioned(&scaled, &b, &mut x_relative)
        .expect("relative tolerance converges on the scaled system");
    assert_close(&x_relative, &REFERENCE_SOLUTION, 1e-4);
    assert_eq!(relative.threshold(1.0), 1e-10);
    assert_eq!(relative.threshold(1e6), 1e-6);
}

#[test]
fn exhausted_iteration_budget_reports_convergence_error() {
    let a = reference_operator::<f64>();
    let b = vector::<f64>(&REFERENCE_RHS);
    let mut x = Array1::zeros([5]);
    let solver = GMRES::new(IterativeSolverConfig::new(1e-14).with_max_iterations(1), 1);
    match solver.solve_unpreconditioned(&a, &b, &mut x) {
        Err(LetoError::ConvergenceError {
            max_iters,
            residual,
            tol,
        }) => {
            assert_eq!(max_iters, 1);
            assert!(
                residual > tol,
                "reported residual {residual} must exceed {tol}"
            );
        }
        other => panic!("expected ConvergenceError, got {other:?}"),
    }
}

#[test]
fn singular_preconditioner_is_reported_not_silently_ignored() {
    let a = reference_operator::<f64>();
    let b = vector::<f64>(&REFERENCE_RHS);
    let mut x = Array1::zeros([5]);
    let solver = GMRES::new(IterativeSolverConfig::new(1e-10), 30);
    let precond = ScaledIdentity { scale: 0.0_f64 };
    assert!(matches!(
        solver.solve_preconditioned(&a, &b, &precond, &mut x),
        Err(LetoError::NumericalBreakdown(_))
    ));
}

#[test]
fn non_finite_right_hand_side_is_rejected() {
    let a = reference_operator::<f64>();
    let mut b = vector::<f64>(&REFERENCE_RHS);
    b[2] = f64::NAN;
    let mut x = Array1::zeros([5]);
    let solver = GMRES::new(IterativeSolverConfig::new(1e-10), 30);
    assert!(matches!(
        solver.solve_unpreconditioned(&a, &b, &mut x),
        Err(LetoError::NumericalBreakdown(_))
    ));
}

#[test]
fn dimension_mismatch_is_rejected() {
    let a = reference_operator::<f64>();
    let b = vector::<f64>(&[1.0, 2.0, 3.0]);
    let mut x = Array1::zeros([3]);
    let solver = GMRES::new(IterativeSolverConfig::new(1e-10), 30);
    assert!(matches!(
        solver.solve_unpreconditioned(&a, &b, &mut x),
        Err(LetoError::InvalidInput(_))
    ));

    let b = vector::<f64>(&REFERENCE_RHS);
    let mut short = Array1::zeros([4]);
    assert!(matches!(
        solver.solve_unpreconditioned(&a, &b, &mut short),
        Err(LetoError::InvalidInput(_))
    ));
}

#[test]
fn zero_right_hand_side_returns_the_zero_solution() {
    let a = reference_operator::<f64>();
    let b: Array1<f64> = Array1::zeros([5]);
    let mut x = Array1::zeros([5]);
    let solver = GMRES::new(IterativeSolverConfig::new(1e-10), 30);
    let monitor = solver
        .solve_unpreconditioned(&a, &b, &mut x)
        .expect("zero system is already solved");
    assert_eq!(monitor.iteration, 0);
    assert_close(&x, &[0.0; 5], 0.0);
}

#[test]
fn a_converged_initial_guess_is_returned_unchanged() {
    let a = reference_operator::<f64>();
    let b = vector::<f64>(&REFERENCE_RHS);
    let mut x = vector::<f64>(&REFERENCE_SOLUTION);
    let solver = GMRES::new(IterativeSolverConfig::new(1e-3), 30);
    let monitor = solver
        .solve_unpreconditioned(&a, &b, &mut x)
        .expect("exact guess is already converged");
    assert_eq!(monitor.iteration, 0);
    assert_close(&x, &REFERENCE_SOLUTION, 0.0);
}

#[test]
fn workspace_is_reused_across_solves_of_differing_sizes() {
    // The solver caches scratch space keyed on (n, restart); a size change must
    // reallocate rather than reuse a stale-shaped buffer.
    let solver = GMRES::new(IterativeSolverConfig::new(1e-12), 4);

    let small = diagonal_operator::<f64>(&[1.0, 2.0, 3.0]);
    let mut x_small = Array1::zeros([3]);
    solver
        .solve_unpreconditioned(&small, &vector::<f64>(&[2.0, 2.0, 2.0]), &mut x_small)
        .expect("small system converges");
    assert_close(&x_small, &[2.0, 1.0, 2.0 / 3.0], 1e-10);

    let large = reference_operator::<f64>();
    let mut x_large = Array1::zeros([5]);
    solver
        .solve_unpreconditioned(&large, &vector::<f64>(&REFERENCE_RHS), &mut x_large)
        .expect("large system converges");
    assert_close(&x_large, &REFERENCE_SOLUTION, 1e-4);

    // Back to the first shape, from the cached larger workspace.
    let mut x_again = Array1::zeros([3]);
    solver
        .solve_unpreconditioned(&small, &vector::<f64>(&[2.0, 2.0, 2.0]), &mut x_again)
        .expect("small system converges again");
    assert_close(&x_again, &[2.0, 1.0, 2.0 / 3.0], 1e-10);
}

#[test]
fn csr_transpose_application_matches_the_dense_product() {
    let a = reference_operator::<f64>();
    let x = vector::<f64>(&[1.0, -2.0, 3.0, -4.0, 5.0]);
    let mut y = Array1::zeros([5]);
    a.apply_transpose(&x, &mut y).expect("transpose applies");
    for column in 0..5 {
        let expected: f64 = (0..5)
            .map(|row| REFERENCE_MATRIX[row][column] * NumericElement::to_f64(x[row]))
            .sum();
        assert!((NumericElement::to_f64(y[column]) - expected).abs() <= 1e-12);
    }
}

#[test]
fn solver_trait_route_produces_the_same_solution() {
    let a = reference_operator::<f64>();
    let b = vector::<f64>(&REFERENCE_RHS);
    let solver = GMRES::new(IterativeSolverConfig::new(1e-12), 30);
    assert_eq!(solver.config().max_iterations, 1000);
    let mut x = Array1::zeros([5]);
    solver
        .solve(&a, &b, &mut x, None::<&JacobiPreconditioner<f64>>)
        .expect("trait route converges");
    assert_close(&x, &REFERENCE_SOLUTION, 1e-4);
}
