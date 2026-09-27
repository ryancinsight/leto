#![allow(clippy::unwrap_used, reason = "test scope")]

use super::types::{
    OrderingStrategy, SparseLuSolver, DENSITY_THRESHOLD_DEFAULT, SMALL_SWITCH_DEFAULT,
};
use super::{csr_to_dense, sparse_lu_solve};
use crate::application::sparse::lu_symbolic::factor_symbolic;
use crate::application::sparse::{CooMatrix, CscMatrix, CsrMatrix};
use leto::{Array1, LetoError};

fn make_csr(nrows: usize, ncols: usize, triplets: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(nrows, ncols);
    for &(r, c, v) in triplets {
        coo.push(r, c, v);
    }
    coo.to_csr()
}

#[test]
fn solves_2x2_identity() {
    // I · x = b → x = b
    let a = make_csr(2, 2, &[(0, 0, 1.0), (1, 1, 1.0)]);
    let b = vec![3.0_f64, 7.0];
    let x = sparse_lu_solve(&a, &b).expect("identity system solves");
    assert!((x[0] - 3.0).abs() < 1e-12, "x[0] = {}", x[0]);
    assert!((x[1] - 7.0).abs() < 1e-12, "x[1] = {}", x[1]);
}

#[test]
fn solves_native_array_view() {
    let a = make_csr(2, 2, &[(0, 0, 3.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)]);
    let b = Array1::from_shape_vec([2], vec![9.0_f64, 8.0]).expect("RHS shape");
    let x = SparseLuSolver::default()
        .solve_view(&a, &b.view())
        .expect("native view solve");

    assert_eq!(x.as_slice(), Some(&[2.0, 3.0][..]));
}

#[test]
fn solves_small_diagonally_dominant_system() {
    // [ 3  1 ] [ x0 ]   [ 9 ]      x0 = 2, x1 = 3
    // [ 1  2 ] [ x1 ] = [ 8 ]
    let a = make_csr(2, 2, &[(0, 0, 3.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)]);
    let b = vec![9.0_f64, 8.0];
    let x = sparse_lu_solve(&a, &b).expect("2×2 solve");
    assert!((x[0] - 2.0).abs() < 1e-10, "x[0] = {}", x[0]);
    assert!((x[1] - 3.0).abs() < 1e-10, "x[1] = {}", x[1]);
}

#[test]
fn solves_3x3_system() {
    // [ 2  1  0 ] [ x ]   [ 5  ]   Exact: x=13/9, y=19/9, z=20/9
    // [ 1  3  1 ] [ y ] = [ 10 ]
    // [ 0  1  4 ] [ z ]   [ 11 ]
    let a = make_csr(
        3,
        3,
        &[
            (0, 0, 2.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, 4.0),
        ],
    );
    let b = vec![5.0_f64, 10.0, 11.0];
    let x = sparse_lu_solve(&a, &b).expect("3×3 solve");
    let expected_x0 = 13.0 / 9.0;
    let expected_x1 = 19.0 / 9.0;
    let expected_x2 = 20.0 / 9.0;
    assert!(
        (x[0] - expected_x0).abs() < 1e-10,
        "x[0] = {} expected {}",
        x[0],
        expected_x0
    );
    assert!(
        (x[1] - expected_x1).abs() < 1e-10,
        "x[1] = {} expected {}",
        x[1],
        expected_x1
    );
    assert!(
        (x[2] - expected_x2).abs() < 1e-10,
        "x[2] = {} expected {}",
        x[2],
        expected_x2
    );
}

#[test]
fn rejects_system_over_dense_limit() {
    let solver = SparseLuSolver {
        max_size: 4,
        pivot_tolerance: 1e-12,
        small_switch: SMALL_SWITCH_DEFAULT,
        density_threshold: DENSITY_THRESHOLD_DEFAULT,
        ordering: OrderingStrategy::default(),
    };
    let a = make_csr(
        5,
        5,
        &[
            (0, 0, 1.0),
            (1, 1, 1.0),
            (2, 2, 1.0),
            (3, 3, 1.0),
            (4, 4, 1.0),
        ],
    );
    let b = vec![1.0_f64; 5];
    let err = solver.solve(&a, &b).expect_err("over limit");
    match &err {
        LetoError::StorageError { reason } => {
            assert!(reason.contains("exceeds max_size"), "unexpected: {reason}");
            assert!(
                reason.contains("iterative"),
                "should suggest iterative: {reason}"
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn rejects_non_square_matrix() {
    let a = make_csr(2, 3, &[(0, 0, 1.0), (1, 1, 1.0)]);
    let b = vec![1.0_f64, 2.0];
    let err = sparse_lu_solve(&a, &b).expect_err("non-square");
    match &err {
        LetoError::StorageError { reason } => {
            assert!(reason.contains("square"), "unexpected: {reason}");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn rejects_rhs_length_mismatch() {
    let a = make_csr(3, 3, &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)]);
    let b = vec![1.0_f64, 2.0]; // wrong length
    let err = sparse_lu_solve(&a, &b).expect_err("length mismatch");
    match &err {
        LetoError::StorageError { reason } => {
            assert!(reason.contains("RHS length"), "unexpected: {reason}");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn csr_to_dense_round_trips() {
    let a = make_csr(2, 3, &[(0, 0, 1.0), (0, 2, 3.0), (1, 1, 2.0)]);
    let d = csr_to_dense(&a);
    assert_eq!(d.get([0, 0]).copied().unwrap(), 1.0);
    assert_eq!(d.get([0, 1]).copied().unwrap(), 0.0);
    assert_eq!(d.get([0, 2]).copied().unwrap(), 3.0);
    assert_eq!(d.get([1, 1]).copied().unwrap(), 2.0);
}

#[test]
fn solver_is_generic_over_f32() {
    let mut coo = CooMatrix::new(2, 2);
    coo.push(0, 0, 2.0_f32);
    coo.push(0, 1, 0.0_f32);
    coo.push(1, 0, 0.0_f32);
    coo.push(1, 1, 4.0_f32);
    let a = coo.to_csr();
    let b = vec![6.0_f32, 8.0_f32];
    let rhs = Array1::from_shape_vec([2], b).expect("RHS shape");
    let x = SparseLuSolver::default()
        .solve_view(&a, &rhs.view())
        .expect("f32 solve");
    assert!((x[0] - 3.0_f32).abs() < 1e-5, "x[0] = {}", x[0]);
    assert!((x[1] - 2.0_f32).abs() < 1e-5, "x[1] = {}", x[1]);
}

#[test]
fn dense_path_taken_for_near_dense_4x4() {
    // A 4x4 completely dense matrix: n <= small_switch (32) so the dense
    // path is selected by the small-matrix criterion. Both paths produce
    // a value-equivalent solution; this test verifies the dispatch helper
    // routes by small n and that the solve matches a closed-form x.
    let mut coo = CooMatrix::new(4, 4);
    let entries: [(usize, usize, f64); 16] = [
        (0, 0, 9.5),
        (0, 1, 1.0),
        (0, 2, 3.0),
        (0, 3, 2.0),
        (1, 0, 0.5),
        (1, 1, 4.0),
        (1, 2, 1.0),
        (1, 3, 7.0),
        (2, 0, 2.0),
        (2, 1, 5.0),
        (2, 2, 12.0),
        (2, 3, 1.0),
        (3, 0, -1.0),
        (3, 1, 0.0),
        (3, 2, 1.0),
        (3, 3, 8.0),
    ];
    for &(r, c, v) in &entries {
        coo.push(r, c, v);
    }
    let a = coo.to_csr();
    // Construct a dense rhs from a closed-form solution.
    let x_known: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let mut b = vec![0.0_f64; 4];
    for &(r, c, v) in &entries {
        b[r] += v * x_known[c];
    }
    let b_arr = Array1::from_shape_vec([4], b).expect("b shape");

    // Sanity: dispatch must select the dense path due to n==4 <=
    // small_switch (32). Verify the solve works.
    let solver = SparseLuSolver::default();
    debug_assert!(solver.use_dense_path(4, a.nnz()));
    let x = solver
        .solve_view(&a, &b_arr.view())
        .expect("dense dispatch solve");
    for i in 0..4 {
        assert!((x[i] - x_known[i]).abs() < 1e-10, "x[{i}] = {}", x[i]);
    }
}

/// The tridiagonal Poisson-Laplacian used by the owned-factor tests:
/// n=64, density ≈ 0.047 < 0.1 and n > small_switch, so the sparse
/// arm is dispatched. `diag` scales the main diagonal so two matrices
/// share one pattern with different values.
fn tridiagonal_csr(n: usize, diag: f64) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, diag);
        if i > 0 {
            coo.push(i, i - 1, -1.0_f64);
        }
        if i + 1 < n {
            coo.push(i, i + 1, -1.0_f64);
        }
    }
    coo.to_csr()
}

/// `‖A x - b‖∞` computed directly from the CSR nonzeros.
fn csr_residual_inf(a: &CsrMatrix<f64>, x: &Array1<f64>, b: &[f64]) -> f64 {
    let mut max_residual = 0.0_f64;
    for (row, &rhs) in b.iter().enumerate().take(a.nrows()) {
        let mut ax = 0.0_f64;
        for (&col, &value) in a.row(row).col_indices().iter().zip(a.row(row).values()) {
            ax += value * x[col];
        }
        let d = (ax - rhs).abs();
        if d > max_residual {
            max_residual = d;
        }
    }
    max_residual
}

#[test]
fn owned_factor_reuses_symbolic_across_value_changes() {
    // The CFDrs block-preconditioner pattern: one symbolic analysis,
    // several numeric factors over matrices sharing the pattern.
    let n = 64usize;
    let solver = SparseLuSolver::default();
    let a1 = tridiagonal_csr(n, 2.0);
    let symbolic = factor_symbolic(&CscMatrix::from_csr(&a1));
    let b: Vec<f64> = (1..=n).map(|k| k as f64).collect();
    let b_arr = Array1::from_shape_vec([n], b.clone()).expect("b shape");

    for diag in [2.0_f64, 4.0] {
        let a = tridiagonal_csr(n, diag);
        let factor = solver
            .factor_sparse_with_symbolic(&a, &symbolic)
            .expect("pivoting-free factorization");
        assert_eq!(factor.n(), n);
        let mut x = Array1::from_shape_vec([n], vec![0.0_f64; n]).expect("x shape");
        factor
            .solve_into(&b_arr.view(), &mut x.view_mut())
            .expect("solve_into");
        let residual = csr_residual_inf(&a, &x, &b);
        assert!(residual < 1e-8, "diag={diag}: residual = {residual}");
    }
}

#[test]
fn owned_factor_matches_solve_view() {
    // Value-semantic differential: the cached factor and the one-shot
    // dispatcher must produce the same solution on the sparse arm.
    let n = 64usize;
    let solver = SparseLuSolver::default();
    let a = tridiagonal_csr(n, 2.0);
    assert!(!solver.use_dense_path(n, a.nnz()), "sparse arm expected");
    let b: Vec<f64> = (0..n).map(|k| (k as f64) * 0.25 - 3.0).collect();
    let b_arr = Array1::from_shape_vec([n], b).expect("b shape");

    let symbolic = factor_symbolic(&CscMatrix::from_csr(&a));
    let factor = solver
        .factor_sparse_with_symbolic(&a, &symbolic)
        .expect("factor");
    let x_factor = factor.solve(&b_arr.view()).expect("factor solve");
    let x_direct = solver.solve_view(&a, &b_arr.view()).expect("direct solve");
    for i in 0..n {
        let d = (x_factor[i] - x_direct[i]).abs();
        assert!(d < 1e-12, "x[{i}] differs by {d}");
    }
}

#[test]
fn owned_factor_falls_back_to_dense_when_pivoting_required() {
    // Zero the leading diagonal entry so column 0's pivot must come
    // from row 1 — the sparse convention reports NumericalBreakdown and
    // the owned factor must transparently hold the dense factorization.
    let n = 64usize;
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        if i != 0 {
            coo.push(i, i, 4.0_f64);
        }
        if i > 0 {
            coo.push(i, i - 1, -1.0_f64);
        }
        if i + 1 < n {
            coo.push(i, i + 1, -1.0_f64);
        }
    }
    let a = coo.to_csr();
    let solver = SparseLuSolver::default();
    assert!(!solver.use_dense_path(n, a.nnz()), "sparse arm expected");

    let x_known: Vec<f64> = (0..n).map(|i| 1.0_f64 / (i as f64 + 1.0)).collect();
    let mut b = vec![0.0_f64; n];
    for (row, rhs) in b.iter_mut().enumerate() {
        for (&col, &value) in a.row(row).col_indices().iter().zip(a.row(row).values()) {
            *rhs += value * x_known[col];
        }
    }
    let symbolic = factor_symbolic(&CscMatrix::from_csr(&a));
    let factor = solver
        .factor_sparse_with_symbolic(&a, &symbolic)
        .expect("dense fallback factors the pivot-requiring matrix");
    let b_arr = Array1::from_shape_vec([n], b.clone()).expect("b shape");
    let x = factor.solve(&b_arr.view()).expect("solve");
    for i in 0..n {
        let d = (x[i] - x_known[i]).abs();
        assert!(d < 1e-8, "x[{i}] = {} expected {}", x[i], x_known[i]);
    }
}

#[test]
fn owned_factor_small_matrix_routes_dense() {
    // n=2 ≤ small_switch: dispatch takes the dense arm outright.
    let a = make_csr(2, 2, &[(0, 0, 3.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)]);
    let solver = SparseLuSolver::default();
    let symbolic = factor_symbolic(&CscMatrix::from_csr(&a));
    let factor = solver
        .factor_sparse_with_symbolic(&a, &symbolic)
        .expect("dense-arm factor");
    let b = Array1::from_shape_vec([2], vec![9.0_f64, 8.0]).expect("b shape");
    let x = factor.solve(&b.view()).expect("solve");
    assert!((x[0] - 2.0).abs() < 1e-10, "x[0] = {}", x[0]);
    assert!((x[1] - 3.0).abs() < 1e-10, "x[1] = {}", x[1]);
}

#[test]
fn owned_factor_solve_into_rejects_wrong_lengths() {
    let n = 64usize;
    let a = tridiagonal_csr(n, 2.0);
    let solver = SparseLuSolver::default();
    let symbolic = factor_symbolic(&CscMatrix::from_csr(&a));
    let factor = solver
        .factor_sparse_with_symbolic(&a, &symbolic)
        .expect("factor");

    let short_rhs = Array1::from_shape_vec([n - 1], vec![1.0_f64; n - 1]).expect("rhs");
    let mut out = Array1::from_shape_vec([n], vec![0.0_f64; n]).expect("out");
    let err = factor
        .solve_into(&short_rhs.view(), &mut out.view_mut())
        .expect_err("short RHS must be rejected");
    assert!(matches!(err, LetoError::ShapeMismatch { .. }), "{err:?}");

    let rhs = Array1::from_shape_vec([n], vec![1.0_f64; n]).expect("rhs");
    let mut short_out = Array1::from_shape_vec([n - 1], vec![0.0_f64; n - 1]).expect("out");
    let err = factor
        .solve_into(&rhs.view(), &mut short_out.view_mut())
        .expect_err("short output must be rejected");
    assert!(matches!(err, LetoError::ShapeMismatch { .. }), "{err:?}");
}

#[test]
fn owned_factor_rejects_symbolic_order_mismatch() {
    let a = tridiagonal_csr(64, 2.0);
    let wrong = factor_symbolic(&CscMatrix::from_csr(&tridiagonal_csr(32, 2.0)));
    let err = SparseLuSolver::default()
        .factor_sparse_with_symbolic(&a, &wrong)
        .expect_err("order mismatch must be rejected");
    assert!(matches!(err, LetoError::ShapeMismatch { .. }), "{err:?}");
}

#[test]
fn sparse_path_routes_correctly_for_tridiagonal_n64() {
    // n=64 tridiagonal Poisson-Laplacian: density = 64*3 / 64^2 ≈ 0.047
    // (below 0.1), n > small_switch (32) — sparse path must run.
    let n = 64usize;
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0_f64);
        if i > 0 {
            coo.push(i, i - 1, -1.0_f64);
        }
        if i + 1 < n {
            coo.push(i, i + 1, -1.0_f64);
        }
    }
    let a = coo.to_csr();
    let solver = SparseLuSolver::default();
    assert!(
        !solver.use_dense_path(n, a.nnz()),
        "dispatch predicate: sparse path expected for n={n}, nnz={}, density={}",
        a.nnz(),
        (a.nnz() as f64) / ((n as f64) * (n as f64))
    );
    let b = (1..=n).map(|k| k as f64).collect::<Vec<f64>>();
    let b_arr = Array1::from_shape_vec([n], b.clone()).expect("b shape");
    let x = solver.solve_view(&a, &b_arr.view()).expect("sparse solve");
    // Residual against closed-form dense reconstruction.
    let mut max_residual = 0.0_f64;
    for i in 0..n {
        let mut ax = 0.0;
        if i > 0 {
            ax -= x[i - 1];
        }
        ax += 2.0 * x[i];
        if i + 1 < n {
            ax -= x[i + 1];
        }
        let d = (ax - b[i]).abs();
        if d > max_residual {
            max_residual = d;
        }
    }
    assert!(max_residual < 1e-8, "max_residual = {max_residual}");
}
