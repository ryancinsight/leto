//! Tests for column-pivoted (rank-revealing) QR `A P = Q R`.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::{a_posteriori, backward_error};
use leto::{Array, Array2, Storage};
use leto_ops::{col_piv_qr, solve_least_squares};

#[track_caller]
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-9 * expected.abs().max(1.0),
        "actual {actual} expected {expected}"
    );
}

#[track_caller]
fn assert_close_slice(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_close(*a, *e);
    }
}

fn assert_factorization(
    values: &[f64],
    rows: usize,
    columns: usize,
    q: &[f64],
    r: &[f64],
    perm: &[usize],
) {
    assert_eq!(perm.len(), columns);
    let mut sorted_permutation = perm.to_vec();
    sorted_permutation.sort_unstable();
    assert_eq!(sorted_permutation, (0..columns).collect::<Vec<_>>());

    for row in 0..rows {
        for column in 0..columns {
            if row > column {
                assert_eq!(r[row * columns + column], 0.0);
            }
        }
    }

    let mut permuted = vec![0.0; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            permuted[row * columns + column] = values[row * columns + perm[column]];
        }
    }

    let mut square_r = vec![0.0; rows * rows];
    for row in 0..rows {
        for column in 0..columns {
            square_r[row * rows + column] = r[row * columns + column];
        }
    }
    let mut selector = vec![0.0; columns * rows];
    for diagonal in 0..columns.min(rows) {
        selector[diagonal * rows + diagonal] = 1.0;
    }

    let residual = a_posteriori::residual(&permuted, q, &square_r, &selector, rows, columns, rows);
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    let factorization_bound = backward_error::col_piv_qr(rows, columns, f64::EPSILON) * norm;
    assert!(
        residual <= factorization_bound,
        "‖A·P − Q·R‖_F {residual:e} exceeds derived bound {factorization_bound:e}"
    );

    let reflector_error = backward_error::householder(rows, f64::EPSILON);
    let steps = rows.min(columns) as f64;
    let accumulated_error = (steps * reflector_error.ln_1p()).exp_m1();
    let q_error = (rows as f64).sqrt() * accumulated_error;
    let orthogonality_bound = 2.0 * q_error + q_error * q_error;
    let orthogonality = a_posteriori::gram_defect(q, rows, rows);
    assert!(
        orthogonality <= orthogonality_bound,
        "‖QᵀQ − I‖_F {orthogonality:e} exceeds derived bound {orthogonality_bound:e}"
    );
}

#[test]
fn col_piv_qr_reconstructs_a_p() {
    let (m, n) = (4, 3);
    let values = vec![
        4.0, 1.0, -2.0, 2.0, 3.0, 0.0, 1.0, -1.0, 2.0, 0.0, 5.0, -3.0,
    ];
    let a = Array2::from_shape_vec([m, n], values.clone()).unwrap();
    let f = col_piv_qr(&a.view()).unwrap();
    assert_eq!(f.rank(), n);

    let q = f.q();
    let r = f.r();
    assert_eq!(q.shape(), [m, m]);
    assert_eq!(r.shape(), [m, n]);

    assert_factorization(
        &values,
        m,
        n,
        q.storage().as_slice(),
        r.storage().as_slice(),
        f.permutation(),
    );
}

#[test]
fn col_piv_qr_least_squares_matches_qr_and_normal_equations() {
    // Overdetermined, full column rank.
    let (m, n) = (4, 2);
    let a_vals = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
    let b_vals = vec![6.0, 5.0, 7.0, 10.0];
    let a = Array2::from_shape_vec([m, n], a_vals).unwrap();
    let b = Array::from_shape_vec([m], b_vals).unwrap();

    let x = col_piv_qr(&a.view())
        .unwrap()
        .solve_least_squares(&b.view())
        .unwrap();

    // Same least-squares problem as the plain QR solver.
    let x_qr = solve_least_squares(&a.view(), &b.view()).unwrap();
    assert_close_slice(x.storage().as_slice(), x_qr.storage().as_slice());

    // Hand-computed normal-equations solution:
    // AᵀA = [[4,10],[10,30]], Aᵀb = [28,77]
    // det(AᵀA) = 120-100 = 20
    // (AᵀA)⁻¹ = (1/20)[[30,-10],[-10,4]]
    // x = (AᵀA)⁻¹Aᵀb = (1/20)[30*28-10*77, -10*28+4*77] = (1/20)[840-770, -280+308]
    //   = (1/20)[70, 28] = [3.5, 1.4]
    assert_close_slice(x.storage().as_slice(), &[3.5, 1.4]);
}

#[test]
fn col_piv_qr_reveals_rank_deficiency() {
    // Column 2 = column 0 + column 1 ⇒ rank 2 (of 3).
    let (m, n) = (4, 3);
    let values = vec![1.0, 0.0, 1.0, 2.0, 1.0, 3.0, 3.0, 0.0, 3.0, 4.0, 1.0, 5.0];
    let a = Array2::from_shape_vec([m, n], values.clone()).unwrap();
    let f = col_piv_qr(&a.view()).unwrap();
    assert_eq!(f.rank(), 2);
    let q = f.q().storage().as_slice().to_vec();
    let r = f.r().storage().as_slice().to_vec();
    assert_factorization(&values, m, n, &q, &r, f.permutation());
    // Rank-deficient least squares is rejected (not silently wrong).
    let b = Array::from_shape_vec([m], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(f.solve_least_squares(&b.view()).is_err());
}

#[test]
fn col_piv_qr_preserves_squared_norm_order_when_norms_round_together() {
    let delta = f64::EPSILON.sqrt();
    let matrix = Array2::from_shape_vec([2, 2], vec![1.0, 1.0, 0.0, delta]).unwrap();

    let decomposition = col_piv_qr(&matrix.view()).unwrap();

    // The squared norms are 1 and 1 + EPSILON, while sqrt(1 + EPSILON)
    // rounds to 1. Comparing square roots would change the pivot to column 0.
    assert_eq!(decomposition.permutation(), &[1, 0]);
}

#[test]
fn col_piv_qr_recomputes_a_cancelled_partial_norm() {
    let delta = f64::EPSILON.sqrt() / 2.0;
    let values = vec![2.0, 1.0, 1.0, 0.0, delta, 0.0, 0.0, 0.0, 2.0 * delta];
    let matrix = Array2::from_shape_vec([3, 3], values).unwrap();
    let decomposition = col_piv_qr(&matrix.view()).unwrap();

    // With ε = f64::EPSILON and δ² = ε/4, the initial squared norms round to
    // 1 and 1 + ε. Removing row zero estimates tails 0 and ε, while exact
    // tails are ε/4 and ε. Both estimates cross LAPACK's sqrt(ε/2) reliability
    // threshold; this checks recomputed internal state, while the public pivot
    // remains column two.
    assert_eq!(decomposition.permutation(), &[0, 2, 1]);
    assert_eq!(decomposition.rank(), 3);
}

#[test]
fn col_piv_qr_near_tied_tail_order_preserves_the_factorization() {
    // At 2, the spacing is 2ε and 3ε/4 stays below the midpoint; at 1, it
    // exceeds the ε/2 midpoint. Downdated squared norms therefore tie at 1
    // although the exact tails are 1 and 1 + ε, selecting opposite columns.
    let delta = (0.75 * f64::EPSILON).sqrt();
    let values = [2.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, delta];
    let matrix = Array2::from_shape_vec([4, 3], values.to_vec()).unwrap();
    let decomposition = col_piv_qr(&matrix.view()).unwrap();

    assert_eq!(decomposition.permutation(), &[0, 2, 1]);
    assert_eq!(decomposition.rank(), 3);

    let q = decomposition.q().storage().as_slice().to_vec();
    let r = decomposition.r().storage().as_slice().to_vec();
    assert_factorization(&values, 4, 3, &q, &r, decomposition.permutation());
}
