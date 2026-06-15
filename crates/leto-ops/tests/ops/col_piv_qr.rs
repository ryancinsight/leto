//! Tests for column-pivoted (rank-revealing) QR `A P = Q R`.

use leto::{Array, Array2, Storage};
use leto_ops::{col_piv_qr, solve_least_squares, MatrixProduct};
use nalgebra::{DMatrix, DVector};

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

fn assert_orthogonal(q: &Array2<f64>, n: usize) {
    let gram = q.transpose([1, 0]).unwrap().matmul(q).unwrap();
    let g = gram.storage().as_slice();
    for i in 0..n {
        for j in 0..n {
            assert_close(g[i * n + j], if i == j { 1.0 } else { 0.0 });
        }
    }
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

    assert_orthogonal(&q, m);

    // R upper triangular (zero strictly below the diagonal).
    let rs = r.storage().as_slice();
    for i in 0..m {
        for j in 0..n {
            if i > j {
                assert_close(rs[i * n + j], 0.0);
            }
        }
    }

    // A P = Q R: (A P)[i][k] = A[i][perm[k]].
    let qr = q.matmul(&r).unwrap();
    let perm = f.permutation();
    let mut ap = vec![0.0; m * n];
    for i in 0..m {
        for k in 0..n {
            ap[i * n + k] = values[i * n + perm[k]];
        }
    }
    assert_close_slice(qr.storage().as_slice(), &ap);
}

#[test]
fn col_piv_qr_least_squares_matches_qr_and_normal_equations() {
    // Overdetermined, full column rank.
    let (m, n) = (4, 2);
    let a_vals = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
    let b_vals = vec![6.0, 5.0, 7.0, 10.0];
    let a = Array2::from_shape_vec([m, n], a_vals.clone()).unwrap();
    let b = Array::from_shape_vec([m], b_vals.clone()).unwrap();

    let x = col_piv_qr(&a.view())
        .unwrap()
        .solve_least_squares(&b.view())
        .unwrap();

    // Same least-squares problem as the plain QR solver.
    let x_qr = solve_least_squares(&a.view(), &b.view()).unwrap();
    assert_close_slice(x.storage().as_slice(), x_qr.storage().as_slice());

    // And the normal-equations solution via nalgebra.
    let na = DMatrix::from_row_slice(m, n, &a_vals);
    let na_b = DVector::from_vec(b_vals);
    let normal = (na.transpose() * &na)
        .lu()
        .solve(&(na.transpose() * &na_b))
        .unwrap();
    assert_close_slice(x.storage().as_slice(), normal.as_slice());
}

#[test]
fn col_piv_qr_reveals_rank_deficiency() {
    // Column 2 = column 0 + column 1 ⇒ rank 2 (of 3).
    let (m, n) = (4, 3);
    let values = vec![1.0, 0.0, 1.0, 2.0, 1.0, 3.0, 3.0, 0.0, 3.0, 4.0, 1.0, 5.0];
    let a = Array2::from_shape_vec([m, n], values).unwrap();
    let f = col_piv_qr(&a.view()).unwrap();
    assert_eq!(f.rank(), 2);
    // Rank-deficient least squares is rejected (not silently wrong).
    let b = Array::from_shape_vec([m], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(f.solve_least_squares(&b.view()).is_err());
}
