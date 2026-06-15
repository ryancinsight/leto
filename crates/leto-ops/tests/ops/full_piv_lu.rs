//! Tests for complete-pivoting LU `P A Q = L U` (rank-revealing).

use leto::{Array, Array2, Storage};
use leto_ops::{det, full_piv_lu, solve, MatrixProduct};
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

#[test]
fn full_piv_lu_reconstructs_p_a_q() {
    let n = 4;
    let values = vec![
        2.0, 5.0, -2.0, 2.0, 1.0, 2.0, 3.0, 1.0, -2.0, 4.0, 3.0, -2.0, 2.0, 1.0, -1.0, -1.0,
    ];
    let a = Array2::from_shape_vec([n, n], values.clone()).unwrap();
    let f = full_piv_lu(&a.view()).unwrap();
    assert_eq!(f.rank(), n);

    // L·U
    let lu = f.l().matmul(&f.u()).unwrap();

    // P A Q: (PAQ)[k][j] = A[row_perm[k]][col_perm[j]].
    let rp = f.row_permutation();
    let cp = f.col_permutation();
    let mut paq = vec![0.0; n * n];
    for k in 0..n {
        for j in 0..n {
            paq[k * n + j] = values[rp[k] * n + cp[j]];
        }
    }
    assert_close_slice(lu.storage().as_slice(), &paq);
}

#[test]
fn full_piv_lu_det_solve_inv_match_oracles() {
    let n = 3;
    let values = vec![4.0, -2.0, 1.0, -2.0, 4.0, -2.0, 1.0, -2.0, 4.0];
    let rhs_values = vec![11.0, -16.0, 17.0];
    let a = Array2::from_shape_vec([n, n], values.clone()).unwrap();
    let rhs = Array::from_shape_vec([n], rhs_values.clone()).unwrap();
    let f = full_piv_lu(&a.view()).unwrap();

    // Determinant vs partial-pivot LU det and nalgebra.
    let na = DMatrix::from_row_slice(n, n, &values);
    assert_close(f.det(), det(&a.view()).unwrap());
    assert_close(f.det(), na.determinant());

    // Solve vs leto partial-pivot solve and nalgebra full-pivot solve.
    let x = f.solve(&rhs.view()).unwrap();
    let leto_solve = solve(&a.view(), &rhs.view()).unwrap();
    assert_close_slice(x.storage().as_slice(), leto_solve.storage().as_slice());
    let na_x = na
        .clone()
        .full_piv_lu()
        .solve(&DVector::from_vec(rhs_values))
        .unwrap();
    assert_close_slice(x.storage().as_slice(), na_x.as_slice());

    // Inverse vs nalgebra.
    let inv = f.inv().unwrap();
    let na_inv = na.try_inverse().unwrap();
    let mut expected = Vec::with_capacity(n * n);
    for r in 0..n {
        for c in 0..n {
            expected.push(na_inv[(r, c)]);
        }
    }
    assert_close_slice(inv.storage().as_slice(), &expected);
}

#[test]
fn full_piv_lu_reveals_rank_deficiency() {
    // Row 2 = 2 · row 1 ⇒ rank 2 (of 3), determinant 0.
    let n = 3;
    let values = vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0];
    let a = Array2::from_shape_vec([n, n], values.clone()).unwrap();
    let f = full_piv_lu(&a.view()).unwrap();

    // Complete pivoting reveals the exact rank (2) robustly — more reliably than
    // the Gram-spectrum `matrix_rank`, whose condition-squaring inflates the
    // near-zero singular value on this borderline case.
    assert_eq!(f.rank(), 2);
    assert_close(f.det(), 0.0);
    // Rank-deficient solve/inverse must be rejected, not silently wrong.
    assert!(f.inv().is_err());
}

#[test]
fn full_piv_lu_rejects_non_square() {
    let a = Array2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert!(full_piv_lu(&a.view()).is_err());
}
