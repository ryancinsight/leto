//! Bunch–Kaufman `P A Pᵀ = L D Lᵀ`: exact reconstruction + LU differential.

use leto::{Array1, Array2, LetoError, Storage};
use leto_ops::{bunch_kaufman, det as lu_det, solve as lu_solve, MatrixDecompose};

#[track_caller]
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-9 * expected.abs().max(1.0),
        "actual {actual} expected {expected}"
    );
}

fn mat(n: usize, data: Vec<f64>) -> Array2<f64> {
    Array2::from_shape_vec([n, n], data).unwrap()
}

/// `(L D Lᵀ)[i,j]` reconstructed by nested loops (small `n`).
fn reconstruct(l: &Array2<f64>, d: &Array2<f64>, n: usize) -> Vec<f64> {
    let ls = l.storage().as_slice();
    let ds = d.storage().as_slice();
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..n {
                for q in 0..n {
                    acc += ls[i * n + p] * ds[p * n + q] * ls[j * n + q];
                }
            }
            m[i * n + j] = acc;
        }
    }
    m
}

/// Assert `L D Lᵀ == P A Pᵀ`, i.e. `(L D Lᵀ)[i,j] == A[perm[i], perm[j]]`.
#[track_caller]
fn assert_reconstructs(a: &Array2<f64>, n: usize) {
    let bk = bunch_kaufman(&a.view()).unwrap();
    let m = reconstruct(&bk.l(), &bk.d(), n);
    let perm = bk.permutation();
    let asl = a.storage().as_slice();
    for i in 0..n {
        for j in 0..n {
            assert_close(m[i * n + j], asl[perm[i] * n + perm[j]]);
        }
    }
}

#[test]
fn reconstructs_zero_diagonal_indefinite() {
    // [[0,1],[1,0]] has zero diagonal — unpivoted UDU fails, Bunch–Kaufman uses
    // a 2×2 pivot.
    let a = mat(2, vec![0.0, 1.0, 1.0, 0.0]);
    assert_reconstructs(&a, 2);
    let bk = bunch_kaufman(&a.view()).unwrap();
    assert!(bk.is_two_by_two()[0], "zero-diagonal forces a 2x2 pivot");
    assert_close(bk.det(), -1.0); // det([[0,1],[1,0]]) = -1
}

#[test]
fn reconstructs_definite_and_indefinite() {
    // Symmetric positive definite.
    assert_reconstructs(
        &mat(3, vec![4.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 6.0]),
        3,
    );
    // Symmetric indefinite (mixed-sign spectrum).
    assert_reconstructs(
        &mat(
            4,
            vec![
                1.0, 2.0, 0.0, 1.0, //
                2.0, 1.0, 3.0, 0.0, //
                0.0, 3.0, 1.0, 2.0, //
                1.0, 0.0, 2.0, 1.0,
            ],
        ),
        4,
    );
}

#[test]
fn reconstructs_one_by_one_symmetric_interchange() {
    // Small a00, large off-diagonal in column 0, and dominant a11 force the
    // Bunch-Kaufman 1x1 pivot with symmetric interchange 0 <-> 1.
    let a = mat(3, vec![0.1, 10.0, 0.0, 10.0, 1000.0, 0.0, 0.0, 0.0, 2.0]);
    let bk = bunch_kaufman(&a.view()).unwrap();
    assert_eq!(bk.permutation()[0], 1);
    assert_reconstructs(&a, 3);
}

#[test]
fn solve_matches_lu() {
    let a = mat(3, vec![2.0, 1.0, 0.0, 1.0, -3.0, 2.0, 0.0, 2.0, 1.0]);
    let b = Array1::from_shape_vec([3], vec![1.0, -2.0, 3.0]).unwrap();
    let bk = bunch_kaufman(&a.view()).unwrap();
    let x_bk = bk.solve(&b.view()).unwrap();
    let x_lu = lu_solve(&a.view(), &b.view()).unwrap();
    for i in 0..3 {
        assert_close(*x_bk.get([i]).unwrap(), *x_lu.get([i]).unwrap());
    }
    // Residual A x ≈ b.
    let asl = a.storage().as_slice();
    for i in 0..3 {
        let mut row = 0.0;
        for j in 0..3 {
            row += asl[i * 3 + j] * *x_bk.get([j]).unwrap();
        }
        assert_close(row, *b.get([i]).unwrap());
    }
}

#[test]
fn det_matches_lu() {
    let a = mat(
        4,
        vec![
            3.0, 1.0, 0.0, 2.0, //
            1.0, -2.0, 1.0, 0.0, //
            0.0, 1.0, 4.0, 1.0, //
            2.0, 0.0, 1.0, -1.0,
        ],
    );
    let bk = bunch_kaufman(&a.view()).unwrap();
    assert_close(bk.det(), lu_det(&a.view()).unwrap());
}

#[test]
fn inverse_satisfies_identity() {
    let a = mat(3, vec![4.0, 1.0, 2.0, 1.0, -3.0, 0.0, 2.0, 0.0, 5.0]);
    let bk = bunch_kaufman(&a.view()).unwrap();
    let inv = bk.inv().unwrap();
    let asl = a.storage().as_slice();
    let isl = inv.storage().as_slice();
    // A · A⁻¹ = I.
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += asl[i * 3 + k] * isl[k * 3 + j];
            }
            assert_close(acc, if i == j { 1.0 } else { 0.0 });
        }
    }
}

#[test]
fn fluent_method_matches_free_function() {
    let a = mat(3, vec![4.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 6.0]);
    let free = bunch_kaufman(&a.view()).unwrap();
    let fluent = a.bunch_kaufman().unwrap();
    assert_close(fluent.det(), free.det());
}

#[test]
fn rejects_non_square_nonsymmetric_nonfinite() {
    let rect = Array2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(
        bunch_kaufman(&rect.view()).unwrap_err(),
        LetoError::ShapeMismatch {
            lhs: vec![2, 3],
            rhs: vec![2, 2],
        }
    );
    let asym = mat(2, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        bunch_kaufman(&asym.view()).unwrap_err(),
        LetoError::StorageError {
            reason: "Bunch-Kaufman requires a symmetric matrix".to_string(),
        }
    );
    let nan = mat(2, vec![1.0, 0.0, 0.0, f64::NAN]);
    assert_eq!(
        bunch_kaufman(&nan.view()).unwrap_err(),
        LetoError::StorageError {
            reason: "Bunch-Kaufman input contains a non-finite value".to_string(),
        }
    );
}
