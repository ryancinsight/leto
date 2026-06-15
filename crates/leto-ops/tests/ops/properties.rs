//! Differential + algebraic-property tests for matrix properties (trace, rank)
//! and structural products (Kronecker), against nalgebra and oracle-independent
//! identities.

use leto::{Array2, Storage};
use leto_ops::{kron, matrix_rank, trace, MatrixProduct, MatrixProperties};
use nalgebra::DMatrix;

const EPS: f64 = 1.0e-9;

#[track_caller]
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= EPS * expected.abs().max(1.0),
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

fn dmatrix(rows: usize, cols: usize, values: &[f64]) -> DMatrix<f64> {
    DMatrix::from_row_slice(rows, cols, values)
}

fn dmatrix_row_major(m: &DMatrix<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.nrows() * m.ncols());
    for r in 0..m.nrows() {
        for c in 0..m.ncols() {
            out.push(m[(r, c)]);
        }
    }
    out
}

// ── trace ───────────────────────────────────────────────────────────────────

#[test]
fn trace_matches_nalgebra_and_trait() {
    let values = vec![6.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
    let a = Array2::from_shape_vec([3, 3], values.clone()).unwrap();
    let expected = dmatrix(3, 3, &values).trace();

    assert_close(trace(&a.view()).unwrap(), expected); // free fn vs nalgebra
    assert_close(a.trace().unwrap(), trace(&a.view()).unwrap()); // trait == free
    assert_close(expected, 15.0); // 6 + 5 + 4
}

#[test]
fn trace_is_scalar_generic_over_integers() {
    let a = Array2::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();
    assert_eq!(trace(&a.view()).unwrap(), 5); // native integer precision
}

#[test]
fn trace_rejects_non_square() {
    let a = Array2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert!(trace(&a.view()).is_err());
}

// ── rank ──────────────────────────────────────────────────────────────────--

#[test]
fn rank_full_and_deficient_match_nalgebra() {
    // Full-rank 3x3 (matches nalgebra rank).
    let full_vals = vec![4.0, -2.0, 1.0, -2.0, 4.0, -2.0, 1.0, -2.0, 4.0];
    let full = Array2::from_shape_vec([3, 3], full_vals.clone()).unwrap();
    assert_eq!(matrix_rank(&full.view()).unwrap(), 3);
    assert_eq!(
        matrix_rank(&full.view()).unwrap(),
        dmatrix(3, 3, &full_vals).rank(1.0e-9)
    );
    assert_eq!(full.rank().unwrap(), 3); // trait == free

    // Rank-deficient: row 2 = 2 * row 1, so rank 1.
    let def_vals = vec![1.0, 2.0, 2.0, 4.0];
    let def = Array2::from_shape_vec([2, 2], def_vals.clone()).unwrap();
    assert_eq!(matrix_rank(&def.view()).unwrap(), 1);
    assert_eq!(
        matrix_rank(&def.view()).unwrap(),
        dmatrix(2, 2, &def_vals).rank(1.0e-9)
    );

    // Tall full-column-rank: rank = min(rows, cols) = 2.
    let tall =
        Array2::from_shape_vec([4, 2], vec![1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0]).unwrap();
    assert_eq!(matrix_rank(&tall.view()).unwrap(), 2);
}

// ── Kronecker product ─────────────────────────────────────────────────────--

#[test]
fn kron_matches_nalgebra_and_trait() {
    let a_vals = vec![1.0, 2.0, 3.0, 4.0];
    let b_vals = vec![0.0, 5.0, 6.0, 7.0];
    let a = Array2::from_shape_vec([2, 2], a_vals.clone()).unwrap();
    let b = Array2::from_shape_vec([2, 2], b_vals.clone()).unwrap();

    let leto_k = kron(&a.view(), &b.view()).unwrap();
    assert_eq!(leto_k.shape(), [4, 4]);

    let expected = dmatrix(2, 2, &a_vals).kronecker(&dmatrix(2, 2, &b_vals));
    assert_close_slice(leto_k.storage().as_slice(), &dmatrix_row_major(&expected));

    // trait == free
    assert_close_slice(
        a.kron(&b).unwrap().storage().as_slice(),
        leto_k.storage().as_slice(),
    );
}

#[test]
fn kron_rectangular_shape_matches_nalgebra() {
    let a_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b_vals = vec![7.0, 8.0]; // 1x2
    let a = Array2::from_shape_vec([2, 3], a_vals.clone()).unwrap();
    let b = Array2::from_shape_vec([1, 2], b_vals.clone()).unwrap();

    let leto_k = kron(&a.view(), &b.view()).unwrap();
    assert_eq!(leto_k.shape(), [2, 6]); // [2*1, 3*2]
    let expected = dmatrix(2, 3, &a_vals).kronecker(&dmatrix(1, 2, &b_vals));
    assert_close_slice(leto_k.storage().as_slice(), &dmatrix_row_major(&expected));
}

#[test]
fn kron_mixed_product_property() {
    // (A ⊗ B)(C ⊗ D) == (A C) ⊗ (B D), oracle-independent identity.
    let a = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 0.0, 3.0]).unwrap();
    let b = Array2::from_shape_vec([2, 2], vec![4.0, 0.0, 1.0, 2.0]).unwrap();
    let c = Array2::from_shape_vec([2, 2], vec![1.0, 1.0, 2.0, 0.0]).unwrap();
    let d = Array2::from_shape_vec([2, 2], vec![0.0, 3.0, 1.0, 1.0]).unwrap();

    let lhs = a.kron(&b).unwrap().matmul(&c.kron(&d).unwrap()).unwrap();
    let rhs = a.matmul(&c).unwrap().kron(&b.matmul(&d).unwrap()).unwrap();
    assert_close_slice(lhs.storage().as_slice(), rhs.storage().as_slice());

    // Trace corollary: tr(A ⊗ B) = tr(A) · tr(B).
    assert_close(
        a.kron(&b).unwrap().trace().unwrap(),
        a.trace().unwrap() * b.trace().unwrap(),
    );
}
