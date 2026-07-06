//! Differential + algebraic-property tests for matrix properties (trace, rank)
//! and structural products (Kronecker), against nalgebra and oracle-independent
//! identities.

use leto::{Array2, LetoError, SliceArg, Storage};
use leto_ops::application::{
    kron as application_kron, matrix_rank as application_matrix_rank, trace as application_trace,
};
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

fn view_row_major<const R: usize, const C: usize>(
    view: &leto::ArrayView<'_, f64, 2>,
    shape: [usize; 2],
) -> Vec<f64> {
    let mut out = Vec::with_capacity(shape[0] * shape[1]);
    for row in 0..R {
        for col in 0..C {
            out.push(*view.get([row, col]).unwrap());
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
    assert_eq!(
        trace(&a.view()).unwrap_err(),
        LetoError::ShapeMismatch {
            lhs: vec![2, 3],
            rhs: vec![2, 2],
        }
    );
}

#[test]
fn trace_handles_negative_stride_square_view() {
    let a = Array2::from_shape_vec(
        [4, 4],
        vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],
    )
    .unwrap();
    let reversed = a
        .slice_with::<2>(&[
            SliceArg::range(Some(-1), None, -1),
            SliceArg::range(Some(-1), None, -1),
        ])
        .unwrap();

    assert_eq!(trace(&reversed).unwrap(), 30.0);
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

#[test]
fn rank_with_tolerance_matches_free_function_and_nalgebra() {
    let values = vec![1.0, 0.0, 0.0, 1.0e-12];
    let matrix = Array2::from_shape_vec([2, 2], values.clone()).unwrap();
    let tolerance = 1.0e-9;

    assert_eq!(matrix.rank_with_tolerance(tolerance).unwrap(), 1);
    assert_eq!(
        matrix.rank_with_tolerance(tolerance).unwrap(),
        leto_ops::matrix_rank_with_tolerance(&matrix.view(), tolerance).unwrap()
    );
    assert_eq!(
        matrix.rank_with_tolerance(tolerance).unwrap(),
        dmatrix(2, 2, &values).rank(tolerance)
    );
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
fn kron_handles_negative_stride_views() {
    let a = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Array2::from_shape_vec([2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let a_reversed_rows = a
        .slice_with::<2>(&[SliceArg::range(Some(-1), None, -1), SliceArg::All])
        .unwrap();
    let b_reversed_cols = b
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(Some(-1), None, -1)])
        .unwrap();

    let leto_k = kron(&a_reversed_rows, &b_reversed_cols).unwrap();
    let expected = dmatrix(2, 2, &view_row_major::<2, 2>(&a_reversed_rows, [2, 2])).kronecker(
        &dmatrix(2, 2, &view_row_major::<2, 2>(&b_reversed_cols, [2, 2])),
    );

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

#[test]
fn application_reexports_match_crate_root_exports() {
    let a = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Array2::from_shape_vec([1, 2], vec![5.0, 6.0]).unwrap();

    assert_eq!(
        application_trace(&a.view()).unwrap(),
        trace(&a.view()).unwrap()
    );
    assert_eq!(
        application_matrix_rank(&a.view()).unwrap(),
        matrix_rank(&a.view()).unwrap()
    );
    assert_close_slice(
        application_kron(&a.view(), &b.view())
            .unwrap()
            .storage()
            .as_slice(),
        kron(&a.view(), &b.view()).unwrap().storage().as_slice(),
    );
}
