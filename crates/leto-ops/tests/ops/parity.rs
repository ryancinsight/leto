//! Completeness parity harness: differential value comparison of Leto against
//! `ndarray` 0.16 and `nalgebra` 0.35.
//!
//! This module is the executable form of the completeness matrix in
//! `docs/completeness/parity_matrix.md`. Each test maps to one matrix row and
//! asserts value-semantic parity (bounded epsilon) between a Leto operation and
//! the corresponding oracle. It is the broad array/reduction/matmul/structure
//! companion to `oracle_parity.rs`, which owns the dense linear-algebra
//! decompositions (LU, Cholesky, symmetric eigen, SVD).
//!
//! Coverage gaps recorded in the matrix (operations Leto does not yet expose)
//! are tracked there as `MISSING` rows, not as ignored stubs here: a test exists
//! only for surface Leto actually implements, so a green run is an honest parity
//! signal and never test-gaming.

use leto::{concat, stack, Array, Array2, SliceArg, Storage};
use leto_ops::{
    add, batched_matmul, cumsum, div, dot, matmul, mean_axis, mul, scalar_map, solve_least_squares,
    sub, sum, sum_axis, unary_map, AddOp, ExpOp, MulOp, SqrtOp,
};
use nalgebra::{DMatrix, DVector};
use ndarray::{concatenate, stack as nd_stack, Array1 as NdArray1, Array2 as NdArray2, Axis};

const EPS: f64 = 1.0e-9;

#[track_caller]
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= EPS * expected.abs().max(1.0),
        "actual {actual} expected {expected} (delta {})",
        (actual - expected).abs()
    );
}

#[track_caller]
fn assert_close_slice(actual: &[f64], expected: &[f64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_close(*a, *e);
    }
}

fn seq(len: usize, scale: f64, bias: f64) -> Vec<f64> {
    (0..len).map(|i| i as f64 * scale + bias).collect()
}

// ── Elementwise binary parity (ndarray oracle) ──────────────────────────────

#[test]
fn elementwise_add_sub_mul_div_match_ndarray() {
    let shape = [3usize, 4];
    let len = shape[0] * shape[1];
    let a_vals = seq(len, 0.5, 1.0);
    let b_vals = seq(len, 0.25, 2.0);
    let a = Array2::from_shape_vec(shape, a_vals.clone()).unwrap();
    let b = Array2::from_shape_vec(shape, b_vals.clone()).unwrap();
    let nd_a = NdArray2::from_shape_vec((shape[0], shape[1]), a_vals).unwrap();
    let nd_b = NdArray2::from_shape_vec((shape[0], shape[1]), b_vals).unwrap();

    let mut out = Array2::zeros(shape);
    add(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_close_slice(
        out.storage().as_slice(),
        (&nd_a + &nd_b).as_slice().unwrap(),
    );

    let mut out = Array2::zeros(shape);
    sub(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_close_slice(
        out.storage().as_slice(),
        (&nd_a - &nd_b).as_slice().unwrap(),
    );

    let mut out = Array2::zeros(shape);
    mul(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_close_slice(
        out.storage().as_slice(),
        (&nd_a * &nd_b).as_slice().unwrap(),
    );

    let mut out = Array2::zeros(shape);
    div(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_close_slice(
        out.storage().as_slice(),
        (&nd_a / &nd_b).as_slice().unwrap(),
    );
}

#[test]
fn scalar_add_mul_match_ndarray() {
    let shape = [2usize, 5];
    let len = shape[0] * shape[1];
    let vals = seq(len, 0.7, -1.0);
    let a = Array2::from_shape_vec(shape, vals.clone()).unwrap();
    let nd = NdArray2::from_shape_vec((shape[0], shape[1]), vals).unwrap();

    let leto_add = scalar_map::<AddOp, f64, 2>(&a.view(), 3.5).unwrap();
    assert_close_slice(
        leto_add.storage().as_slice(),
        (&nd + 3.5).as_slice().unwrap(),
    );

    let leto_mul = scalar_map::<MulOp, f64, 2>(&a.view(), 2.0).unwrap();
    assert_close_slice(
        leto_mul.storage().as_slice(),
        (&nd * 2.0).as_slice().unwrap(),
    );
}

#[test]
fn unary_exp_sqrt_match_ndarray() {
    let shape = [4usize, 4];
    let len = shape[0] * shape[1];
    let vals = seq(len, 0.13, 0.5);
    let a = Array2::from_shape_vec(shape, vals.clone()).unwrap();
    let nd = NdArray2::from_shape_vec((shape[0], shape[1]), vals).unwrap();

    let leto_exp = unary_map(ExpOp, &a.view()).unwrap();
    assert_close_slice(
        leto_exp.storage().as_slice(),
        nd.mapv(f64::exp).as_slice().unwrap(),
    );

    let leto_sqrt = unary_map(SqrtOp, &a.view()).unwrap();
    assert_close_slice(
        leto_sqrt.storage().as_slice(),
        nd.mapv(f64::sqrt).as_slice().unwrap(),
    );
}

// ── Reduction parity (ndarray oracle) ───────────────────────────────────────

#[test]
fn sum_all_matches_ndarray() {
    let shape = [8usize, 8];
    let len = shape[0] * shape[1];
    let vals = seq(len, 0.31, -3.0);
    let a = Array2::from_shape_vec(shape, vals.clone()).unwrap();
    let nd = NdArray2::from_shape_vec((shape[0], shape[1]), vals).unwrap();
    assert_close(sum(&a.view()), nd.sum());
}

#[test]
fn sum_mean_axis_match_ndarray() {
    let shape = [3usize, 5];
    let len = shape[0] * shape[1];
    let vals = seq(len, 0.41, 1.0);
    let a = Array2::from_shape_vec(shape, vals.clone()).unwrap();
    let nd = NdArray2::from_shape_vec((shape[0], shape[1]), vals).unwrap();

    for axis in 0..2usize {
        let leto_sum = sum_axis(&a.view(), axis).unwrap();
        assert_close_slice(
            leto_sum.storage().as_slice(),
            nd.sum_axis(Axis(axis)).as_slice().unwrap(),
        );
        let leto_mean = mean_axis(&a.view(), axis).unwrap();
        assert_close_slice(
            leto_mean.storage().as_slice(),
            nd.mean_axis(Axis(axis)).unwrap().as_slice().unwrap(),
        );
    }
}

#[test]
fn cumsum_matches_ndarray_accumulate() {
    let shape = [3usize, 4];
    let len = shape[0] * shape[1];
    let vals = seq(len, 1.0, 0.0);
    let a = Array2::from_shape_vec(shape, vals.clone()).unwrap();
    let leto_cs = cumsum(&a.view(), 1).unwrap();

    // ndarray has no native cumsum: reference accumulate along axis 1.
    let nd = NdArray2::from_shape_vec((shape[0], shape[1]), vals).unwrap();
    let mut expected = nd.clone();
    for mut row in expected.rows_mut() {
        let mut acc = 0.0;
        for v in row.iter_mut() {
            acc += *v;
            *v = acc;
        }
    }
    assert_close_slice(leto_cs.storage().as_slice(), expected.as_slice().unwrap());
}

// ── Matmul / dot parity (ndarray oracle) ────────────────────────────────────

#[test]
fn matmul_matches_ndarray_dot() {
    let (m, k, n) = (5usize, 3, 4);
    let a_vals = seq(m * k, 0.2, 1.0);
    let b_vals = seq(k * n, 0.3, -1.0);
    let a = Array2::from_shape_vec([m, k], a_vals.clone()).unwrap();
    let b = Array2::from_shape_vec([k, n], b_vals.clone()).unwrap();
    let mut out = Array2::zeros([m, n]);
    matmul(&a.view(), &b.view(), &mut out.view_mut()).unwrap();

    let nd_a = NdArray2::from_shape_vec((m, k), a_vals).unwrap();
    let nd_b = NdArray2::from_shape_vec((k, n), b_vals).unwrap();
    assert_close_slice(
        out.storage().as_slice(),
        nd_a.dot(&nd_b).as_slice().unwrap(),
    );
}

#[test]
fn matmul_transposed_rhs_matches_ndarray() {
    let n = 6usize;
    let a_vals = seq(n * n, 0.11, 1.0);
    let b_vals = seq(n * n, 0.07, 2.0);
    let a = Array2::from_shape_vec([n, n], a_vals.clone()).unwrap();
    let b = Array2::from_shape_vec([n, n], b_vals.clone()).unwrap();
    let bt = b.transpose([1, 0]).unwrap();
    let mut out = Array2::zeros([n, n]);
    matmul(&a.view(), &bt, &mut out.view_mut()).unwrap();

    let nd_a = NdArray2::from_shape_vec((n, n), a_vals).unwrap();
    let nd_b = NdArray2::from_shape_vec((n, n), b_vals).unwrap();
    let expected = nd_a.dot(&nd_b.t());
    assert_close_slice(out.storage().as_slice(), expected.as_slice().unwrap());
}

#[test]
fn batched_matmul_matches_ndarray_per_batch() {
    let (batch, m, k, n) = (2usize, 3, 4, 2);
    let a_vals = seq(batch * m * k, 0.05, 1.0);
    let b_vals = seq(batch * k * n, 0.09, -2.0);
    let a = Array::from_shape_vec([batch, m, k], a_vals.clone()).unwrap();
    let b = Array::from_shape_vec([batch, k, n], b_vals.clone()).unwrap();
    let mut out = Array::zeros([batch, m, n]);
    batched_matmul(&a.view(), &b.view(), &mut out.view_mut()).unwrap();

    let mut expected = Vec::with_capacity(batch * m * n);
    for bi in 0..batch {
        let a_slice = &a_vals[bi * m * k..(bi + 1) * m * k];
        let b_slice = &b_vals[bi * k * n..(bi + 1) * k * n];
        let nd_a = NdArray2::from_shape_vec((m, k), a_slice.to_vec()).unwrap();
        let nd_b = NdArray2::from_shape_vec((k, n), b_slice.to_vec()).unwrap();
        expected.extend_from_slice(nd_a.dot(&nd_b).as_slice().unwrap());
    }
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn dot_matches_ndarray() {
    let len = 32usize;
    let a_vals = seq(len, 0.5, 1.0);
    let b_vals = seq(len, 0.25, -1.0);
    let a = Array::from_shape_vec([len], a_vals.clone()).unwrap();
    let b = Array::from_shape_vec([len], b_vals.clone()).unwrap();
    let nd_a = NdArray1::from_vec(a_vals);
    let nd_b = NdArray1::from_vec(b_vals);
    assert_close(dot(&a.view(), &b.view()).unwrap(), nd_a.dot(&nd_b));
}

// ── Structure parity (ndarray oracle) ───────────────────────────────────────

#[test]
fn concat_matches_ndarray_concatenate() {
    let a = Array2::from_shape_vec([2, 3], seq(6, 1.0, 0.0)).unwrap();
    let b = Array2::from_shape_vec([2, 3], seq(6, 1.0, 6.0)).unwrap();
    let leto_cat = concat(&[a.view(), b.view()], 0).unwrap();

    let nd_a = NdArray2::from_shape_vec((2, 3), seq(6, 1.0, 0.0)).unwrap();
    let nd_b = NdArray2::from_shape_vec((2, 3), seq(6, 1.0, 6.0)).unwrap();
    let expected = concatenate(Axis(0), &[nd_a.view(), nd_b.view()]).unwrap();
    assert_close_slice(leto_cat.storage().as_slice(), expected.as_slice().unwrap());
}

#[test]
fn stack_matches_ndarray_stack() {
    let a = Array2::from_shape_vec([2, 3], seq(6, 1.0, 0.0)).unwrap();
    let b = Array2::from_shape_vec([2, 3], seq(6, 1.0, 6.0)).unwrap();
    let leto_stacked = stack::<f64, 2, 3>(&[a.view(), b.view()], 0).unwrap();

    let nd_a = NdArray2::from_shape_vec((2, 3), seq(6, 1.0, 0.0)).unwrap();
    let nd_b = NdArray2::from_shape_vec((2, 3), seq(6, 1.0, 6.0)).unwrap();
    let expected = nd_stack(Axis(0), &[nd_a.view(), nd_b.view()]).unwrap();
    assert_close_slice(
        leto_stacked.storage().as_slice(),
        expected.as_slice().unwrap(),
    );
}

// ── Linear algebra parity not covered by oracle_parity.rs (nalgebra oracle) ──

#[test]
fn least_squares_matches_nalgebra_qr() {
    // Overdetermined system: 4 equations, 2 unknowns, full column rank.
    let rows = 4usize;
    let cols = 2usize;
    let a_vals = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
    let rhs_vals = vec![6.0, 5.0, 7.0, 10.0];
    let a = Array2::from_shape_vec([rows, cols], a_vals.clone()).unwrap();
    let rhs = Array::from_shape_vec([rows], rhs_vals.clone()).unwrap();
    let leto_x = solve_least_squares(&a.view(), &rhs.view()).unwrap();

    // nalgebra `QR::solve` rejects non-square systems; the least-squares oracle
    // is the normal-equations solution (A^T A) x = A^T b for full-column-rank A.
    let nd_a = DMatrix::from_row_slice(rows, cols, &a_vals);
    let nd_rhs = DVector::from_vec(rhs_vals);
    let ata = nd_a.transpose() * &nd_a;
    let atb = nd_a.transpose() * &nd_rhs;
    let expected = ata.lu().solve(&atb).unwrap();
    assert_close_slice(leto_x.storage().as_slice(), expected.as_slice());
}

// ── Reverse-strided cross-check (ndarray oracle) ────────────────────────────

#[test]
fn reverse_axis_sum_matches_ndarray() {
    let n = 6usize;
    let vals = seq(n * n, 0.5, -4.0);
    let a = Array2::from_shape_vec([n, n], vals.clone()).unwrap();
    let reversed = a
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();
    let nd = NdArray2::from_shape_vec((n, n), vals).unwrap();
    let nd_rev = nd.slice(ndarray::s![.., ..;-1]);
    assert_close(sum(&reversed), nd_rev.sum());
}
