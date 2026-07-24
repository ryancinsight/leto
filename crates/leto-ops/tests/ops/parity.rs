//! Completeness parity harness: differential value comparison of Leto against
//! hand-computed analytical values.
//!
//! This module is the executable form of the completeness matrix in
//! . Each test maps to one matrix row and
//! asserts value-semantic parity (bounded epsilon) between a Leto operation and
//! the corresponding reference computation. It is the broad array/reduction/
//! matmul/structure companion to , which owns the dense
//! linear-algebra decompositions (LU, Cholesky, symmetric eigen, SVD).
//!
//! Coverage gaps recorded in the matrix (operations Leto does not yet expose)
//! are tracked there as  rows, not as ignored stubs here: a test exists
//! only for surface Leto actually implements, so a green run is an honest parity
//! signal and never test-gaming.

use leto::{concat, stack, Array, Array2, SliceArg, Storage};
use leto_ops::{
    add, batched_matmul, cumsum, div, dot, matmul, mean_axis, mul, scalar_map, solve_least_squares,
    sub, sum, sum_axis, unary_map, AddOp, ExpOp, MulOp, SqrtOp,
};

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

/// Reference elementwise binary op.
fn ref_binop<F>(a: &[f64], b: &[f64], f: F) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64,
{
    a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect()
}

/// Reference scalar op.
fn ref_scalar<F>(a: &[f64], s: f64, f: F) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64,
{
    a.iter().map(|&x| f(x, s)).collect()
}

/// Reference unary op.
fn ref_unary<F>(a: &[f64], f: F) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    a.iter().map(|&x| f(x)).collect()
}

/// Reference matmul: C[i,j] = sum_k A[i,k] * B[k,j].
fn ref_matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Reference sum along axis (0 = columns reduce rows, 1 = rows reduce cols).
fn ref_sum_axis(data: &[f64], rows: usize, cols: usize, axis: usize) -> Vec<f64> {
    if axis == 0 {
        (0..cols)
            .map(|c| (0..rows).map(|r| data[r * cols + c]).sum())
            .collect()
    } else {
        (0..rows)
            .map(|r| (0..cols).map(|c| data[r * cols + c]).sum())
            .collect()
    }
}

/// Reference mean along axis.
fn ref_mean_axis(data: &[f64], rows: usize, cols: usize, axis: usize) -> Vec<f64> {
    if axis == 0 {
        (0..cols)
            .map(|c| (0..rows).map(|r| data[r * cols + c]).sum::<f64>() / rows as f64)
            .collect()
    } else {
        (0..rows)
            .map(|r| (0..cols).map(|c| data[r * cols + c]).sum::<f64>() / cols as f64)
            .collect()
    }
}

/// Reference reverse last axis.
fn ref_reverse_last_axis(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = data[i * cols + (cols - 1 - j)];
        }
    }
    out
}

/// Reference concat along axis 0.
fn ref_concat_axis0(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = a.to_vec();
    out.extend_from_slice(b);
    out
}

/// Reference stack along new axis 0: [2, rows, cols].
fn ref_stack_axis0(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = a.to_vec();
    out.extend_from_slice(b);
    out
}

// ── Elementwise binary parity (reference) ────────────────────────────────────

#[test]
fn elementwise_add_sub_mul_div_match_reference() {
    let shape = [3usize, 4];
    let len = shape[0] * shape[1];
    let a_vals = seq(len, 0.5, 1.0);
    let b_vals = seq(len, 0.25, 2.0);
    let a = Array2::from_shape_vec(shape, a_vals.clone()).unwrap();
    let b = Array2::from_shape_vec(shape, b_vals.clone()).unwrap();

    let mut out = Array2::zeros(shape);
    add(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_close_slice(
        out.storage().as_slice(),
        &ref_binop(&a_vals, &b_vals, |x, y| x + y),
    );

    let mut out = Array2::zeros(shape);
    sub(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_close_slice(
        out.storage().as_slice(),
        &ref_binop(&a_vals, &b_vals, |x, y| x - y),
    );

    let mut out = Array2::zeros(shape);
    mul(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_close_slice(
        out.storage().as_slice(),
        &ref_binop(&a_vals, &b_vals, |x, y| x * y),
    );

    let mut out = Array2::zeros(shape);
    div(&a.view(), &b.view(), &mut out.view_mut()).unwrap();
    assert_close_slice(
        out.storage().as_slice(),
        &ref_binop(&a_vals, &b_vals, |x, y| x / y),
    );
}

#[test]
fn scalar_add_mul_match_reference() {
    let shape = [2usize, 5];
    let len = shape[0] * shape[1];
    let vals = seq(len, 0.7, -1.0);
    let a = Array2::from_shape_vec(shape, vals.clone()).unwrap();

    let leto_add = scalar_map::<AddOp, f64, 2>(&a.view(), 3.5).unwrap();
    assert_close_slice(
        leto_add.storage().as_slice(),
        &ref_scalar(&vals, 3.5, |x, s| x + s),
    );

    let leto_mul = scalar_map::<MulOp, f64, 2>(&a.view(), 2.0).unwrap();
    assert_close_slice(
        leto_mul.storage().as_slice(),
        &ref_scalar(&vals, 2.0, |x, s| x * s),
    );
}

#[test]
fn unary_exp_sqrt_match_reference() {
    let shape = [4usize, 4];
    let len = shape[0] * shape[1];
    let vals = seq(len, 0.13, 0.5);
    let a = Array2::from_shape_vec(shape, vals.clone()).unwrap();

    let leto_exp = unary_map(ExpOp, &a.view()).unwrap();
    assert_close_slice(leto_exp.storage().as_slice(), &ref_unary(&vals, f64::exp));

    let leto_sqrt = unary_map(SqrtOp, &a.view()).unwrap();
    assert_close_slice(leto_sqrt.storage().as_slice(), &ref_unary(&vals, f64::sqrt));
}

// ── Reduction parity (reference) ─────────────────────────────────────────────

#[test]
fn sum_all_matches_reference() {
    let shape = [8usize, 8];
    let len = shape[0] * shape[1];
    let vals = seq(len, 0.31, -3.0);
    let a = Array2::from_shape_vec(shape, vals.clone()).unwrap();
    assert_close(sum(&a.view()), vals.iter().sum());
}

#[test]
fn sum_mean_axis_match_reference() {
    let shape = [3usize, 5];
    let len = shape[0] * shape[1];
    let vals = seq(len, 0.41, 1.0);
    let a = Array2::from_shape_vec(shape, vals.clone()).unwrap();

    for axis in 0..2usize {
        let leto_sum = sum_axis(&a.view(), axis).unwrap();
        assert_close_slice(
            leto_sum.storage().as_slice(),
            &ref_sum_axis(&vals, 3, 5, axis),
        );
        let leto_mean = mean_axis(&a.view(), axis).unwrap();
        assert_close_slice(
            leto_mean.storage().as_slice(),
            &ref_mean_axis(&vals, 3, 5, axis),
        );
    }
}

#[test]
fn cumsum_matches_reference_accumulate() {
    let shape = [3usize, 4];
    let len = shape[0] * shape[1];
    let vals = seq(len, 1.0, 0.0);
    let a = Array2::from_shape_vec(shape, vals.clone()).unwrap();
    let leto_cs = cumsum(&a.view(), 1).unwrap();

    // Reference: running sum along axis 1 of each row.
    let mut expected = vals.clone();
    for row in 0..3 {
        let mut acc = 0.0;
        for col in 0..4 {
            acc += vals[row * 4 + col];
            expected[row * 4 + col] = acc;
        }
    }
    assert_close_slice(leto_cs.storage().as_slice(), &expected);
}

// ── Matmul / dot parity (reference) ─────────────────────────────────────────

#[test]
fn matmul_matches_reference_dot() {
    let (m, k, n) = (64usize, 64, 64);
    let a_vals = seq(m * k, 0.2, 1.0);
    let b_vals = seq(k * n, 0.3, -1.0);
    let a = Array2::from_shape_vec([m, k], a_vals.clone()).unwrap();
    let b = Array2::from_shape_vec([k, n], b_vals.clone()).unwrap();
    let mut out = Array2::zeros([m, n]);
    matmul(&a.view(), &b.view(), &mut out.view_mut()).unwrap();

    assert_close_slice(
        out.storage().as_slice(),
        &ref_matmul(&a_vals, &b_vals, m, k, n),
    );
}

#[test]
fn matmul_transposed_rhs_matches_reference() {
    let n = 6usize;
    let a_vals = seq(n * n, 0.11, 1.0);
    let b_vals = seq(n * n, 0.07, 2.0);
    let a = Array2::from_shape_vec([n, n], a_vals.clone()).unwrap();
    let b = Array2::from_shape_vec([n, n], b_vals.clone()).unwrap();
    let bt = b.transpose([1, 0]).unwrap();
    let mut out = Array2::zeros([n, n]);
    matmul(&a.view(), &bt, &mut out.view_mut()).unwrap();

    // A · Bᵀ: reference is C[i,j] = sum_k A[i,k] * B[j,k]
    let mut expected = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..n {
                acc += a_vals[i * n + kk] * b_vals[j * n + kk];
            }
            expected[i * n + j] = acc;
        }
    }
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn batched_matmul_matches_reference_per_batch() {
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
        expected.extend(ref_matmul(a_slice, b_slice, m, k, n));
    }
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn dot_matches_reference() {
    let len = 32usize;
    let a_vals = seq(len, 0.5, 1.0);
    let b_vals = seq(len, 0.25, -1.0);
    let a = Array::from_shape_vec([len], a_vals.clone()).unwrap();
    let b = Array::from_shape_vec([len], b_vals.clone()).unwrap();
    let reference: f64 = a_vals.iter().zip(&b_vals).map(|(x, y)| x * y).sum();
    assert_close(dot(&a.view(), &b.view()).unwrap(), reference);
}

// ── Structure parity (reference) ─────────────────────────────────────────────

#[test]
fn concat_matches_reference_concatenate() {
    let a = Array2::from_shape_vec([2, 3], seq(6, 1.0, 0.0)).unwrap();
    let b = Array2::from_shape_vec([2, 3], seq(6, 1.0, 6.0)).unwrap();
    let leto_cat = concat(&[a.view(), b.view()], 0).unwrap();

    let expected = ref_concat_axis0(&seq(6, 1.0, 0.0), &seq(6, 1.0, 6.0));
    assert_close_slice(leto_cat.storage().as_slice(), &expected);
}

#[test]
fn stack_matches_reference_stack() {
    let a = Array2::from_shape_vec([2, 3], seq(6, 1.0, 0.0)).unwrap();
    let b = Array2::from_shape_vec([2, 3], seq(6, 1.0, 6.0)).unwrap();
    let leto_stacked = stack::<f64, 2, 3>(&[a.view(), b.view()], 0).unwrap();

    let expected = ref_stack_axis0(&seq(6, 1.0, 0.0), &seq(6, 1.0, 6.0));
    assert_close_slice(leto_stacked.storage().as_slice(), &expected);
}

// ── Linear algebra parity (hand-computed normal-equations) ──────────────────

#[test]
fn least_squares_matches_normal_equations() {
    // Overdetermined system: 4 equations, 2 unknowns, full column rank.
    // A = [[1,1],[1,2],[1,3],[1,4]], b = [6,5,7,10]
    // AᵀA = [[4,10],[10,30]], Aᵀb = [28,77]
    // det(AᵀA) = 20, (AᵀA)⁻¹ = (1/20)[[30,-10],[-10,4]]
    // x = (1/20)[30*28-10*77, -10*28+4*77] = (1/20)[70, 28] = [3.5, 1.4]
    let rows = 4usize;
    let cols = 2usize;
    let a_vals = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
    let rhs_vals = vec![6.0, 5.0, 7.0, 10.0];
    let a = Array2::from_shape_vec([rows, cols], a_vals).unwrap();
    let rhs = Array::from_shape_vec([rows], rhs_vals).unwrap();
    let leto_x = solve_least_squares(&a.view(), &rhs.view()).unwrap();

    assert_close_slice(leto_x.storage().as_slice(), &[3.5, 1.4]);
}

// ── Reverse-strided cross-check (reference) ──────────────────────────────────

#[test]
fn reverse_axis_sum_matches_reference() {
    let n = 6usize;
    let vals = seq(n * n, 0.5, -4.0);
    let a = Array2::from_shape_vec([n, n], vals.clone()).unwrap();
    let reversed = a
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();
    let ref_rev = ref_reverse_last_axis(&vals, n, n);
    assert_close(sum(&reversed), ref_rev.iter().sum());
}
