//! Differential tests for the fluent rank-2 LA trait layer (ADR 0003).
//!
//! Each test asserts the trait-method surface is identical to (a) the
//! authoritative free-function kernel it delegates to and (b) oracle-independent
//! identities (Moore-Penrose conditions, normal-equations solution, analytical
//! values). A transposed-receiver case proves arbitrary-layout support flows
//! through the `AsMatrixView` bridge unchanged.

use leto::{Array, Array2, Storage};
use leto_ops::{det, matmul, norm_l2, MatrixDecompose, MatrixNorm, MatrixProduct, MatrixSolve};
use ndarray::Array2 as NdArray2;

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

#[test]
fn pinv_method_matches_moore_penrose_conditions() {
    // Tall (full column rank), wide (full row rank), and square invertible.
    let cases: [(usize, usize, Vec<f64>); 3] = [
        (4, 2, vec![1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0]),
        (2, 4, vec![1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0]),
        (3, 3, vec![4.0, -2.0, 1.0, -2.0, 4.0, -2.0, 1.0, -2.0, 4.0]),
    ];

    for (rows, cols, values) in cases {
        let a = Array2::from_shape_vec([rows, cols], values.clone()).unwrap();
        let leto_pinv = a.pinv().unwrap();
        assert_eq!(leto_pinv.shape(), [cols, rows]);

        // Moore-Penrose condition 1: A A⁺ A == A (oracle-independent).
        let mut a_pinv_a = Array2::zeros([rows, cols]);
        {
            let mut tmp = Array2::zeros([rows, rows]);
            matmul(&a.view(), &leto_pinv.view(), &mut tmp.view_mut()).unwrap();
            matmul(&tmp.view(), &a.view(), &mut a_pinv_a.view_mut()).unwrap();
        }
        assert_close_slice(a_pinv_a.storage().as_slice(), &values);

        // Moore-Penrose condition 2: (A A⁺)ᵀ == A A⁺ (symmetry).
        let mut aap = Array2::zeros([rows, rows]);
        matmul(&a.view(), &leto_pinv.view(), &mut aap.view_mut()).unwrap();
        let aap_t = aap.transpose([1, 0]).unwrap();
        // aap_t is strided (non-contiguous); to_contiguous materializes the copy.
        let aap_t_contig = aap_t.to_contiguous();
        assert_close_slice(aap.as_slice().unwrap(), aap_t_contig.as_slice().unwrap());
    }
}

#[test]
fn matmul_method_matches_kernel_and_ndarray() {
    let (m, k, n) = (4usize, 3, 5);
    let a_vals: Vec<f64> = (0..m * k).map(|i| i as f64 * 0.3 + 1.0).collect();
    let b_vals: Vec<f64> = (0..k * n).map(|i| i as f64 * 0.2 - 1.0).collect();
    let a = Array2::from_shape_vec([m, k], a_vals.clone()).unwrap();
    let b = Array2::from_shape_vec([k, n], b_vals.clone()).unwrap();

    // Fluent method.
    let fluent = a.matmul(&b).unwrap();

    // Authoritative caller-owned kernel.
    let mut kernel_out = Array2::from_shape_vec([m, n], vec![0.0; m * n]).unwrap();
    matmul(&a.view(), &b.view(), &mut kernel_out.view_mut()).unwrap();

    // ndarray oracle.
    let nd = NdArray2::from_shape_vec((m, k), a_vals)
        .unwrap()
        .dot(&NdArray2::from_shape_vec((k, n), b_vals).unwrap());

    assert_close_slice(fluent.storage().as_slice(), kernel_out.storage().as_slice());
    assert_close_slice(fluent.storage().as_slice(), nd.as_slice().unwrap());
}

#[test]
fn solve_det_inv_methods_match_analytical() {
    // Tridiagonal Toeplitz: det=36, x=[1,-2,3], inverse known.
    let values = vec![4.0, -2.0, 1.0, -2.0, 4.0, -2.0, 1.0, -2.0, 4.0];
    let rhs_values = vec![11.0, -16.0, 17.0];
    let a = Array2::from_shape_vec([3, 3], values).unwrap();
    let rhs = Array::from_shape_vec([3], rhs_values).unwrap();

    let x = a.solve(&rhs.view()).unwrap();
    let d = a.det().unwrap();
    let inverse = a.inv().unwrap();

    assert_close_slice(x.storage().as_slice(), &[1.0, -2.0, 3.0]);
    assert_close(d, 36.0);
    let expected_inv = vec![
        12.0 / 36.0,
        6.0 / 36.0,
        0.0,
        6.0 / 36.0,
        15.0 / 36.0,
        6.0 / 36.0,
        0.0,
        6.0 / 36.0,
        12.0 / 36.0,
    ];
    assert_close_slice(inverse.storage().as_slice(), &expected_inv);
}

#[test]
fn cholesky_and_eigen_methods_match_analytical() {
    let values = vec![6.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
    let a = Array2::from_shape_vec([3, 3], values).unwrap();

    // Analytical eigenvalues via Newton identities (self-consistency, no hard-coded roots).
    //   λ₁+λ₂+λ₃ = trace(A) = 15
    //   λ₁λ₂+λ₁λ₃+λ₂λ₃ = sum of principal minors = 65
    //   λ₁λ₂λ₃ = det(A) = 83
    let mut eig = a.symmetric_eigenvalues().unwrap();
    eig.sort_by(|x: &f64, y: &f64| x.total_cmp(y));
    let trace: f64 = eig.iter().sum();
    assert_close(trace, 15.0);
    let pairwise_sum: f64 = eig
        .iter()
        .enumerate()
        .map(|(i, &li)| eig.iter().skip(i + 1).map(|&lj| li * lj).sum::<f64>())
        .sum();
    assert_close(pairwise_sum, 65.0);
    let product: f64 = eig.iter().product();
    assert_close(product, 83.0);

    // Analytical Cholesky L (lower-triangular):
    // L = [[√6, 0, 0], [2/√6, √(13/3), 0], [1/√6, 5/√39, √(83/26)]]
    let chol = a.cholesky().unwrap();
    let sqrt6 = 6.0_f64.sqrt();
    let expected_lower = vec![
        sqrt6,
        0.0,
        0.0,
        2.0 / sqrt6,
        (13.0_f64 / 3.0).sqrt(),
        0.0,
        1.0 / sqrt6,
        5.0 / 39.0_f64.sqrt(),
        (83.0_f64 / 26.0).sqrt(),
    ];
    assert_close_slice(chol.lower().storage().as_slice(), &expected_lower);
}

#[test]
fn singular_values_and_least_squares_methods() {
    // Singular values of [[1,0],[0,2],[2,0],[0,1]].
    // AᵀA = [[5,0],[0,5]]; eigenvalues of AᵀA = [5,5].
    // σ = √eigvals = [√5, √5].
    let sv_vals = vec![1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0];
    let a = Array2::from_shape_vec([4, 2], sv_vals).unwrap();
    let mut sv = a.singular_values().unwrap();
    sv.sort_by(|x: &f64, y: &f64| y.total_cmp(x));
    let sqrt5 = 5.0_f64.sqrt();
    assert_close_slice(&sv, &[sqrt5, sqrt5]);

    // Overdetermined least squares vs hand-computed normal-equations solution.
    // A = [[1,1],[1,2],[1,3],[1,4]], b = [6,5,7,10]
    // AᵀA = [[4,10],[10,30]], Aᵀb = [28,77]
    // det(AᵀA) = 4*30 - 10*10 = 20
    // (AᵀA)⁻¹ = (1/20)*[[30,-10],[-10,4]] = [[1.5,-0.5],[-0.5,0.2]]
    // x = (AᵀA)⁻¹ Aᵀb = [[1.5,-0.5],[-0.5,0.2]] * [28,77]
    //   = [1.5*28 - 0.5*77, -0.5*28 + 0.2*77] = [42-38.5, -14+15.4] = [3.5, 1.4]
    let ls_vals = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
    let rhs_vals = vec![6.0, 5.0, 7.0, 10.0];
    let ls = Array2::from_shape_vec([4, 2], ls_vals).unwrap();
    let rhs = Array::from_shape_vec([4], rhs_vals).unwrap();
    let x = ls.solve_least_squares(&rhs.view()).unwrap();

    let expected_x = vec![3.5, 1.4];
    assert_close_slice(x.storage().as_slice(), &expected_x);
}

#[test]
fn norm_methods_match_kernel_and_analytical() {
    let values = vec![3.0, -4.0, 12.0, 0.0, -5.0, 0.0];
    let a = Array2::from_shape_vec([2, 3], values.clone()).unwrap();

    assert_close(a.norm_l2().unwrap(), norm_l2(&a.view()).unwrap());
    // Frobenius norm: √(9+16+144+0+25+0) = √194 ≈ 13.928.
    assert_close(a.norm_l2().unwrap(), 194.0_f64.sqrt());
    // L1 / max entrywise.
    assert_close(a.norm_l1().unwrap(), values.iter().map(|v| v.abs()).sum());
    assert_close(
        a.norm_max().unwrap(),
        values.iter().map(|v| v.abs()).fold(0.0, f64::max),
    );
}

#[test]
fn strided_transposed_receiver_matches_contiguous() {
    // A transposed view is strided (non-C-contiguous); the trait must produce
    // the same result as the same op on its contiguous materialization.
    let values = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0];
    let a = Array2::from_shape_vec([3, 3], values).unwrap();
    let transposed = a.transpose([1, 0]).unwrap();
    let materialized = transposed.to_contiguous();

    // det through the strided view == det of the dense copy (free fn).
    assert_close(
        transposed.det().unwrap(),
        det(&materialized.view()).unwrap(),
    );
    // norm through the strided view == norm of the dense copy.
    assert_close(
        transposed.norm_l2().unwrap(),
        materialized.norm_l2().unwrap(),
    );
}
