use leto::{Array, Array2, SliceArg, Storage};
use leto_ops::{
    cholesky_decompose, det, inv, norm_l1, norm_l2, norm_max, singular_values, solve,
    symmetric_eigenvalues_jacobi,
};
use ndarray::{s, Array2 as NdArray2};

const EPS: f64 = 1.0e-9;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= EPS * expected.abs().max(1.0),
        "actual {actual} expected {expected}"
    );
}

fn assert_close_slice(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_close(*actual, *expected);
    }
}

#[test]
fn dense_lu_contract_analytical() {
    // Tridiagonal Toeplitz: det = 36, solution x = [1, -2, 3], inverse known.
    let values = vec![4.0, -2.0, 1.0, -2.0, 4.0, -2.0, 1.0, -2.0, 4.0];
    let rhs_values = vec![11.0, -16.0, 17.0];
    let a = Array2::from_shape_vec([3, 3], values).unwrap();
    let rhs = Array::from_shape_vec([3], rhs_values).unwrap();

    let leto_solution = solve(&a.view(), &rhs.view()).unwrap();
    let leto_det = det(&a.view()).unwrap();
    let leto_inv = inv(&a.view()).unwrap();

    // Analytical solution: x = [1, -2, 3] (verified: A·x = b).
    let expected_x = vec![1.0, -2.0, 3.0];
    assert_close_slice(leto_solution.storage().as_slice(), &expected_x);

    // Analytical determinant: 36 (det via cofactor expansion).
    assert_close(leto_det, 36.0);

    // Analytical inverse of [[4,-2,1],[-2,4,-2],[1,-2,4]]:
    // cofactor matrix = [[12,6,0],[6,15,6],[0,6,12]], det=36
    // A^{-1} = (1/36) * cofactor^T (symmetric, so cofactor^T == cofactor).
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
    assert_close_slice(leto_inv.storage().as_slice(), &expected_inv);
}

#[test]
fn symmetric_linalg_contract_analytical() {
    // SPD matrix [[6,2,1],[2,5,2],[1,2,4]] with characteristic polynomial
    // λ³ − 15λ² + 65λ − 83 = 0.
    let values = vec![6.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
    let a = Array2::from_shape_vec([3, 3], values).unwrap();

    let mut leto_eigenvalues = symmetric_eigenvalues_jacobi(&a.view()).unwrap();
    leto_eigenvalues.sort_by(|lhs: &f64, rhs: &f64| lhs.total_cmp(rhs));

    // Characteristic polynomial: λ³ − 15λ² + 65λ − 83 = 0
    // Self-consistency: verify the three Newton identities without hard-coding individual roots.
    //   trace(A) = λ₁+λ₂+λ₃ = 15
    //   sum of principal minors = λ₁λ₂+λ₁λ₃+λ₂λ₃ = 65
    //   det(A) = λ₁λ₂λ₃ = 83
    let trace: f64 = leto_eigenvalues.iter().sum();
    assert_close(trace, 15.0);

    let pairwise_sum: f64 = leto_eigenvalues.iter().enumerate().map(|(i, &li)|
        leto_eigenvalues.iter().skip(i + 1).map(|&lj| li * lj).sum::<f64>()
    ).sum();
    assert_close(pairwise_sum, 65.0);

    let product: f64 = leto_eigenvalues.iter().product();
    assert_close(product, 83.0);

    // Cholesky L of the same SPD matrix (analytical):
    // L = [[√6, 0, 0], [2/√6, √(13/3), 0], [1/√6, 5/√39, √(83/26)]]
    let leto_cholesky = cholesky_decompose(&a.view()).unwrap();
    let ls = leto_cholesky.lower().storage().as_slice();
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
    assert_close_slice(ls, &expected_lower);
}

#[test]
fn singular_values_analytical() {
    // [[1,0,0],[2,2,0],[0,1,0],[0,0,1]] — compact, known singular values.
    // σ₁ = √(10) ≈ 3.162277, σ₂ = √(4) = 2.0, σ₃ = 0 (rank-deficient).
    let values = vec![1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let a = Array2::from_shape_vec([4, 3], values).unwrap();
    let mut leto_values = singular_values(&a.view()).unwrap();
    leto_values.sort_by(|lhs: &f64, rhs: &f64| rhs.total_cmp(lhs));

    // AᵀA = [[5,4,0],[4,5,0],[0,0,1]]; eigenvalues of AᵀA = [9,1,1]
    // σ = √eigenvalues = [3, 1, 1]
    let expected_sv = vec![3.0, 1.0, 1.0];
    assert_close_slice(&leto_values, &expected_sv);
}

#[test]
fn reverse_row_reductions_match_ndarray() {
    let values: Vec<f64> = (0..16).map(|index| index as f64 - 7.5).collect();
    let a = Array2::from_shape_vec([4, 4], values.clone()).unwrap();
    let reversed = a
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();
    let ndarray = NdArray2::from_shape_vec((4, 4), values).unwrap();
    let expected = ndarray.slice(s![.., ..;-1]);

    assert_close(leto_ops::sum(&reversed), expected.sum());
    assert_close(
        norm_l1(&reversed).unwrap(),
        expected.iter().map(|x| x.abs()).sum(),
    );
    assert_close(
        norm_l2(&reversed).unwrap(),
        expected.iter().map(|x| x * x).sum::<f64>().sqrt(),
    );
    assert_close(
        norm_max(&reversed).unwrap(),
        expected.iter().map(|x| x.abs()).fold(0.0, f64::max),
    );
}
