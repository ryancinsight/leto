use leto::{Array, Array2, SliceArg, Storage};
use leto_ops::{
    cholesky_decompose, det, inv, norm_l1, norm_l2, norm_max, singular_values, solve,
    symmetric_eigenvalues_jacobi,
};
use nalgebra::{Cholesky, DMatrix, DVector, SymmetricEigen};
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

fn dmatrix_from_row_major(rows: usize, cols: usize, values: &[f64]) -> DMatrix<f64> {
    DMatrix::from_row_slice(rows, cols, values)
}

#[test]
fn dense_lu_contract_matches_nalgebra() {
    let values = vec![4.0, -2.0, 1.0, -2.0, 4.0, -2.0, 1.0, -2.0, 4.0];
    let rhs_values = vec![11.0, -16.0, 17.0];
    let a = Array2::from_shape_vec([3, 3], values.clone()).unwrap();
    let rhs = Array::from_shape_vec([3], rhs_values.clone()).unwrap();

    let leto_solution = solve(&a.view(), &rhs.view()).unwrap();
    let leto_det = det(&a.view()).unwrap();
    let leto_inv = inv(&a.view()).unwrap();

    let nalgebra_matrix = dmatrix_from_row_major(3, 3, &values);
    let nalgebra_rhs = DVector::from_vec(rhs_values);
    let nalgebra_lu = nalgebra_matrix.clone().lu();
    let nalgebra_solution = nalgebra_lu.solve(&nalgebra_rhs).unwrap();
    let nalgebra_det = nalgebra_lu.determinant();
    let nalgebra_inv = nalgebra_matrix.try_inverse().unwrap();

    assert_close_slice(
        leto_solution.storage().as_slice(),
        nalgebra_solution.as_slice(),
    );
    assert_close(leto_det, nalgebra_det);
    let mut expected_inverse = Vec::with_capacity(9);
    for row in 0..3 {
        for col in 0..3 {
            expected_inverse.push(nalgebra_inv[(row, col)]);
        }
    }
    assert_close_slice(leto_inv.storage().as_slice(), &expected_inverse);
}

#[test]
fn symmetric_linalg_contract_matches_nalgebra() {
    let values = vec![6.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
    let a = Array2::from_shape_vec([3, 3], values.clone()).unwrap();
    let nalgebra_matrix = dmatrix_from_row_major(3, 3, &values);

    let mut leto_eigenvalues = symmetric_eigenvalues_jacobi(&a.view()).unwrap();
    let nalgebra_eigen = SymmetricEigen::new(nalgebra_matrix.clone());
    let mut nalgebra_eigenvalues = nalgebra_eigen.eigenvalues.as_slice().to_vec();
    leto_eigenvalues.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    nalgebra_eigenvalues.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    assert_close_slice(&leto_eigenvalues, &nalgebra_eigenvalues);

    let leto_cholesky = cholesky_decompose(&a.view()).unwrap();
    let nalgebra_cholesky = Cholesky::new(nalgebra_matrix).unwrap();
    let lower = nalgebra_cholesky.l();
    let mut expected_lower = Vec::with_capacity(9);
    for row in 0..3 {
        for col in 0..3 {
            expected_lower.push(lower[(row, col)]);
        }
    }
    assert_close_slice(leto_cholesky.lower().storage().as_slice(), &expected_lower);
}

#[test]
fn singular_values_match_nalgebra_svd() {
    let values = vec![1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0];
    let a = Array2::from_shape_vec([4, 2], values.clone()).unwrap();
    let mut leto_values = singular_values(&a.view()).unwrap();

    let nalgebra_matrix = dmatrix_from_row_major(4, 2, &values);
    let nalgebra_svd = nalgebra_matrix.svd(false, false);
    let mut nalgebra_values = nalgebra_svd.singular_values.as_slice().to_vec();
    leto_values.sort_by(|lhs, rhs| rhs.total_cmp(lhs));
    nalgebra_values.sort_by(|lhs, rhs| rhs.total_cmp(lhs));

    assert_close_slice(&leto_values, &nalgebra_values);
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
