use leto::{Array2, SliceArg, Storage};
use leto_ops::symmetric_eigen_jacobi;

fn assert_close(lhs: f64, rhs: f64, epsilon: f64) {
    assert!(
        (lhs - rhs).abs() <= epsilon,
        "left {lhs} differs from right {rhs}"
    );
}

fn column_norm(values: &[f64], n: usize, col: usize) -> f64 {
    (0..n)
        .map(|row| values[row * n + col] * values[row * n + col])
        .sum::<f64>()
        .sqrt()
}

fn column_dot(values: &[f64], n: usize, lhs: usize, rhs: usize) -> f64 {
    (0..n)
        .map(|row| values[row * n + lhs] * values[row * n + rhs])
        .sum()
}

#[test]
fn symmetric_eigen_jacobi_solves_known_two_by_two_matrix() {
    let matrix = Array2::from_shape_vec([2, 2], vec![2.0, 1.0, 1.0, 2.0]).unwrap();
    let decomposition = symmetric_eigen_jacobi(&matrix.view()).unwrap();

    assert_close(decomposition.eigenvalues[0], 1.0, 1.0e-12);
    assert_close(decomposition.eigenvalues[1], 3.0, 1.0e-12);
    let eigenvectors = decomposition.eigenvectors.storage().as_slice();
    assert_close(column_norm(eigenvectors, 2, 0), 1.0, 1.0e-12);
    assert_close(column_norm(eigenvectors, 2, 1), 1.0, 1.0e-12);
    assert_close(column_dot(eigenvectors, 2, 0, 1), 0.0, 1.0e-12);
}

#[test]
fn symmetric_eigen_jacobi_accepts_strided_symmetric_view() {
    let matrix = Array2::from_shape_vec(
        [4, 4],
        vec![
            4.0, 0.0, 1.0, 0.0, 0.0, 9.0, 0.0, 8.0, 1.0, 0.0, 4.0, 0.0, 0.0, 8.0, 0.0, 9.0,
        ],
    )
    .unwrap();
    let view = matrix
        .view()
        .slice_with::<2>(&[
            SliceArg::range(Some(0), None, 2),
            SliceArg::range(Some(0), None, 2),
        ])
        .unwrap();

    let decomposition = symmetric_eigen_jacobi(&view).unwrap();
    assert_close(decomposition.eigenvalues[0], 3.0, 1.0e-12);
    assert_close(decomposition.eigenvalues[1], 5.0, 1.0e-12);
}

#[test]
fn symmetric_eigen_jacobi_matches_nalgebra_for_path_graph_laplacian() {
    let matrix = Array2::from_shape_vec(
        [3, 3],
        vec![1.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 1.0],
    )
    .unwrap();
    let decomposition = symmetric_eigen_jacobi(&matrix.view()).unwrap();

    let reference = nalgebra::SymmetricEigen::new(nalgebra::DMatrix::from_row_slice(
        3,
        3,
        matrix.storage().as_slice(),
    ));
    let mut expected = reference.eigenvalues.as_slice().to_vec();
    expected.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());

    for (actual, expected) in decomposition.eigenvalues.iter().zip(expected.iter()) {
        assert_close(*actual, *expected, 1.0e-10);
    }
}

#[test]
fn symmetric_eigen_jacobi_rejects_invalid_inputs() {
    let rectangular = Array2::from_shape_vec([2, 3], vec![1.0; 6]).unwrap();
    assert!(symmetric_eigen_jacobi(&rectangular.view()).is_err());

    let asymmetric = Array2::from_shape_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(symmetric_eigen_jacobi(&asymmetric.view()).is_err());

    let non_finite = Array2::from_shape_vec([2, 2], vec![1.0, f64::NAN, f64::NAN, 1.0]).unwrap();
    assert!(symmetric_eigen_jacobi(&non_finite.view()).is_err());
}
