use leto::{Array2, SliceArg, Storage};
use leto_ops::{singular_values, svd_decompose};

fn assert_close(lhs: f64, rhs: f64, epsilon: f64) {
    assert!(
        (lhs - rhs).abs() <= epsilon,
        "left {lhs} differs from right {rhs}"
    );
}

fn reconstruct(decomposition: &leto_ops::SvdDecomposition<f64>) -> Vec<f64> {
    let [rows, cols] = decomposition.left_singular_vectors.shape();
    let mut output = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let mut value = 0.0;
            for k in 0..cols {
                let u = *decomposition.left_singular_vectors.get([row, k]).unwrap();
                let sigma = decomposition.singular_values[k];
                let v = *decomposition.right_singular_vectors.get([col, k]).unwrap();
                value += u * sigma * v;
            }
            output[row * cols + col] = value;
        }
    }
    output
}

fn column_norm(values: &[f64], rows: usize, cols: usize, col: usize) -> f64 {
    (0..rows)
        .map(|row| values[row * cols + col] * values[row * cols + col])
        .sum::<f64>()
        .sqrt()
}

fn column_dot(values: &[f64], rows: usize, cols: usize, lhs: usize, rhs: usize) -> f64 {
    (0..rows)
        .map(|row| values[row * cols + lhs] * values[row * cols + rhs])
        .sum::<f64>()
}

#[test]
fn svd_decompose_reconstructs_tall_full_rank_matrix() {
    let values = vec![1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0];
    let matrix = Array2::from_shape_vec([4, 2], values.clone()).unwrap();
    let decomposition = svd_decompose(&matrix.view()).unwrap();

    assert_eq!(decomposition.singular_values.len(), 2);
    assert!(decomposition.singular_values[0] >= decomposition.singular_values[1]);

    let reconstructed = reconstruct(&decomposition);
    for (actual, expected) in reconstructed.iter().zip(values.iter()) {
        assert_close(*actual, *expected, 1.0e-9);
    }

    let left = decomposition.left_singular_vectors.storage().as_slice();
    assert_close(column_norm(left, 4, 2, 0), 1.0, 1.0e-9);
    assert_close(column_norm(left, 4, 2, 1), 1.0, 1.0e-9);
    assert_close(column_dot(left, 4, 2, 0, 1), 0.0, 1.0e-9);
}

#[test]
fn singular_values_match_diagonal_closed_form() {
    let matrix =
        Array2::from_shape_vec([3, 3], vec![3.0f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
    let values = singular_values(&matrix.view()).unwrap();
    assert_close(values[0], 3.0, 1.0e-12);
    assert_close(values[1], 2.0, 1.0e-12);
    assert_close(values[2], 1.0, 1.0e-12);
}

#[test]
fn svd_accepts_strided_full_rank_view() {
    let backing = Array2::from_shape_vec(
        [4, 4],
        vec![
            3.0, 99.0, 0.0, 99.0, 99.0, 99.0, 99.0, 99.0, 0.0, 99.0, 2.0, 99.0, 99.0, 99.0, 99.0,
            99.0,
        ],
    )
    .unwrap();
    let view = backing
        .view()
        .slice_with::<2>(&[
            SliceArg::range(Some(0), None, 2),
            SliceArg::range(Some(0), None, 2),
        ])
        .unwrap();

    let values = singular_values(&view).unwrap();
    assert_close(values[0], 3.0, 1.0e-12);
    assert_close(values[1], 2.0, 1.0e-12);
}

#[test]
fn svd_is_generic_over_f32() {
    let matrix = Array2::from_shape_vec([2, 2], vec![2.0f32, 0.0, 0.0, 1.0]).unwrap();
    let values = singular_values(&matrix.view()).unwrap();
    assert!((values[0] - 2.0).abs() <= 1.0e-5);
    assert!((values[1] - 1.0).abs() <= 1.0e-5);
}

#[test]
fn svd_rejects_unsupported_or_invalid_inputs() {
    let wide = Array2::from_shape_vec([2, 3], vec![1.0f64; 6]).unwrap();
    assert!(svd_decompose(&wide.view()).is_err());

    let rank_deficient =
        Array2::from_shape_vec([3, 2], vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();
    assert!(svd_decompose(&rank_deficient.view()).is_err());

    let non_finite = Array2::from_shape_vec([2, 2], vec![1.0, f64::NAN, 0.0, 1.0]).unwrap();
    assert!(svd_decompose(&non_finite.view()).is_err());
}
