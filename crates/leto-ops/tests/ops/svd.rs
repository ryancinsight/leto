use leto::{Array2, SliceArg, Storage};
use leto_ops::{pinv, singular_values, svd_decompose, svd_rank_revealing, MatrixProduct};
use nalgebra::DMatrix;

fn assert_close(lhs: f64, rhs: f64, epsilon: f64) {
    assert!(
        (lhs - rhs).abs() <= epsilon,
        "left {lhs} differs from right {rhs}"
    );
}

fn reconstruct(decomposition: &leto_ops::SvdDecomposition<f64>, output_cols: usize) -> Vec<f64> {
    let [rows, rank] = decomposition.left_singular_vectors.shape();
    let mut output = vec![0.0; rows * output_cols];
    for row in 0..rows {
        for col in 0..output_cols {
            let mut value = 0.0;
            for k in 0..rank {
                let u = *decomposition.left_singular_vectors.get([row, k]).unwrap();
                let sigma = decomposition.singular_values[k];
                let v = *decomposition.right_singular_vectors.get([col, k]).unwrap();
                value += u * sigma * v;
            }
            output[row * output_cols + col] = value;
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

    let reconstructed = reconstruct(&decomposition, 2);
    for (actual, expected) in reconstructed.iter().zip(values.iter()) {
        assert_close(*actual, *expected, 1.0e-9);
    }

    let left = decomposition.left_singular_vectors.storage().as_slice();
    assert_close(column_norm(left, 4, 2, 0), 1.0, 1.0e-9);
    assert_close(column_norm(left, 4, 2, 1), 1.0, 1.0e-9);
    assert_close(column_dot(left, 4, 2, 0, 1), 0.0, 1.0e-9);
}

#[test]
fn svd_decompose_reconstructs_wide_full_rank_matrix() {
    let values = vec![3.0, 0.0, 0.0, 0.0, 0.0, 2.0];
    let matrix = Array2::from_shape_vec([2, 3], values.clone()).unwrap();
    let decomposition = svd_decompose(&matrix.view()).unwrap();

    assert_eq!(decomposition.singular_values.len(), 2);
    assert_eq!(decomposition.left_singular_vectors.shape(), [2, 2]);
    assert_eq!(decomposition.right_singular_vectors.shape(), [3, 2]);
    assert_close(decomposition.singular_values[0], 3.0, 1.0e-12);
    assert_close(decomposition.singular_values[1], 2.0, 1.0e-12);

    let reconstructed = reconstruct(&decomposition, 3);
    for (actual, expected) in reconstructed.iter().zip(values.iter()) {
        assert_close(*actual, *expected, 1.0e-9);
    }

    let right = decomposition.right_singular_vectors.storage().as_slice();
    assert_close(column_norm(right, 3, 2, 0), 1.0, 1.0e-9);
    assert_close(column_norm(right, 3, 2, 1), 1.0, 1.0e-9);
    assert_close(column_dot(right, 3, 2, 0, 1), 0.0, 1.0e-9);
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
fn singular_values_accept_rank_deficient_inputs_without_vectors() {
    let tall_rank_deficient =
        Array2::from_shape_vec([3, 2], vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();
    let tall_values = singular_values(&tall_rank_deficient.view()).unwrap();
    assert_eq!(tall_values.len(), 2);
    assert_close(tall_values[0], 8.366_600_265_340_756, 1.0e-12);
    assert_close(tall_values[1], 0.0, 1.0e-12);

    let wide_rank_deficient = Array2::from_shape_vec([2, 3], vec![1.0f64; 6]).unwrap();
    let wide_values = singular_values(&wide_rank_deficient.view()).unwrap();
    assert_eq!(wide_values.len(), 2);
    assert_close(wide_values[0], 2.449_489_742_783_178, 1.0e-12);
    assert_close(wide_values[1], 0.0, 1.0e-12);
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
    let wide_rank_deficient = Array2::from_shape_vec([2, 3], vec![1.0f64; 6]).unwrap();
    assert!(svd_decompose(&wide_rank_deficient.view()).is_err());

    let rank_deficient =
        Array2::from_shape_vec([3, 2], vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();
    assert!(svd_decompose(&rank_deficient.view()).is_err());
    assert!(singular_values(&rank_deficient.view()).is_ok());

    let non_finite = Array2::from_shape_vec([2, 2], vec![1.0, f64::NAN, 0.0, 1.0]).unwrap();
    assert!(svd_decompose(&non_finite.view()).is_err());
    assert!(singular_values(&non_finite.view()).is_err());
}

// ── Rank-revealing one-sided Jacobi SVD (ADR 0005) ──────────────────────────

#[test]
fn svd_rank_revealing_reconstructs_rank_deficient_matrix() {
    // Row 2 = 2 * row 1 → rank 1, one zero singular value.
    let values = vec![1.0, 2.0, 2.0, 4.0];
    let matrix = Array2::from_shape_vec([2, 2], values.clone()).unwrap();
    let svd = svd_rank_revealing(&matrix.view()).unwrap();

    assert_eq!(svd.singular_values.len(), 2);
    assert!(svd.singular_values[0] >= svd.singular_values[1]);
    assert_close(svd.singular_values[1], 0.0, 1.0e-9); // rank-deficient

    // Reconstruction A = U Σ Vᵀ holds despite the zero singular value.
    let reconstructed = reconstruct(&svd, 2);
    for (actual, expected) in reconstructed.iter().zip(values.iter()) {
        assert_close(*actual, *expected, 1.0e-9);
    }

    // V is fully orthonormal (the defining property the Gram path cannot give).
    let v = svd.right_singular_vectors.storage().as_slice();
    assert_close(column_norm(v, 2, 2, 0), 1.0, 1.0e-9);
    assert_close(column_norm(v, 2, 2, 1), 1.0, 1.0e-9);
    assert_close(column_dot(v, 2, 2, 0, 1), 0.0, 1.0e-9);
}

#[test]
fn svd_rank_revealing_matches_nalgebra_singular_values() {
    // Full-rank and rank-deficient, tall and wide.
    let cases: [(usize, usize, Vec<f64>); 3] = [
        (4, 2, vec![1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0]),
        (2, 2, vec![1.0, 2.0, 2.0, 4.0]),
        (2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ];
    for (rows, cols, values) in cases {
        let a = Array2::from_shape_vec([rows, cols], values.clone()).unwrap();
        let mut leto_sv = svd_rank_revealing(&a.view()).unwrap().singular_values;
        let mut na_sv = DMatrix::from_row_slice(rows, cols, &values)
            .singular_values()
            .as_slice()
            .to_vec();
        leto_sv.sort_by(|x, y| y.total_cmp(x));
        na_sv.sort_by(|x, y| y.total_cmp(x));
        assert_eq!(leto_sv.len(), na_sv.len());
        for (l, n) in leto_sv.iter().zip(na_sv.iter()) {
            assert_close(*l, *n, 1.0e-9);
        }
    }
}

#[test]
fn pinv_rank_deficient_matches_nalgebra_and_moore_penrose() {
    let values = vec![1.0, 2.0, 2.0, 4.0]; // rank 1
    let a = Array2::from_shape_vec([2, 2], values.clone()).unwrap();
    let a_pinv = pinv(&a.view()).unwrap();

    // Differential vs nalgebra's SVD-based pseudo_inverse.
    let na_pinv = DMatrix::from_row_slice(2, 2, &values)
        .pseudo_inverse(1.0e-12)
        .unwrap();
    let leto_slice = a_pinv.storage().as_slice();
    for r in 0..2 {
        for c in 0..2 {
            assert_close(leto_slice[r * 2 + c], na_pinv[(r, c)], 1.0e-9);
        }
    }

    // Moore-Penrose conditions: A A⁺ A = A and A⁺ A A⁺ = A⁺.
    let a_ap_a = a.matmul(&a_pinv).unwrap().matmul(&a).unwrap();
    for (actual, expected) in a_ap_a.storage().as_slice().iter().zip(values.iter()) {
        assert_close(*actual, *expected, 1.0e-9);
    }
    let ap_a_ap = a_pinv.matmul(&a).unwrap().matmul(&a_pinv).unwrap();
    for (actual, expected) in ap_a_ap
        .storage()
        .as_slice()
        .iter()
        .zip(a_pinv.storage().as_slice().iter())
    {
        assert_close(*actual, *expected, 1.0e-9);
    }
}
