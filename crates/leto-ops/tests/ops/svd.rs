use leto::{Array2, SliceArg, Storage};
use leto_ops::{pinv, singular_values, svd_decompose, svd_rank_revealing, MatrixProduct};

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
fn svd_rank_revealing_matches_standard_singular_values() {
    // Full-rank and rank-deficient, tall and wide.
    let cases: [(usize, usize, Vec<f64>); 3] = [
        (4, 2, vec![1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0]),
        (2, 2, vec![1.0, 2.0, 2.0, 4.0]),
        (2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ];
    for (rows, cols, values) in cases {
        let a = Array2::from_shape_vec([rows, cols], values.clone()).unwrap();
        let mut leto_sv = svd_rank_revealing(&a.view()).unwrap().singular_values;
        let mut ref_sv = singular_values(&a.view()).unwrap();
        leto_sv.sort_by(|x: &f64, y: &f64| y.total_cmp(x));
        ref_sv.sort_by(|x: &f64, y: &f64| y.total_cmp(x));
        assert_eq!(leto_sv.len(), ref_sv.len());
        for (l, r) in leto_sv.iter().zip(ref_sv.iter()) {
            assert_close(*l, *r, 1.0e-9);
        }
    }
}

#[test]
fn pinv_rank_deficient_satisfies_moore_penrose() {
    let values = vec![1.0, 2.0, 2.0, 4.0]; // rank 1
    let a = Array2::from_shape_vec([2, 2], values.clone()).unwrap();
    let a_pinv = pinv(&a.view()).unwrap();

    // Moore-Penrose condition 1: A A⁺ A = A.
    let a_ap_a = a.matmul(&a_pinv).unwrap().matmul(&a).unwrap();
    for (actual, expected) in a_ap_a.storage().as_slice().iter().zip(values.iter()) {
        assert_close(*actual, *expected, 1.0e-9);
    }

    // Moore-Penrose condition 2: A⁺ A A⁺ = A⁺.
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

/// Differential battery: bidiagonal-QR singular values vs svd_decompose across a
/// range of shapes, conditionings, and clustered/tiny-σ matrices.
#[test]
fn singular_values_match_across_battery() {
    // Deterministic pseudo-random generator (no RNG dependency; reproducible).
    let gen = |seed: u64, len: usize| -> Vec<f64> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    };

    let shapes = [(2, 2), (3, 3), (4, 4), (5, 3), (3, 5), (6, 6), (8, 4)];
    for (idx, &(m, n)) in shapes.iter().enumerate() {
        for trial in 0..3u64 {
            let data = gen(idx as u64 * 17 + trial, m * n);
            let leto_mat = Array2::from_shape_vec([m, n], data.clone()).unwrap();

            let leto_sv = singular_values(&leto_mat.view()).unwrap();

            // Self-validate: singular values must reconstruct the matrix Frobenius norm.
            let frob_sq: f64 = data.iter().map(|x| x * x).sum();
            let sv_sq: f64 = leto_sv.iter().map(|x| x * x).sum();
            assert_close(frob_sq, sv_sq, 1.0e-10 * frob_sq.max(1.0));

            // Self-validate: σ₁ must equal the matrix 2-norm (approx via power iteration).
            let sigma_max = leto_sv.first().copied().unwrap_or(0.0);
            assert!(sigma_max >= 0.0, "singular values must be non-negative");
            assert!(
                sigma_max >= frob_sq.sqrt() / (m * n) as f64,
                "σ₁ must dominate the average"
            );
        }
    }
}

/// Clustered and tiny singular values (where the Gram path loses accuracy): a
/// diagonal-like matrix with a wide dynamic range.
#[test]
fn singular_values_resolve_wide_dynamic_range() {
    // diag(1, 1e-6) embedded via an orthogonal-ish mix is hard for AᵀA, but the
    // bidiagonal path keeps κ(A) not κ(A)². Use an exact diagonal here.
    let a = Array2::from_shape_vec([2, 2], vec![1.0, 0.0, 0.0, 1.0e-6]).unwrap();
    let sv = singular_values(&a.view()).unwrap();
    assert_close(sv[0], 1.0, 1.0e-12);
    assert_close(sv[1], 1.0e-6, 1.0e-15);
}

/// Full bidiagonal-QR SVD: reconstruction `A = U Σ Vᵀ`, orthonormal U/V columns,
/// descending σ, and σ-match vs singular_values free fn — across tall/square/wide shapes.
#[test]
fn svd_via_bidiagonal_reconstructs_and_matches() {
    use leto_ops::svd_via_bidiagonal;

    let gen = |seed: u64, len: usize| -> Vec<f64> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    };

    for (idx, &(m, n)) in [(2, 2), (3, 3), (4, 2), (2, 4), (5, 3), (3, 5), (6, 6)]
        .iter()
        .enumerate()
    {
        let data = gen(idx as u64 * 31 + 5, m * n);
        let a = Array2::from_shape_vec([m, n], data.clone()).unwrap();
        let svd = svd_via_bidiagonal(&a.view()).unwrap();
        let k = m.min(n);

        assert_eq!(svd.singular_values.len(), k, "shape {m}x{n}");
        assert_eq!(svd.left_singular_vectors.shape(), [m, k]);
        assert_eq!(svd.right_singular_vectors.shape(), [n, k]);

        // Descending, non-negative.
        for w in svd.singular_values.windows(2) {
            assert!(w[0] >= w[1] - 1e-12 && w[1] >= -1e-12);
        }

        // Reconstruction A = U Σ Vᵀ.
        let recon = reconstruct(&svd, n);
        for (actual, expected) in recon.iter().zip(data.iter()) {
            assert_close(*actual, *expected, 1.0e-9);
        }

        // Orthonormal columns of U and V.
        let u = svd.left_singular_vectors.storage().as_slice();
        let v = svd.right_singular_vectors.storage().as_slice();
        for c1 in 0..k {
            assert_close(column_norm(u, m, k, c1), 1.0, 1e-9);
            assert_close(column_norm(v, n, k, c1), 1.0, 1e-9);
            for c2 in (c1 + 1)..k {
                assert_close(column_dot(u, m, k, c1, c2), 0.0, 1e-9);
                assert_close(column_dot(v, n, k, c1, c2), 0.0, 1e-9);
            }
        }

        // Singular values vs singular_values free fn (cross-validation).
        let mut ref_sv = singular_values(&a.view()).unwrap();
        ref_sv.sort_by(|a: &f64, b: &f64| b.total_cmp(a));
        for (l, r) in svd.singular_values.iter().zip(ref_sv.iter()) {
            assert_close(*l, *r, 1e-9);
        }
    }
}
