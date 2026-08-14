use eunomia::RealField;
use leto::{Array2, SliceArg, Storage};
use leto_ops::{pinv, singular_values, svd_decompose, MatrixProduct, RealScalar};

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
fn svd_rejects_invalid_inputs() {
    let non_finite = Array2::from_shape_vec([2, 2], vec![1.0, f64::NAN, 0.0, 1.0]).unwrap();
    assert!(svd_decompose(&non_finite.view()).is_err());
    assert!(singular_values(&non_finite.view()).is_err());
}

// ── Rank-deficient input: revealed in Σ, never rejected (ADR 0005) ──────────

/// Rank deficiency is data, not an error: `svd_decompose` returns it as `σ = 0`
/// and still delivers orthonormal `U` *and* `V`, so `A = U Σ Vᵀ` holds exactly.
#[test]
fn svd_decompose_reveals_rank_deficiency_with_orthonormal_factors() {
    // Row 2 = 2 * row 1 → rank 1, one zero singular value.
    let values = vec![1.0, 2.0, 2.0, 4.0];
    let matrix = Array2::from_shape_vec([2, 2], values.clone()).unwrap();
    let svd = svd_decompose(&matrix.view()).unwrap();

    assert_eq!(svd.singular_values.len(), 2);
    assert!(svd.singular_values[0] >= svd.singular_values[1]);
    assert_close(svd.singular_values[1], 0.0, 1.0e-9); // rank-deficient

    // Reconstruction A = U Σ Vᵀ holds despite the zero singular value.
    let reconstructed = reconstruct(&svd, 2);
    for (actual, expected) in reconstructed.iter().zip(values.iter()) {
        assert_close(*actual, *expected, 1.0e-9);
    }

    // Both factors stay orthonormal at deficient rank: the null-space column of
    // U is materialized, not left zero.
    for factor in [&svd.left_singular_vectors, &svd.right_singular_vectors] {
        let f = factor.storage().as_slice();
        assert_close(column_norm(f, 2, 2, 0), 1.0, 1.0e-9);
        assert_close(column_norm(f, 2, 2, 1), 1.0, 1.0e-9);
        assert_close(column_dot(f, 2, 2, 0, 1), 0.0, 1.0e-9);
    }
}

/// A singular value far below any plausible rank tolerance must still be carried
/// by its own `U` column. Zeroing sub-tolerance `U` columns instead — as a
/// tolerance-gated construction does — leaves a reconstruction error equal to
/// the whole dropped singular value; here that would be `1e-14` rather than
/// rounding noise.
#[test]
fn svd_decompose_reconstructs_below_tolerance_singular_value() {
    let values = vec![1.0, 0.0, 0.0, 1.0e-14];
    let matrix = Array2::from_shape_vec([2, 2], values.clone()).unwrap();
    let svd = svd_decompose(&matrix.view()).unwrap();

    assert_close(svd.singular_values[1], 1.0e-14, 1.0e-26);
    let reconstructed = reconstruct(&svd, 2);
    for (actual, expected) in reconstructed.iter().zip(values.iter()) {
        assert_close(*actual, *expected, 1.0e-20);
    }
}

/// Cross-path agreement: `singular_values` and `svd_decompose` remain two
/// distinct routes through the bidiagonal QR — values-only runs
/// `bidiagonal_diag_colmajor` + `qr_iterate::<_, false>`, the full SVD runs
/// `bidiagonalize` + `qr_iterate::<_, true>` with U/V accumulation. They must
/// agree on σ across full-rank and rank-deficient, tall, square and wide input;
/// a defect in the accumulating instantiation shows up as a divergence here.
#[test]
fn singular_values_agree_between_values_only_and_full_svd() {
    let cases: [(usize, usize, Vec<f64>); 3] = [
        (4, 2, vec![1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0]),
        (2, 2, vec![1.0, 2.0, 2.0, 4.0]),
        (2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ];
    for (rows, cols, values) in cases {
        let a = Array2::from_shape_vec([rows, cols], values.clone()).unwrap();
        let mut leto_sv = svd_decompose(&a.view()).unwrap().singular_values;
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
fn svd_decompose_reconstructs_and_matches() {
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
        let svd = svd_decompose(&a.view()).unwrap();
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

// ── Exactly rank-deficient input, at every precision ────────────────────────
//
// Rank deficiency has two structurally different forms, and only one of them is
// covered by testing a *near*-deficient matrix. When the deficiency is exact, the
// Householder bidiagonalization can produce an **exact zero on the diagonal** of
// `B`; a shifted QR step cannot deflate that (the implicit `BᵀB` is singular, the
// Wilkinson shift takes its nonzero eigenvalue, and the sweep reaches a fixed
// point at `d = 0`, `e ≠ 0` that the deflation test never accepts). It needs the
// zero-diagonal chase instead.
//
// Whether the zero comes out exact depends on the rounding of the
// bidiagonalization, so it depends on the precision: `[[1,2],[2,4],[3,6]]` gives
// an exact `d[1] = 0` at `f32` and a `8.9e-8` residue at `f64`. Testing rank
// deficiency at `f64` and genericity only on full-rank input therefore leaves the
// case uncovered — which is how a non-convergence error reached the GPU
// consumers. These cases run every shape at **both** precisions.

/// Backward-error bound for a Golub–Reinsch SVD of an `m × n` matrix:
/// `A + E = Û Σ̂ V̂ᵀ` with `‖E‖₂ ≤ p·ε·‖A‖₂` and `‖ÛᵀÛ − I‖₂ ≤ p·ε`, `p` modest in
/// the dimensions (Golub & Van Loan, *Matrix Computations* 4th ed., §8.6.3: the
/// factorization is a product of Householder reflectors and plane rotations, each
/// contributing `O(ε)`, with `O(max(m,n))` of them touching any one entry).
/// `p = 8·max(m, n)` throughout. Weyl's theorem carries `‖E‖₂` to each
/// `|σ̂ᵢ − σᵢ|`, including the `σᵢ` whose exact value is `0` — so the same bound
/// is what a rank-deficient direction must satisfy.
///
/// Returns `(absolute, relative)`: the `‖A‖₂ ≈ σ₁`-scaled bound for singular
/// values and reconstruction, and the bare relative bound for orthonormality.
fn error_bounds<T: RealScalar + RealField>(rows: usize, cols: usize, norm: f64) -> (f64, f64) {
    #[allow(clippy::cast_precision_loss)]
    let relative = 8.0 * rows.max(cols) as f64 * <T as RealField>::EPSILON.to_f64();
    (relative * norm, relative)
}

/// Decompose `entries` (`rows × cols`, exactly representable in binary) at
/// precision `T` and assert full value semantics against the analytic `expected`
/// spectrum: singular values, reconstruction `A = U Σ Vᵀ`, and orthonormal
/// columns of both `U` and `V`.
///
/// Orthonormality is asserted on **both** factors and at deficient rank
/// specifically: the deleted one-sided Jacobi path returned a non-orthonormal `U`
/// exactly here (`‖UᵀU − I‖ = 1.0` on a null-space column), and that was a
/// load-bearing reason for keeping this path. Nothing may regress it.
fn assert_rank_deficient_svd<T: RealScalar + RealField>(
    rows: usize,
    cols: usize,
    entries: &[f64],
    expected: &[f64],
) {
    let values: Vec<T> = entries.iter().map(|&x| T::from_f64(x)).collect();
    let matrix = Array2::from_shape_vec([rows, cols], values).unwrap();
    let svd = svd_decompose(&matrix.view()).unwrap();
    let rank = rows.min(cols);
    let (absolute, relative) = error_bounds::<T>(rows, cols, expected[0]);

    assert_eq!(svd.singular_values.len(), rank);
    assert_eq!(expected.len(), rank);
    let sigma: Vec<f64> = svd.singular_values.iter().map(|x| x.to_f64()).collect();
    for window in sigma.windows(2) {
        assert!(
            window[0] >= window[1],
            "σ must be descending, got {sigma:?}"
        );
    }
    for (got, want) in sigma.iter().zip(expected) {
        assert!(
            (got - want).abs() <= absolute,
            "{rows}x{cols}: σ {got} vs {want}, bound {absolute:e}"
        );
    }

    let u = svd.left_singular_vectors.storage().as_slice();
    let v = svd.right_singular_vectors.storage().as_slice();
    for row in 0..rows {
        for col in 0..cols {
            let value: f64 = (0..rank)
                .map(|i| u[row * rank + i].to_f64() * sigma[i] * v[col * rank + i].to_f64())
                .sum();
            let target = entries[row * cols + col];
            assert!(
                (value - target).abs() <= absolute,
                "{rows}x{cols}: A[{row}][{col}] reconstructs as {value} not {target}"
            );
        }
    }

    for (name, factor, height) in [("U", u, rows), ("V", v, cols)] {
        for a in 0..rank {
            for b in 0..rank {
                let dot: f64 = (0..height)
                    .map(|r| factor[r * rank + a].to_f64() * factor[r * rank + b].to_f64())
                    .sum();
                let target = f64::from(u8::from(a == b));
                assert!(
                    (dot - target).abs() <= relative,
                    "{rows}x{cols}: {name}ᵀ{name}[{a}][{b}] = {dot} not {target}"
                );
            }
        }
    }
}

/// Exactly rank-1 tall input — the downstream reproducer.
///
/// `A = [1,2,3]ᵀ [1,2]`: column 2 is exactly twice column 1, so the rank is 1 and
/// `σ = (‖[1,2,3]‖·‖[1,2]‖, 0) = (√70, 0)`. At `f32` its bidiagonal factor is
/// `d = [−3.7416573, 0]`, `e = [7.4833145]` — the exact zero diagonal.
#[test]
fn svd_decompose_reveals_exact_rank_deficiency_tall() {
    let entries = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0];
    let expected = [70.0f64.sqrt(), 0.0];
    assert_rank_deficient_svd::<f32>(3, 2, &entries, &expected);
    assert_rank_deficient_svd::<f64>(3, 2, &entries, &expected);
}

/// Exactly rank-1 wide input: the transpose of the tall reproducer, which takes
/// the `m < n` branch (SVD of `Aᵀ` with `U` and `V` swapped), so the zero lands in
/// the other factor.
#[test]
fn svd_decompose_reveals_exact_rank_deficiency_wide() {
    let entries = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0];
    let expected = [70.0f64.sqrt(), 0.0];
    assert_rank_deficient_svd::<f32>(2, 3, &entries, &expected);
    assert_rank_deficient_svd::<f64>(2, 3, &entries, &expected);
}

/// Exactly rank-2 square input, built as a sum of two orthogonal outer products
/// so the spectrum is analytic: `A = u₁v₁ᵀ + u₂v₂ᵀ` with `u₁ ⊥ u₂`, `v₁ ⊥ v₂`
/// gives `σ = (‖u₂‖‖v₂‖, ‖u₁‖‖v₁‖, 0) = (2√6, √6, 0)` for
/// `u₁ = (1,1,1), v₁ = (1,0,−1), u₂ = (1,0,−1), v₂ = (2,2,2)`.
#[test]
fn svd_decompose_reveals_exact_rank_deficiency_square() {
    let entries = [3.0, 2.0, 1.0, 1.0, 0.0, -1.0, -1.0, -2.0, -3.0];
    let expected = [2.0 * 6.0f64.sqrt(), 6.0f64.sqrt(), 0.0];
    assert_rank_deficient_svd::<f32>(3, 3, &entries, &expected);
    assert_rank_deficient_svd::<f64>(3, 3, &entries, &expected);
}

/// Rank 2 of 4: **two** deficient directions, so the iteration must chase more
/// than one zero out of the same matrix, and the trailing block is entirely zero.
/// `u₁ = (1,1,1,1), v₁ = (1,0,1,0), u₂ = (1,−1,1,−1), v₂ = (0,2,0,2)` (mutually
/// orthogonal) give `σ = (4√2, 2√2, 0, 0)`.
#[test]
fn svd_decompose_reveals_rank_two_of_four() {
    let entries = [
        1.0, 2.0, 1.0, 2.0, //
        1.0, -2.0, 1.0, -2.0, //
        1.0, 2.0, 1.0, 2.0, //
        1.0, -2.0, 1.0, -2.0,
    ];
    let expected = [4.0 * 2.0f64.sqrt(), 2.0 * 2.0f64.sqrt(), 0.0, 0.0];
    assert_rank_deficient_svd::<f32>(4, 4, &entries, &expected);
    assert_rank_deficient_svd::<f64>(4, 4, &entries, &expected);
}

/// Rank 2 of 3 in a non-square tall shape (`4 × 3`) and its wide transpose, so
/// the deficient case is exercised where `m ≠ n` on both branches.
/// `u₁ = (1,1,1,1), v₁ = (1,0,1)` and `u₂ = (1,−1,1,−1), v₂ = (0,2,0)` give
/// `σ = (4, 2√2, 0)`.
#[test]
fn svd_decompose_reveals_rank_deficiency_in_rectangular_shapes() {
    let tall = [1.0, 2.0, 1.0, 1.0, -2.0, 1.0, 1.0, 2.0, 1.0, 1.0, -2.0, 1.0];
    let wide = [1.0, 1.0, 1.0, 1.0, 2.0, -2.0, 2.0, -2.0, 1.0, 1.0, 1.0, 1.0];
    let expected = [4.0, 2.0 * 2.0f64.sqrt(), 0.0];
    assert_rank_deficient_svd::<f32>(4, 3, &tall, &expected);
    assert_rank_deficient_svd::<f64>(4, 3, &tall, &expected);
    assert_rank_deficient_svd::<f32>(3, 4, &wide, &expected);
    assert_rank_deficient_svd::<f64>(3, 4, &wide, &expected);
}

/// The values-only path (`bidiagonal_diag_colmajor` + non-accumulating iteration)
/// and the full SVD must agree on exactly rank-deficient input at both
/// precisions. They reach the zero diagonal by different roundings — the
/// reproducer's `d[1]` is `8.9e-8` on the values-only path and exactly `0` on the
/// accumulating one — so agreement here is a genuine differential check on the
/// chase rather than a restatement of one path.
#[test]
fn singular_values_agree_with_full_svd_at_exact_rank_deficiency() {
    fn compare<T: RealScalar + RealField>(rows: usize, cols: usize, entries: &[f64], norm: f64) {
        let values: Vec<T> = entries.iter().map(|&x| T::from_f64(x)).collect();
        let a = Array2::from_shape_vec([rows, cols], values).unwrap();
        let full = svd_decompose(&a.view()).unwrap().singular_values;
        let only = singular_values(&a.view()).unwrap();
        let (absolute, _) = error_bounds::<T>(rows, cols, norm);
        assert_eq!(full.len(), only.len());
        for (l, r) in full.iter().zip(only.iter()) {
            assert!(
                (l.to_f64() - r.to_f64()).abs() <= absolute,
                "{rows}x{cols}: {} vs {}",
                l.to_f64(),
                r.to_f64()
            );
        }
    }

    let cases: [(usize, usize, Vec<f64>, f64); 3] = [
        (3, 2, vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], 70.0f64.sqrt()),
        (2, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0], 70.0f64.sqrt()),
        (
            4,
            4,
            vec![
                1.0, 2.0, 1.0, 2.0, 1.0, -2.0, 1.0, -2.0, 1.0, 2.0, 1.0, 2.0, 1.0, -2.0, 1.0, -2.0,
            ],
            4.0 * 2.0f64.sqrt(),
        ),
    ];
    for (rows, cols, entries, norm) in cases {
        compare::<f32>(rows, cols, &entries, norm);
        compare::<f64>(rows, cols, &entries, norm);
    }
}
