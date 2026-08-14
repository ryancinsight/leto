#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array, Storage};
use leto_ops::{
    cholesky_decompose, cholesky_det, cholesky_inv, cholesky_solve, matmul, qr_decompose, solve,
    solve_least_squares,
};

const EPS: f64 = 1e-9;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= EPS * expected.abs().max(1.0),
        "actual {actual} expected {expected}"
    );
}

fn assert_close_slice(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_close(*a, *e);
    }
}

// ── QR ──────────────────────────────────────────────────────────────────────

#[test]
fn qr_blocked_path_solves_large_system() {
    // n = 256 reaches BLOCK_MIN_ROWS, so the panel-blocked compact-WY trailing
    // update runs (eight panels); a diagonally-dominant A is well-conditioned, so a
    // square QR solve must recover a known x to tight tolerance — value-semantic
    // proof the blocked factorization equals the unblocked one.
    let n = 256usize;
    let a_values: Vec<f64> = (0..n * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            if i == j {
                100.0 + i as f64
            } else {
                ((i * 7 + j * 3) % 5) as f64 - 2.0
            }
        })
        .collect();
    let x_true: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5) - 7.0).collect();
    let rhs_values: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| a_values[i * n + j] * x_true[j]).sum())
        .collect();

    let a = Array::from_shape_vec([n, n], a_values).unwrap();
    let rhs = Array::from_shape_vec([n], rhs_values).unwrap();
    let x = solve_least_squares(&a.view(), &rhs.view()).unwrap();
    assert_close_slice(x.storage().as_slice(), &x_true);
}

#[test]
fn qr_square_solve_matches_lu() {
    let a_values = vec![4.0f64, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 7.0];
    let rhs_values = vec![13.0f64, 20.0, 31.0];
    let a = Array::from_shape_vec([3, 3], a_values).unwrap();
    let rhs = Array::from_shape_vec([3], rhs_values).unwrap();

    let via_qr = solve_least_squares(&a.view(), &rhs.view()).unwrap();
    let via_lu = solve(&a.view(), &rhs.view()).unwrap();
    assert_close_slice(via_qr.storage().as_slice(), via_lu.storage().as_slice());
}

#[test]
fn qr_least_squares_matches_closed_form_line_fit() {
    // Closed-form normal-equation solution for the full-rank line fit.
    let a_values = vec![
        1.0f64, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, //
    ];
    let rhs_values = vec![1.1f64, 2.9, 5.2, 6.8, 9.1];
    let a = Array::from_shape_vec([5, 2], a_values).unwrap();
    let rhs = Array::from_shape_vec([5], rhs_values).unwrap();

    let x = solve_least_squares(&a.view(), &rhs.view()).unwrap();

    assert_close_slice(x.storage().as_slice(), &[1.04, 1.99]);
}

#[test]
fn qr_least_squares_residual_is_orthogonal_to_columns() {
    // Mathematical optimality property: Aᵀ(A·x − b) = 0 at the minimizer.
    let a_values = vec![2.0f64, 1.0, 1.0, 3.0, 0.0, 1.0, 4.0, 2.0];
    let rhs_values = vec![1.0f64, -2.0, 3.0, 0.5];
    let a = Array::from_shape_vec([4, 2], a_values.clone()).unwrap();
    let rhs = Array::from_shape_vec([4], rhs_values.clone()).unwrap();

    let x = solve_least_squares(&a.view(), &rhs.view()).unwrap();
    let xs = x.storage().as_slice();

    for col in 0..2 {
        let mut dot = 0.0f64;
        for row in 0..4 {
            let residual_row =
                a_values[row * 2] * xs[0] + a_values[row * 2 + 1] * xs[1] - rhs_values[row];
            dot += a_values[row * 2 + col] * residual_row;
        }
        assert!(dot.abs() < 1e-9, "column {col} not orthogonal: {dot}");
    }
}

#[test]
fn qr_rejects_underdetermined_and_rank_deficient() {
    let wide = Array::from_shape_vec([2, 3], vec![1.0f64; 6]).unwrap();
    assert!(qr_decompose(&wide.view()).is_err());

    // Exactly-zero second column: pivot norm is exactly zero at k = 1.
    // (Near-deficiency, e.g. col1 = 2·col0, leaves ~1e-16 residue under
    // floating point and is a conditioning concern, not detected by
    // unpivoted Householder QR — the rejection contract is exact zero.)
    let deficient = Array::from_shape_vec([3, 2], vec![1.0f64, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
    assert!(qr_decompose(&deficient.view()).is_err());
}

// ── Cholesky ────────────────────────────────────────────────────────────────

#[test]
fn cholesky_factor_matches_closed_form_fixture() {
    let a = Array::from_shape_vec(
        [3, 3],
        vec![25.0f64, 15.0, -5.0, 15.0, 18.0, 0.0, -5.0, 0.0, 11.0],
    )
    .unwrap();

    let decomposition = cholesky_decompose(&a.view()).unwrap();
    let l = decomposition.lower();
    let expected = [5.0, 0.0, 0.0, 3.0, 3.0, 0.0, -1.0, 1.0, 3.0];

    for r in 0..3 {
        for c in 0..3 {
            assert_close(*l.get([r, c]).unwrap(), expected[r * 3 + c]);
        }
    }
}

#[test]
fn cholesky_solve_matches_lu_solve() {
    let values = vec![6.0f64, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
    let rhs_values = vec![9.0f64, 9.0, 7.0];
    let a = Array::from_shape_vec([3, 3], values).unwrap();
    let rhs = Array::from_shape_vec([3], rhs_values).unwrap();

    let via_cholesky = cholesky_decompose(&a.view())
        .unwrap()
        .solve(&rhs.view())
        .unwrap();
    let via_lu = solve(&a.view(), &rhs.view()).unwrap();
    assert_close_slice(
        via_cholesky.storage().as_slice(),
        via_lu.storage().as_slice(),
    );

    let via_convenience = cholesky_solve(&a.view(), &rhs.view()).unwrap();
    assert_close_slice(
        via_convenience.storage().as_slice(),
        via_lu.storage().as_slice(),
    );
}

#[test]
fn cholesky_det_and_inv_match_identity_contract() {
    let values = vec![6.0f64, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
    let a = Array::from_shape_vec([3, 3], values.clone()).unwrap();

    let decomposition = cholesky_decompose(&a.view()).unwrap();
    let det = decomposition.det();
    let det_convenience = cholesky_det(&a.view()).unwrap();

    assert_close(det, 83.0);
    assert_close(det_convenience, 83.0);

    let inverse = decomposition.inv().unwrap();
    let inverse_convenience = cholesky_inv(&a.view()).unwrap();
    assert_close_slice(
        inverse.storage().as_slice(),
        inverse_convenience.storage().as_slice(),
    );

    let mut product = Array::zeros([3, 3]);
    matmul(&a.view(), &inverse.view(), &mut product.view_mut()).unwrap();
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert_close_slice(product.storage().as_slice(), &identity);
}

#[test]
fn cholesky_reads_lower_triangle_through_strided_view() {
    // A transposed view of an SPD matrix is the same logical matrix
    // (symmetry), so the factor must be identical.
    let values = vec![25.0f64, 15.0, -5.0, 15.0, 18.0, 0.0, -5.0, 0.0, 11.0];
    let a = Array::from_shape_vec([3, 3], values).unwrap();
    let at = a.transpose([1, 0]).unwrap();

    let direct = cholesky_decompose(&a.view()).unwrap();
    let via_transpose = cholesky_decompose(&at).unwrap();
    assert_close_slice(
        direct.lower().storage().as_slice(),
        via_transpose.lower().storage().as_slice(),
    );
}

#[test]
fn cholesky_rejects_indefinite_and_non_square() {
    // Indefinite: negative eigenvalue.
    let indefinite = Array::from_shape_vec([2, 2], vec![1.0f64, 2.0, 2.0, 1.0]).unwrap();
    assert!(cholesky_decompose(&indefinite.view()).is_err());

    let rect = Array::from_shape_vec([2, 3], vec![1.0f64; 6]).unwrap();
    assert!(cholesky_decompose(&rect.view()).is_err());
}

#[test]
fn qr_and_cholesky_are_generic_over_reduced_width_scalar() {
    // f32 path proves the generic entry points.
    let a = Array::from_shape_vec([2, 2], vec![4.0f32, 2.0, 2.0, 3.0]).unwrap();
    let rhs = Array::from_shape_vec([2], vec![10.0f32, 8.0]).unwrap();

    let via_qr = solve_least_squares(&a.view(), &rhs.view()).unwrap();
    let via_cholesky = cholesky_decompose(&a.view())
        .unwrap()
        .solve(&rhs.view())
        .unwrap();
    for (q, c) in via_qr
        .storage()
        .as_slice()
        .iter()
        .zip(via_cholesky.storage().as_slice())
    {
        assert!((q - c).abs() < 1e-4, "qr {q} vs cholesky {c}");
    }
}

#[test]
fn test_cholesky_solve_into_strided() {
    let values = vec![6.0f64, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
    let rhs_values = vec![9.0f64, 9.0, 7.0];
    let a = Array::from_shape_vec([3, 3], values).unwrap();
    let rhs = Array::from_shape_vec([3], rhs_values).unwrap();

    let decomp = cholesky_decompose(&a.view()).unwrap();

    let mut out_large = Array::from_shape_vec([6], vec![0.0f64; 6]).unwrap();
    {
        let mut out_strided = out_large.slice_mut(&[(0, 6, 2)]).unwrap();
        decomp.solve_into(&rhs.view(), &mut out_strided).unwrap();
    }

    let expected = decomp.solve(&rhs.view()).unwrap();

    assert_close(*out_large.get([0]).unwrap(), *expected.get([0]).unwrap());
    assert_close(*out_large.get([2]).unwrap(), *expected.get([1]).unwrap());
    assert_close(*out_large.get([4]).unwrap(), *expected.get([2]).unwrap());
}

#[test]
fn test_lu_solve_into_strided() {
    let values = vec![6.0f64, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
    let rhs_values = vec![9.0f64, 9.0, 7.0];
    let a = Array::from_shape_vec([3, 3], values).unwrap();
    let rhs = Array::from_shape_vec([3], rhs_values).unwrap();

    let decomp = cholesky_decompose(&a.view()).unwrap();
    let lu_decomp = leto_ops::lu_decompose(&a.view()).unwrap();

    let mut out_large = Array::from_shape_vec([6], vec![0.0f64; 6]).unwrap();
    {
        let mut out_strided = out_large.slice_mut(&[(0, 6, 2)]).unwrap();
        lu_decomp.solve_into(&rhs.view(), &mut out_strided).unwrap();
    }

    let expected = decomp.solve(&rhs.view()).unwrap();

    assert_close(*out_large.get([0]).unwrap(), *expected.get([0]).unwrap());
    assert_close(*out_large.get([2]).unwrap(), *expected.get([1]).unwrap());
    assert_close(*out_large.get([4]).unwrap(), *expected.get([2]).unwrap());
}

#[test]
fn test_qr_solve_least_squares_into_strided() {
    let values = vec![6.0f64, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
    let rhs_values = vec![9.0f64, 9.0, 7.0];
    let a = Array::from_shape_vec([3, 3], values).unwrap();
    let rhs = Array::from_shape_vec([3], rhs_values).unwrap();

    let decomp = qr_decompose(&a.view()).unwrap();

    let mut out_large = Array::from_shape_vec([6], vec![0.0f64; 6]).unwrap();
    {
        let mut out_strided = out_large.slice_mut(&[(0, 6, 2)]).unwrap();
        decomp
            .solve_least_squares_into(&rhs.view(), &mut out_strided)
            .unwrap();
    }

    let expected = decomp.solve_least_squares(&rhs.view()).unwrap();

    assert_close(*out_large.get([0]).unwrap(), *expected.get([0]).unwrap());
    assert_close(*out_large.get([2]).unwrap(), *expected.get([1]).unwrap());
    assert_close(*out_large.get([4]).unwrap(), *expected.get([2]).unwrap());
}
