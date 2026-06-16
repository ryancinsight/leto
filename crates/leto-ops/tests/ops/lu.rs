use leto::{Array, Storage};
use leto_ops::{det, inv, lu_decompose, matmul, solve};

const EPS: f64 = 1e-10;

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

#[test]
fn solve_matches_closed_form_fixture() {
    let a = Array::from_shape_vec(
        [3, 3],
        vec![4.0f64, -2.0, 1.0, -2.0, 4.0, -2.0, 1.0, -2.0, 4.0],
    )
    .unwrap();
    let rhs = Array::from_shape_vec([3], vec![11.0f64, -16.0, 17.0]).unwrap();

    let x = solve(&a.view(), &rhs.view()).unwrap();

    assert_close_slice(x.storage().as_slice(), &[1.0, -2.0, 3.0]);
}

#[test]
fn det_matches_closed_form_with_pivoting_parity() {
    // Requires row swaps (zero leading pivot), exercising the parity sign.
    let a = Array::from_shape_vec([3, 3], vec![0.0f64, 2.0, 1.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0])
        .unwrap();

    assert_close(det(&a.view()).unwrap(), 58.0);
}

#[test]
fn lu_exposes_pivot_permutation_without_copying() {
    let a = Array::from_shape_vec([3, 3], vec![0.0f64, 2.0, 1.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0])
        .unwrap();

    let lu = lu_decompose(&a.view()).unwrap();

    assert_eq!(lu.pivots(), &[2, 1, 0]);
}

#[test]
fn inv_times_original_is_identity() {
    let values = vec![2.0f64, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 0.5];
    let a = Array::from_shape_vec([3, 3], values.clone()).unwrap();

    let a_inv = inv(&a.view()).unwrap();
    let mut product = Array::zeros([3, 3]);
    matmul(&a.view(), &a_inv.view(), &mut product.view_mut()).unwrap();

    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert_close_slice(product.storage().as_slice(), &identity);

    let expected_inverse = [
        1.0,
        -1.0 / 3.0,
        -2.0 / 3.0,
        1.0,
        0.0,
        -2.0,
        -2.0,
        2.0 / 3.0,
        10.0 / 3.0,
    ];
    assert_close_slice(a_inv.storage().as_slice(), &expected_inverse);
}

#[test]
fn strided_transposed_input_decomposes_by_logical_values() {
    // A^T viewed without copying must factor as the logical transpose.
    let values = vec![4.0f64, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 6.0];
    let a = Array::from_shape_vec([3, 3], values.clone()).unwrap();
    let at = a.transpose([1, 0]).unwrap();

    let det_a = det(&a.view()).unwrap();
    let det_at = det(&at).unwrap();
    // det(A) == det(A^T)
    assert_close(det_at, det_a);
}

#[test]
fn singular_matrix_rejected_for_solve_and_zero_for_det() {
    // Row 2 = 2 × row 1: rank deficient.
    let values = vec![1.0f64, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 0.0, 1.0];
    let a = Array::from_shape_vec([3, 3], values).unwrap();
    let rhs = Array::from_shape_vec([3], vec![1.0f64, 2.0, 3.0]).unwrap();

    assert!(lu_decompose(&a.view()).is_err());
    assert!(solve(&a.view(), &rhs.view()).is_err());
    assert_eq!(det(&a.view()).unwrap(), 0.0);
}

#[test]
fn non_square_and_non_finite_rejected() {
    let rect = Array::from_shape_vec([2, 3], vec![1.0f64; 6]).unwrap();
    assert!(lu_decompose(&rect.view()).is_err());

    let nan = Array::from_shape_vec([2, 2], vec![1.0f64, f64::NAN, 0.0, 1.0]).unwrap();
    assert!(lu_decompose(&nan.view()).is_err());
}

#[test]
fn lu_is_generic_over_reduced_width_scalar() {
    // f32 path proves the generic entry point; well-conditioned 2x2.
    let a = Array::from_shape_vec([2, 2], vec![3.0f32, 1.0, 1.0, 2.0]).unwrap();
    let rhs = Array::from_shape_vec([2], vec![9.0f32, 8.0]).unwrap();
    let x = solve(&a.view(), &rhs.view()).unwrap();
    // Solution of [[3,1],[1,2]] x = [9,8] is [2,3].
    let got = x.storage().as_slice();
    assert!((got[0] - 2.0).abs() < 1e-5 && (got[1] - 3.0).abs() < 1e-5);
}
