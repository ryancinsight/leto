#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use super::maps::{assert_close_slice, EPS};
use leto::{Array, Layout, Storage, VecStorage};
use leto_ops::{bessel_k0, j0, j1, sinc, unary_map, unary_map_into, J0Op, J1Op, K0Op, SincOp};

#[test]
fn test_sinc_op_zero_is_exactly_one() {
    let layout = Layout::c_contiguous([3]).unwrap();
    let array = Array::new(layout, VecStorage::new(vec![0.0f64, -0.0, 1.0])).unwrap();

    let mut out = Array::new(layout, VecStorage::fill(3, 0.0f64)).unwrap();
    unary_map_into(SincOp, &array.view(), &mut out.view_mut()).unwrap();
    let got = out.storage().as_slice();
    assert_eq!(got[0], 1.0);
    assert_eq!(got[1], 1.0);
    assert!((got[2] - 1.0f64.sin()).abs() <= EPS);
}

#[test]
fn test_sinc_op_matches_scalar_oracle() {
    let layout = Layout::c_contiguous([5]).unwrap();
    let values = vec![0.5f64, 1.0, -1.0, 2.0, core::f64::consts::PI];
    let array = Array::new(layout, VecStorage::new(values.clone())).unwrap();

    let out = unary_map(SincOp, &array.view()).unwrap();
    let expected: Vec<f64> = values.iter().map(|&x| sinc(x)).collect();
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_sinc_op_f32_and_nan() {
    let layout = Layout::c_contiguous([4]).unwrap();
    let array = Array::new(layout, VecStorage::new(vec![0.0f32, 1.0, -2.0, f32::NAN])).unwrap();

    let out = unary_map(SincOp, &array.view()).unwrap();
    let got = out.storage().as_slice();
    assert_eq!(got[0], 1.0);
    assert!((got[1] - 1.0f32.sin()).abs() <= 1e-6);
    assert!((got[2] - (-2.0f32).sin() / -2.0).abs() <= 1e-6);
    assert!(got[3].is_nan());
}

#[test]
fn test_bessel_ops_match_scalar_oracle_both_branches() {
    // Values span the rational branch, the |x| = 8 crossover, the Hankel
    // branch, and negative arguments (J0 even, J1 odd).
    let values = vec![0.0f64, 0.5, 1.0, 2.0, 5.0, 7.999, 8.0, 9.0, 12.0, -3.0];
    let layout = Layout::c_contiguous([values.len()]).unwrap();
    let array = Array::new(layout, VecStorage::new(values.clone())).unwrap();

    let j0_out = unary_map(J0Op, &array.view()).unwrap();
    let expected_j0: Vec<f64> = values.iter().map(|&x| j0(x)).collect();
    assert_close_slice(j0_out.storage().as_slice(), &expected_j0);

    let mut j1_out = Array::new(layout, VecStorage::fill(values.len(), 0.0f64)).unwrap();
    unary_map_into(J1Op, &array.view(), &mut j1_out.view_mut()).unwrap();
    let expected_j1: Vec<f64> = values.iter().map(|&x| j1(x)).collect();
    assert_close_slice(j1_out.storage().as_slice(), &expected_j1);
}

#[test]
fn test_bessel_ops_f32_within_lane_precision() {
    let values = vec![0.0f32, 1.0, 4.0, 9.0];
    let layout = Layout::c_contiguous([values.len()]).unwrap();
    let array = Array::new(layout, VecStorage::new(values.clone())).unwrap();

    let j0_out = unary_map(J0Op, &array.view()).unwrap();
    for (got, &x) in j0_out.storage().as_slice().iter().zip(&values) {
        let expected = j0(x as f64) as f32;
        assert!(
            (got - expected).abs() <= 1e-5,
            "j0({x}): got {got} expected {expected}"
        );
    }
    let j1_out = unary_map(J1Op, &array.view()).unwrap();
    for (got, &x) in j1_out.storage().as_slice().iter().zip(&values) {
        let expected = j1(x as f64) as f32;
        assert!(
            (got - expected).abs() <= 1e-5,
            "j1({x}): got {got} expected {expected}"
        );
    }
}

#[test]
fn test_k0_op_matches_scalar_oracle_both_branches() {
    // Values span the 9.8.5 branch, the x = 2 crossover, and the 9.8.6
    // branch; the reference values are DLMF 10.32 / A&S Table 9.8 via the
    // scalar oracle.
    let values = vec![0.1f64, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0];
    let layout = Layout::c_contiguous([values.len()]).unwrap();
    let array = Array::new(layout, VecStorage::new(values.clone())).unwrap();

    let out = unary_map(K0Op, &array.view()).unwrap();
    let expected: Vec<f64> = values.iter().map(|&x| bessel_k0(x)).collect();
    assert_close_slice(out.storage().as_slice(), &expected);
}

#[test]
fn test_k0_op_rejects_nonpositive_and_nonfinite() {
    let values = vec![
        0.0f64,
        -0.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let layout = Layout::c_contiguous([values.len()]).unwrap();
    let array = Array::new(layout, VecStorage::new(values)).unwrap();

    let mut out = Array::new(layout, VecStorage::fill(6, 0.0f64)).unwrap();
    unary_map_into(K0Op, &array.view(), &mut out.view_mut()).unwrap();
    for got in out.storage().as_slice() {
        assert!(got.is_nan(), "K0 outside (0, +inf) must be NaN");
    }
}

#[test]
fn test_k0_op_f32_within_lane_precision() {
    let values = vec![0.1f32, 1.0, 2.0, 5.0];
    let layout = Layout::c_contiguous([values.len()]).unwrap();
    let array = Array::new(layout, VecStorage::new(values.clone())).unwrap();

    let out = unary_map(K0Op, &array.view()).unwrap();
    for (got, &x) in out.storage().as_slice().iter().zip(&values) {
        let expected = bessel_k0(x as f64) as f32;
        assert!(
            (got - expected).abs() <= 1e-5,
            "k0({x}): got {got} expected {expected}"
        );
    }
}
