//! Quantile / median reductions: closed-form analytical oracles.
//!
//! Oracles use the numpy/ndarray-stats fractional-rank definition
//! `h = q·(n−1)` with linear interpolation between bracketing order statistics;
//! every expected value below is computed by hand from that definition.

use leto::application::reduction::{
    median_all, median_axis, quantile_all, quantile_axis, Interpolation,
};
use leto::{Array2, LetoError, Storage};

#[track_caller]
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
        "actual {actual} expected {expected}"
    );
}

fn leto(shape: [usize; 2], data: Vec<f64>) -> Array2<f64> {
    Array2::from_shape_vec(shape, data).unwrap()
}

#[track_caller]
fn assert_storage_reason(result: leto::Result<f64>, expected: &str) {
    match result {
        Err(LetoError::StorageError { reason }) => assert_eq!(reason, expected),
        other => panic!("expected storage error {expected:?}, got {other:?}"),
    }
}

#[test]
fn quantile_all_linear_closed_form() {
    // [1,2,3,4]: h = q·3.
    let a = leto([1, 4], vec![1.0, 2.0, 3.0, 4.0]);
    // q=0.5 → h=1.5, g=0.5 → 2 + 0.5·1 = 2.5
    assert_close(quantile_all(&a, 0.5, Interpolation::Linear).unwrap(), 2.5);
    // q=0.25 → h=0.75 → 1 + 0.75·1 = 1.75
    assert_close(quantile_all(&a, 0.25, Interpolation::Linear).unwrap(), 1.75);
    // boundaries q=0 → min, q=1 → max
    assert_close(quantile_all(&a, 0.0, Interpolation::Linear).unwrap(), 1.0);
    assert_close(quantile_all(&a, 1.0, Interpolation::Linear).unwrap(), 4.0);
}

#[test]
fn quantile_all_methods_on_tie() {
    // [1,2,3,4], q=0.5 → h=1.5, lo=1, g=0.5.
    let a = leto([1, 4], vec![1.0, 2.0, 3.0, 4.0]);
    assert_close(quantile_all(&a, 0.5, Interpolation::Lower).unwrap(), 2.0);
    assert_close(quantile_all(&a, 0.5, Interpolation::Higher).unwrap(), 3.0);
    assert_close(quantile_all(&a, 0.5, Interpolation::Midpoint).unwrap(), 2.5);
    // Nearest with g=0.5 and odd lo (=1) rounds to even index 2 → 3.
    assert_close(quantile_all(&a, 0.5, Interpolation::Nearest).unwrap(), 3.0);
}

#[test]
fn median_all_odd_and_even() {
    // even length → interpolated midpoint
    let even = leto([1, 4], vec![1.0, 2.0, 3.0, 4.0]);
    assert_close(median_all(&even).unwrap(), 2.5);
    // odd length → exact middle order statistic (h integral, g=0)
    let odd = leto([1, 5], vec![5.0, 1.0, 3.0, 2.0, 4.0]);
    assert_close(median_all(&odd).unwrap(), 3.0);
}

#[test]
fn quantile_unsorted_input() {
    // Sorting is internal: shuffled data yields the same quantile.
    let a = leto([1, 5], vec![4.0, 1.0, 5.0, 2.0, 3.0]);
    // sorted [1,2,3,4,5], q=0.25 → h=1.0, g=0 → v[1]=2
    assert_close(quantile_all(&a, 0.25, Interpolation::Linear).unwrap(), 2.0);
}

#[test]
fn median_axis_closed_form() {
    // [[1,2],[3,4]]
    let a = leto([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    // axis 0: cols [1,3],[2,4] → medians 2, 3
    let m0 = median_axis::<f64, _, 2, 1>(&a, 0).unwrap();
    assert_close(m0.storage().as_slice()[0], 2.0);
    assert_close(m0.storage().as_slice()[1], 3.0);
    // axis 1: rows [1,2],[3,4] → medians 1.5, 3.5
    let m1 = median_axis::<f64, _, 2, 1>(&a, 1).unwrap();
    assert_close(m1.storage().as_slice()[0], 1.5);
    assert_close(m1.storage().as_slice()[1], 3.5);
}

#[test]
fn quantile_axis_matches_quantile_all_per_lane() {
    // 3×4: quantile_axis along axis 1 must equal quantile_all over each row.
    let data: Vec<f64> = (0..12).map(f64::from).collect();
    let a = leto([3, 4], data);
    let q = 0.4;
    let axis_q = quantile_axis::<f64, _, 2, 1>(&a, 1, q, Interpolation::Linear).unwrap();
    for row in 0..3usize {
        let row_data: Vec<f64> = (0..4).map(|c| f64::from((row * 4 + c) as i32)).collect();
        let row_arr = leto([1, 4], row_data);
        let expected = quantile_all(&row_arr, q, Interpolation::Linear).unwrap();
        assert_close(axis_q.storage().as_slice()[row], expected);
    }
}

#[test]
fn quantile_rejects_empty_q_range_and_nan() {
    let a = leto([1, 3], vec![1.0, 2.0, 3.0]);
    assert_storage_reason(
        quantile_all(&a, -0.1, Interpolation::Linear),
        "quantile q must be finite and within [0, 1]",
    );
    assert_storage_reason(
        quantile_all(&a, 1.5, Interpolation::Linear),
        "quantile q must be finite and within [0, 1]",
    );
    assert_storage_reason(
        quantile_all(&a, f64::NAN, Interpolation::Linear),
        "quantile q must be finite and within [0, 1]",
    );
    let with_nan = leto([1, 3], vec![1.0, f64::NAN, 3.0]);
    assert_storage_reason(
        quantile_all(&with_nan, 0.5, Interpolation::Linear),
        "quantile over data containing NaN is undefined",
    );
    let empty: Array2<f64> = Array2::from_shape_vec([0, 0], vec![]).unwrap();
    assert_storage_reason(
        quantile_all(&empty, 0.5, Interpolation::Linear),
        "quantile over empty array is undefined",
    );
    assert_storage_reason(median_all(&empty), "quantile over empty array is undefined");
}
