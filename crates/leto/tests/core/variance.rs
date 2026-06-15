//! Variance / standard-deviation reductions: closed-form + ndarray differential.

use leto::application::reduction::{std_all, std_axis, var_all, var_axis};
use leto::{Array2, Storage};
use ndarray::{Array2 as NdArray2, Axis};

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

fn nd(shape: [usize; 2], data: Vec<f64>) -> NdArray2<f64> {
    NdArray2::from_shape_vec((shape[0], shape[1]), data).unwrap()
}

#[test]
fn var_all_closed_form_population_and_sample() {
    // [1,2,3,4]: mean 2.5, Σ(x−x̄)² = 5. ddof=0 → 1.25, ddof=1 → 5/3.
    let a = leto([1, 4], vec![1.0, 2.0, 3.0, 4.0]);
    assert_close(var_all(&a, 0.0).unwrap(), 1.25);
    assert_close(var_all(&a, 1.0).unwrap(), 5.0 / 3.0);
    assert_close(std_all(&a, 0.0).unwrap(), 1.25_f64.sqrt());
}

#[test]
fn var_all_matches_ndarray() {
    let data = vec![6.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
    let a = leto([3, 3], data.clone());
    let na = nd([3, 3], data);
    assert_close(var_all(&a, 0.0).unwrap(), na.var(0.0));
    assert_close(var_all(&a, 1.0).unwrap(), na.var(1.0));
    assert_close(std_all(&a, 1.0).unwrap(), na.std(1.0));
}

#[test]
fn var_axis_matches_ndarray() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for axis in 0..2usize {
        let a = leto([2, 3], data.clone());
        let na = nd([2, 3], data.clone());
        for &ddof in &[0.0_f64, 1.0] {
            let leto_var = var_axis::<f64, _, 2, 1>(&a, axis, ddof).unwrap();
            let nd_var = na.var_axis(Axis(axis), ddof);
            assert_eq!(leto_var.storage().as_slice().len(), nd_var.len());
            for (l, n) in leto_var
                .storage()
                .as_slice()
                .iter()
                .zip(nd_var.as_slice().unwrap().iter())
            {
                assert_close(*l, *n);
            }
            // std_axis = sqrt(var_axis).
            let leto_std = std_axis::<f64, _, 2, 1>(&a, axis, ddof).unwrap();
            for (s, v) in leto_std
                .storage()
                .as_slice()
                .iter()
                .zip(leto_var.storage().as_slice().iter())
            {
                assert_close(*s, v.sqrt());
            }
        }
    }
}

#[test]
fn var_axis_population_closed_form() {
    // [[1,2],[3,4]]: axis 0 cols [1,3],[2,4] → var 1, 1. axis 1 rows → 0.25, 0.25.
    let a = leto([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let v0 = var_axis::<f64, _, 2, 1>(&a, 0, 0.0).unwrap();
    assert_close(v0.storage().as_slice()[0], 1.0);
    assert_close(v0.storage().as_slice()[1], 1.0);
    let v1 = var_axis::<f64, _, 2, 1>(&a, 1, 0.0).unwrap();
    assert_close(v1.storage().as_slice()[0], 0.25);
    assert_close(v1.storage().as_slice()[1], 0.25);
}

#[test]
fn variance_rejects_empty_and_excess_ddof() {
    let a = leto([1, 3], vec![1.0, 2.0, 3.0]);
    // ddof == n ⇒ non-positive degrees of freedom.
    assert!(var_all(&a, 3.0).is_err());
    assert!(var_all(&a, f64::NAN).is_err());
    assert!(var_axis::<f64, _, 2, 1>(&a, 1, f64::INFINITY).is_err());
    let empty: Array2<f64> = Array2::from_shape_vec([0, 0], vec![]).unwrap();
    assert!(var_all(&empty, 0.0).is_err());
}
