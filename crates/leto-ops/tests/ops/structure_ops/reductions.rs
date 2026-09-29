//! Reductions (`min`/`max`) and the deterministic RNG distributions.

use super::*;

#[test]
fn test_reduce_min_max_contiguous_inputs() {
    let input = arr([2, 3], vec![3.0, -7.0, 2.0, 11.0, 0.5, -1.0]);

    assert_eq!(reduce_min(&input.view()).unwrap(), -7.0);
    assert_eq!(reduce_max(&input.view()).unwrap(), 11.0);
}

#[test]
fn test_reduce_min_max_sliced_inputs_follow_logical_view() {
    let input = arr(
        [3, 4],
        vec![
            100.0, -2.0, 9.0, 5.0, 7.0, -11.0, 3.0, 42.0, -99.0, 6.0, -4.0, 10.0,
        ],
    );
    let view = input
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(Some(1), Some(4), 2)])
        .unwrap();

    assert_eq!(reduce_min(&view).unwrap(), -11.0);
    assert_eq!(reduce_max(&view).unwrap(), 42.0);
}

#[test]
fn test_reduce_min_max_reject_empty_inputs() {
    let empty = arr([0], Vec::<f64>::new());

    let min_err = reduce_min(&empty.view()).unwrap_err();
    let max_err = reduce_max(&empty.view()).unwrap_err();

    assert_eq!(
        min_err,
        LetoError::StorageError {
            reason: "all-elements reduction requires a non-empty input".to_string()
        }
    );
    assert_eq!(
        max_err,
        LetoError::StorageError {
            reason: "all-elements reduction requires a non-empty input".to_string()
        }
    );
}

#[test]
fn test_uniform_is_deterministic_and_in_range() {
    let a = uniform_with_seed([1000], -2.0, 5.0, 42).unwrap();
    let b = uniform_with_seed([1000], -2.0, 5.0, 42).unwrap();
    assert_eq!(a.storage().as_slice(), b.storage().as_slice());
    for &v in a.storage().as_slice() {
        assert!((-2.0..5.0).contains(&v), "out of range: {v}");
    }
    // Different seed yields a different stream.
    let c = uniform_with_seed([1000], -2.0, 5.0, 43).unwrap();
    assert_ne!(a.storage().as_slice(), c.storage().as_slice());
}

#[test]
fn test_uniform_mean_matches_closed_form() {
    let n = 100_000usize;
    let a = uniform_with_seed([n], 0.0, 1.0, 7).unwrap();
    let mean: f64 = a.storage().as_slice().iter().sum::<f64>() / n as f64;
    // Closed-form mean of U(0,1) is 0.5; sampling error well under 0.02 at this n.
    assert!((mean - 0.5).abs() < 0.02, "mean {mean}");
}

#[test]
fn test_normal_mean_and_std_match_closed_form() {
    let n = 100_000usize;
    let a = normal_with_seed([n], 1.0, 2.0, 11).unwrap();
    let data = a.storage().as_slice();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let std = var.sqrt();
    assert!((mean - 1.0).abs() < 0.05, "mean {mean}");
    assert!((std - 2.0).abs() < 0.05, "std {std}");
}
