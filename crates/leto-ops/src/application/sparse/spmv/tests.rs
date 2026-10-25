//! Value oracles for the row kernel and the parallel driver.
//!
//! The row kernel fixes its reduction order (four accumulators, fixed final
//! fold), so it is *not* bitwise equal to a stored-order naive sum; that
//! comparison carries a rounding bound. The parallel path must however be
//! bitwise identical to the serial one — rows are independent and each sums its
//! own entries in that same fixed order — so it is compared exactly.

#![allow(clippy::unwrap_used, reason = "test scope")]

use super::*;
use crate::application::sparse::CsrMatrix;
use eunomia::FloatElement;
use leto::{Array, Array1, Storage};

/// Seven-point banded operator: `2` on the diagonal, `-1` on the ±half_band
/// off-diagonals. Same shape the scaling bench times.
fn banded(n: usize, half_band: usize) -> CsrMatrix<f64> {
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptr = vec![0usize];
    for row in 0..n {
        for col in row.saturating_sub(half_band)..=(row + half_band).min(n - 1) {
            values.push(if col == row { 2.0 } else { -1.0 });
            col_indices.push(col);
        }
        row_ptr.push(values.len());
    }
    CsrMatrix::from_parts(values, col_indices, row_ptr, n, n).unwrap()
}

fn signal(n: usize) -> Array1<f64> {
    Array::from_shape_vec(
        [n],
        (0..n).map(|i| (f64::from_count(i) * 0.5).sin()).collect(),
    )
    .unwrap()
}

fn naive(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let view = a.as_view();
    let (values, col_indices, row_ptr) = view.as_parts();
    (0..view.nrows())
        .map(|row| {
            let (start, end) = (row_ptr[row], row_ptr[row + 1]);
            (start..end).fold(0.0, |acc, p| acc + values[p] * x[col_indices[p]])
        })
        .collect()
}

fn assert_close(actual: &[f64], expected: &[f64], epsilon: f64) {
    assert_eq!(actual.len(), expected.len());
    for (row, (a, e)) in actual.iter().zip(expected).enumerate() {
        let bound = epsilon * e.abs().max(1.0);
        assert!(
            (a - e).abs() <= bound,
            "row {row}: {a} differs from the naive sum {e} by more than {bound:e}"
        );
    }
}

#[test]
fn row_window_walk_matches_the_naive_row_loop() {
    let n = 4096;
    let a = banded(n, 3);
    let x = signal(n);
    let mut y = vec![0.0; n];
    spmv_into(&a, &x.view(), &mut y).unwrap();
    // Seven-term rows: the fixed reassociation loses at most a few ulps.
    assert_close(&y, &naive(&a, x.storage().as_slice()), 1e-14);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_rows_are_bitwise_identical_to_serial() {
    let n = 4096;
    let a = banded(n, 3);
    let x = signal(n);
    let xs = x.storage().as_slice();
    let view = a.as_view();
    let (values, col_indices, row_ptr) = view.as_parts();

    let mut serial = vec![0.0; n];
    rows::spmv_row_window_into(values, col_indices, row_ptr, xs, &mut serial);

    // Run the driver directly: the gate keeps this size serial, and the point
    // here is the row split itself, not the policy that opens it.
    let mut spread = vec![0.0; n];
    parallel::spmv_rows_parallel(values, col_indices, row_ptr, xs, &mut spread);

    assert_eq!(spread, serial, "row split changed a row's result");
    assert_close(&spread, &naive(&a, xs), 1e-14);
}

#[cfg(feature = "parallel")]
#[test]
fn gate_opens_at_the_measured_nonzero_crossover() {
    // The measured split loses at 28,660 nonzeros (0.88x) and wins from
    // 114,676 (2.15x); the gate sits between them at a round power of two.
    assert_eq!(parallel::PARALLEL_MIN_SPMV_NNZ, 65_536);
    assert!(!parallel::should_parallelize(0));
    assert!(!parallel::should_parallelize(28_660));
    assert!(!parallel::should_parallelize(65_535));
    assert!(parallel::should_parallelize(65_536));
    assert!(parallel::should_parallelize(458_740));
}
