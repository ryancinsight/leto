//! Value oracles for CSC SpMV: the serial column walk, the strided entry, and
//! the gated row-range driver's bitwise agreement with it.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::*;
use crate::application::sparse::CscMatrix;
use eunomia::FloatElement;

#[test]
fn csc_spmv_basic() {
    let a = CscMatrix::from_dense(
        &leto::Array2::from_shape_vec([3, 2], vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0])
            .unwrap()
            .view(),
    );
    let x = leto::Array1::from_shape_vec([2], vec![2.0, 5.0]).unwrap();
    let y = csc_spmv(&a, &x.view()).unwrap();
    assert_eq!(
        y,
        leto::Array1::from_shape_vec([3], vec![2.0, 10.0, 6.0]).unwrap()
    );
}

#[test]
fn csc_spmv_identity() {
    let a = CscMatrix::from_dense(
        &leto::Array2::from_shape_vec([2, 2], vec![1.0, 0.0, 0.0, 1.0])
            .unwrap()
            .view(),
    );
    let x = leto::Array1::from_shape_vec([2], vec![3.0, 7.0]).unwrap();
    let y = csc_spmv(&a, &x.view()).unwrap();
    assert_eq!(
        y,
        leto::Array1::from_shape_vec([2], vec![3.0, 7.0]).unwrap()
    );
}

#[test]
fn column_window_scatters_only_its_rows() {
    // Column 0 stores rows {0, 3} and column 1 stores rows {3, 5}. The window
    // rows [3, 5) must absorb both columns' row-3 entries and nothing else.
    let values = vec![2.0f64, 3.0, 5.0, 7.0];
    let row_indices = vec![0usize, 3, 3, 5];
    let col_ptr = [0usize, 2, 4];
    let xs = [1.0f64, 10.0];
    let mut y = vec![0.0f64; 2];
    for (j, window) in col_ptr.windows(2).enumerate() {
        columns::scatter_column_window(
            &values,
            &row_indices,
            window[0],
            window[1],
            xs[j],
            3,
            &mut y,
        );
    }
    assert_eq!(y, [3.0 + 50.0, 0.0]);
}

#[test]
fn csc_spmv_shape_mismatch() {
    let a = CscMatrix::<f64>::zeros(3, 2);
    let x = leto::Array1::from_shape_vec([3], vec![1.0, 2.0, 3.0]).unwrap();
    assert!(csc_spmv(&a, &x.view()).is_err());
}

/// A CSC matrix whose columns are long enough to clear the parallel gate:
/// `nrows` rows, each column `j` covering rows `[2j, 2j + degree)` clipped at
/// `nrows`. Row indices stay strictly increasing per column, and the mixed-sign
/// values keep the sums from cancelling into an exact-zero answer that would
/// hide an ordering mistake.
#[cfg(feature = "parallel")]
fn long_column_csc(nrows: usize, ncols: usize, degree: usize) -> (CscMatrix<f64>, Vec<f64>) {
    let mut values = Vec::new();
    let mut row_indices = Vec::new();
    let mut col_ptr = vec![0];
    for j in 0..ncols {
        let lo = 2 * j % nrows;
        let hi = lo.saturating_add(degree).min(nrows);
        for i in lo..hi {
            row_indices.push(i);
            let mixed = (i * 37 + j * 11) % 101;
            values.push((f64::from_count(mixed) - 50.0) / 7.0);
        }
        col_ptr.push(values.len());
    }
    let matrix = CscMatrix::from_parts(values, row_indices, col_ptr, nrows, ncols)
        .expect("fixture CSC is valid");
    let x: Vec<f64> = (0..ncols)
        .map(|j| f64::from_count(j).mul_add(0.5, 1.0))
        .collect();
    (matrix, x)
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_gate_keys_on_column_degree() {
    assert!(!parallel::should_parallelize(0, 0));
    // Past the nonzero floor, the degree decides: 65,536 nonzeros is a degree of
    // 32 over 2,048 columns and of 64 over 1,024, which is the gate.
    assert!(!parallel::should_parallelize(65_535, 64));
    assert!(!parallel::should_parallelize(65_536, 2_048));
    assert!(parallel::should_parallelize(65_536, 1_024));
}

#[cfg(feature = "parallel")]
#[test]
fn row_range_driver_matches_the_serial_walk_bitwise() {
    let (a, xs) = long_column_csc(1024, 512, 300);
    let view = a.as_view();
    let (values, row_indices, col_ptr) = view.as_parts();

    let mut serial = vec![0.0f64; view.nrows()];
    columns::csc_column_run_into(values, row_indices, col_ptr, &xs, &mut serial);

    let mut spread = vec![0.0f64; view.nrows()];
    parallel::csc_spmv_rows_parallel(values, row_indices, col_ptr, &xs, &mut spread);

    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&spread), bits(&serial));
}

#[cfg(feature = "parallel")]
#[test]
fn gated_entry_matches_the_serial_entry_on_long_columns() {
    let (a, xs) = long_column_csc(1024, 512, 300);
    let view = a.as_view();
    let (values, row_indices, col_ptr) = view.as_parts();
    let mut serial = vec![0.0f64; view.nrows()];
    columns::csc_column_run_into(values, row_indices, col_ptr, &xs, &mut serial);

    let x = leto::Array1::from_shape_vec([view.ncols()], xs).unwrap();
    let mut y = vec![0.0f64; view.nrows()];
    csc_spmv_into(&a, &x.view(), &mut y).unwrap();

    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&y), bits(&serial));
}
