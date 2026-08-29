//! Cross-repo consumer contract tests.
//!
//! These pin usage shapes that a downstream consumer relies on in production
//! code, so that a change here surfaces as a leto regression rather than as a
//! consumer build break. The consumer is `kwavers`:
//!
//! - `crates/kwavers-solver/src/inverse/reconstruction/photoacoustic/filters/core.rs`
//!   (`apply_ram_lak_filter`) iterates `columns_mut()` and, per column, applies a
//!   1-D transform via `to_contiguous()` then writes the result back with
//!   `assign`. Columns of a C-order matrix are **interleaved**: each yielded
//!   view's physical window overlaps its siblings', so it reports
//!   `has_exclusive_window() == false` and the whole-window slice accessors are
//!   gated. `to_contiguous` and `assign` must keep working on such views.
//! - `crates/kwavers-solver/.../frequency_continuation.rs` zips a read-only
//!   `axis_iter::<1>(0)` over one array with a mutable `axis_iter_mut::<1>(0)`
//!   over a second and calls `assign` per row. Rows of a C-order matrix are
//!   dense, so those views do have an exclusive window.
//!
//! Both gates arrived with PR #129 (mutable lane/axis injectivity + shared-window
//! accessor gating). Do not relax these assertions to make a change pass: they
//! are a consumer's working code.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use leto::{Array1, Array2, Storage};

const ROWS: usize = 4;
const COLS: usize = 3;

fn source() -> Array2<f64> {
    let values: Vec<f64> = (0..ROWS * COLS).map(|i| i as f64 + 1.0).collect();
    Array2::from_shape_vec([ROWS, COLS], values).unwrap()
}

/// Per-column transform standing in for the consumer's 1-D filter. Reversing
/// makes it order-sensitive within a column; scaling by `col + 1` makes it
/// differ between columns, so a cross-column mix-up cannot pass.
fn filter_column(column: &[f64], col: usize) -> Vec<f64> {
    let scale = col as f64 + 1.0;
    column.iter().rev().map(|value| value * scale).collect()
}

#[test]
fn consumer_contract_column_filter_roundtrip() {
    let mut data = source();
    let original: Vec<f64> = data.storage().as_slice().to_vec();

    for (col, mut view) in data.columns_mut().unwrap().enumerate() {
        // Columns of a C-order matrix interleave with their siblings; the
        // consumer's code path depends on this staying usable.
        assert!(
            !view.has_exclusive_window(),
            "column {col} of a C-order matrix must be an interleaved view"
        );

        let contiguous = view.to_contiguous();
        let filtered =
            Array1::from_shape_vec([ROWS], filter_column(contiguous.storage().as_slice(), col))
                .unwrap();
        view.assign(&filtered);
    }

    // Whole-array check against an oracle computed by plain indexing, so a
    // column written into the wrong place fails here even if each column is
    // internally correct.
    let mut expected = vec![0.0; ROWS * COLS];
    for col in 0..COLS {
        let column: Vec<f64> = (0..ROWS).map(|row| original[row * COLS + col]).collect();
        for (row, value) in filter_column(&column, col).into_iter().enumerate() {
            expected[row * COLS + col] = value;
        }
    }
    assert_eq!(data.storage().as_slice(), &expected);
}

#[test]
fn consumer_contract_row_zip_assign() {
    let input = source();
    let mut output = Array2::from_shape_vec([ROWS, COLS], vec![0.0; ROWS * COLS]).unwrap();

    let input_view = input.view();
    let output_view = output.view_mut();
    let rows_in = input_view.axis_iter::<1>(0).unwrap();
    let rows_out = output_view.axis_iter_mut::<1>(0).unwrap();

    for (row, (source_row, mut target_row)) in rows_in.zip(rows_out).enumerate() {
        // Rows of a C-order matrix are dense, so the window is exclusive.
        assert!(
            target_row.has_exclusive_window(),
            "row {row} of a C-order matrix must be a dense, exclusively-windowed view"
        );

        let response: Vec<f64> = source_row
            .iter()
            .map(|value| value * 2.0 + row as f64)
            .collect();
        let response = Array1::from_shape_vec([COLS], response).unwrap();
        target_row.assign(&response);
    }

    let expected: Vec<f64> = (0..ROWS)
        .flat_map(|row| {
            (0..COLS).map(move |col| (row * COLS + col) as f64 * 2.0 + 2.0 + row as f64)
        })
        .collect();
    assert_eq!(output.storage().as_slice(), &expected);
}
