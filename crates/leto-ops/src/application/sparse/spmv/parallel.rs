//! The gated parallel row driver for CSR SpMV.
//!
//! Rows are the unit of work and they are independent, so the parallel path
//! splits `y` into runs, hands each run the matching `row_ptr` window, and lets
//! [`spmv_row_window_into`](super::rows::spmv_row_window_into) walk it. No
//! per-row result changes with the split — every row sums its own entries in
//! the same fixed order — so the parallel path is bitwise identical to the
//! serial one, and no scratch, reduction buffer, or allocation is involved.

use super::rows::spmv_row_window_into;
use crate::domain::scalar::Scalar;

/// Nonzeros past which the product spreads over the runtime's workers.
///
/// The kernel is gather-bound rather than a pure stream: each row's gather
/// loads and dependent accumulate chain leave a single core waiting on load
/// latency, so extra cores pay before the values and column indices outgrow
/// the last-level cache — but the runtime's spawn-and-join is not free either.
/// Measured on a 24-worker host with the `kernels` bench (`spmv/banded7`, a
/// `f64` seven-point banded operator, 50 samples, serial and gated runs
/// back-to-back):
///
/// | rows | nnz | serial | spread | ratio |
/// | ---: | ---: | ---: | ---: | ---: |
/// | 4,096 | 28,660 | 11.3 µs | 12.9 µs | 0.88 |
/// | 16,384 | 114,676 | 45.3 µs | 21.1 µs | 2.15 |
/// | 32,768 | 229,352 | 121.3 µs | 35.4 µs | 3.42 |
/// | 65,536 | 458,740 | 304.3 µs | 60.5 µs | 5.03 |
/// | 1,048,576 | 7,340,020 | 7.02 ms | 2.19 ms | 3.21 |
///
/// The split loses at 28,660 nonzeros and wins from 114,676 up, so the gate
/// sits between them at a round power of two. It is a nonzero count rather than
/// a byte total because the cost is the per-nonzero gather, not the footprint:
/// this is latency-bound well inside the cache, unlike the bandwidth-bound
/// passes that gate on
/// [`working_set_spills_llc`](crate::infrastructure::cache::working_set_spills_llc).
#[cfg(feature = "parallel")]
pub(super) const PARALLEL_MIN_SPMV_NNZ: usize = 65_536;

/// Bytes one nonzero streams through the kernel: its value plus its column
/// index. Task width is sized from it (moirai ADR 0059).
#[inline]
fn nonzero_stream_bytes<T: Scalar>() -> usize {
    core::mem::size_of::<T>() + core::mem::size_of::<usize>()
}

/// Whether an `nnz`-nonzero product should spread over the runtime's workers.
#[cfg(feature = "parallel")]
#[inline]
pub(super) fn should_parallelize(nnz: usize) -> bool {
    nnz >= PARALLEL_MIN_SPMV_NNZ
}

/// Compute `y = A · x` with consecutive rows spread over moirai unit tasks.
///
/// `y.len()` is `nrows` and `row_ptr.len()` is `nrows + 1`; both are checked by
/// the caller. Each run receives a disjoint sub-slice of `y` and the
/// `row_ptr` window covering exactly those rows, so the driver never
/// materializes per-row index lists and the row kernel stays the only place
/// that knows how a row is summed.
#[cfg(feature = "parallel")]
pub(super) fn spmv_rows_parallel<T: Scalar>(
    values: &[T],
    col_indices: &[usize],
    row_ptr: &[usize],
    xs: &[T],
    y: &mut [T],
) {
    let rows = y.len();
    // Task width follows the bytes a row streams: the average nonzero count
    // per row, in nonzero-stream bytes. The gate only opens past
    // `PARALLEL_MIN_SPMV_NNZ`, so `rows` is non-zero here; keep the divisor
    // defined regardless.
    let row_bytes = if rows == 0 {
        1
    } else {
        values
            .len()
            .saturating_mul(nonzero_stream_bytes::<T>())
            .saturating_div(rows)
            .max(1)
    };
    crate::infrastructure::parallel::for_each_unit_run_mut(y, row_bytes, |first, run| {
        let window = &row_ptr[first..=first + run.len()];
        spmv_row_window_into(values, col_indices, window, xs, run);
    });
}
