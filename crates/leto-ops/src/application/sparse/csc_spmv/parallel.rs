//! The gated parallel row-range driver for CSC SpMV.
//!
//! CSC's kernel is a scatter-add, so the unit of parallel work cannot be the
//! nonzero run the way it is for CSR: two columns may write the same `y[i]`.
//! Rows are the one decomposition whose writes are disjoint, and column-major
//! storage still lets a row range find its own entries — rows ascend within each
//! column, so a task rejects or binary-searches each column by its first and
//! last stored row and scatters only the entries inside its range
//! ([`scatter_column_window`](super::columns::scatter_column_window)). Every
//! `y[i]` therefore still accumulates its columns in ascending order, the same
//! order the serial walk uses: the parallel path is bitwise identical to it and
//! needs no per-thread accumulator, reduction pass or allocation.
//!
//! What the split adds is one column scan per task, not per nonzero: a task must
//! look at every column to learn whether the column reaches its rows. So the
//! split pays only where the useful work per task (`nnz / tasks` scatter-adds)
//! dwarfs that scan (`ncols` boundary checks) — a question about the average
//! column degree, and about how many tasks the pass cuts itself into.

use super::columns::scatter_column_window;
use crate::domain::scalar::Scalar;

/// Nonzeros past which the CSC product may spread over the runtime's workers.
///
/// The degree condition below is the one that decides for this format; this
/// floor only keeps the split off products too small to hand any worker a task
/// worth taking.
#[cfg(feature = "parallel")]
pub(super) const PARALLEL_MIN_CSC_NNZ: usize = 65_536;

/// Average nonzeros per column a CSC product needs before the row-range split
/// pays.
///
/// Each task scans all `ncols` columns to find the ones reaching its rows, while
/// the scatter-adds themselves split across the workers. At `nnz = ncols · deg`
/// a task owns `nnz / tasks` additions against `ncols` scans, so the scan's
/// share is `tasks / deg`: the split needs columns holding many more entries
/// than the pass has tasks.
///
/// Measured on a 24-worker host with the `kernels` bench (`csc_spmv/longcols`,
/// a degree sweep over a 32,768-column `f64` operator plus one 65,536-column
/// point). The gate is compile-time, so each column is the median of three
/// binary runs per arm, with the arm chosen by moving this constant between
/// campaigns:
///
/// | degree | nnz | serial | spread | ratio |
/// | ---: | ---: | ---: | ---: | ---: |
/// | 16 | 524,288 | 203 µs | 207 µs | 0.98 |
/// | 32 | 1,048,576 | 658 µs | 388 µs | 1.70 |
/// | 64 | 2,097,152 | 2.128 ms | 578 µs | 3.68 |
/// | 128 | 4,194,304 | 3.765 ms | 1.155 ms | 3.26 |
/// | 128 (65,536 columns) | 8,388,608 | 6.996 ms | 3.216 ms | 2.18 |
/// | 384 | 12,582,912 | 11.424 ms | 4.151 ms | 2.75 |
///
/// The crossover is between 16 and 32: at a degree of 16 every column is
/// examined and barely any entry is scattered, so the split lands at parity
/// (0.98x), while 32 already wins 1.70x. 64 is the conservative side of that
/// crossover — there the serial walk has left cache (2.128 ms is 3.2x the
/// 32-degree time for twice the nonzeros) and the split's win more than
/// doubles. The gate does not go to 128: that would run a degree-64 product
/// serially and give up its 3.68x, which is the largest measured win on the
/// sweep. Each arm's three runs agree within about 12 % here, so these ratios
/// are repeatable on this host, unlike the single-run comparisons that first
/// picked the gate.
#[cfg(feature = "parallel")]
pub(super) const PARALLEL_MIN_COLUMN_DEGREE: usize = 64;

/// Scatter-adds a task must own per column it scans.
///
/// The pass takes one strip per worker, not more: a task scans every column, so
/// oversubscribing multiplies that scan instead of hiding it. Measured on the
/// same 24-worker host with the `kernels` bench (`csc_spmv/longcols384_32768`,
/// a 32,768-column `f64` operator of degree 384): 24 strips run 3.30 ms against
/// 48 strips' 5.15 ms and 16 strips' 3.44 ms. This ratio caps the strips only on
/// a host with more workers than the columns can afford — the gate already
/// demands a degree of 64, so a worker count past `degree / 8` would otherwise
/// spend an eighth of the product's additions re-scanning columns.
#[cfg(feature = "parallel")]
const ADDS_PER_COLUMN_SCAN: usize = 8;

/// Whether an `nnz`-nonzero, `ncols`-column product should spread over the
/// runtime's workers.
#[cfg(feature = "parallel")]
#[inline]
pub(super) fn should_parallelize(nnz: usize, ncols: usize) -> bool {
    nnz >= PARALLEL_MIN_CSC_NNZ && ncols > 0 && nnz / ncols >= PARALLEL_MIN_COLUMN_DEGREE
}

/// How many strips of contiguous rows the pass cuts `y` into: one per worker,
/// never more than the rows and never past the scan budget (see
/// [`ADDS_PER_COLUMN_SCAN`]).
#[cfg(feature = "parallel")]
fn strips(rows: usize, degree: usize) -> usize {
    let workers = std::thread::available_parallelism().map_or(1, core::num::NonZeroUsize::get);
    let scan_budget = (degree / ADDS_PER_COLUMN_SCAN).max(1);
    rows.min(workers.min(scan_budget)).max(1)
}

/// Compute `y = A · x` with consecutive strips of rows spread over moirai unit
/// tasks.
///
/// `y.len()` is `nrows` and `col_ptr.len()` is `ncols + 1`; both are checked by
/// the caller, which also zeroes `y` — every slot is owned by exactly one task,
/// so each accumulates onto a zero entry without touching its neighbours'.
#[cfg(feature = "parallel")]
pub(super) fn csc_spmv_rows_parallel<T: Scalar>(
    values: &[T],
    row_indices: &[usize],
    col_ptr: &[usize],
    xs: &[T],
    y: &mut [T],
) {
    let rows = y.len();
    let ncols = col_ptr.len().saturating_sub(1);
    if rows == 0 || ncols == 0 {
        return;
    }
    let strips = strips(rows, values.len() / ncols);
    // The runtime sizes a task from the bytes it is told a unit moves and its
    // own per-task budget, but this pass's per-task cost is the column scan
    // rather than the row bytes. So the strip count decides the width and the
    // byte figure is derived from it: one strip per task.
    let rows_per_strip = rows.div_ceil(strips);
    let row_bytes = (moirai::UNIT_TASK_BYTES / rows_per_strip).max(1);
    crate::infrastructure::parallel::for_each_unit_run_mut(y, row_bytes, |base, run| {
        for (column, window) in col_ptr.windows(2).enumerate() {
            scatter_column_window(
                values,
                row_indices,
                window[0],
                window[1],
                xs[column],
                base,
                run,
            );
        }
    });
}
