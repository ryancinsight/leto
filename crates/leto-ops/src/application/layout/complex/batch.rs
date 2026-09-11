//! Allocation-free layout movement for homogeneous complex matrix batches.

use super::tile::{load_tile, store_tile};
use eunomia::Pod;
use hermes_simd::{
    vectorize_hardware_lanes, ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel,
};
#[cfg(feature = "parallel")]
use leto::transpose_copy_strided;
use leto::{transpose_copy, Complex, LetoError, Result};

/// Pinned Apollo phase measurements show that dispatch amortizes at this batch
/// count while smaller batches remain with Leto's generic tiled transpose.
const REGISTER_TRANSPOSE_MIN_MATRICES: usize = 256;
/// The measured register-tile regime ends before matrices become cache-copy
/// dominated; larger matrices retain Leto's cache-budgeted generic kernel.
const REGISTER_TRANSPOSE_MAX_MATRIX_SIDE: usize = 16;

/// Bytes of destination rows one scheduled task writes when a batch runs in
/// parallel.
///
/// Apollo's 3-D pass probe (`dimension_3d::pass_attribution`, 2026-09-09)
/// timed the axis-1 transpose pair of a 64³ `Complex64` volume — 64 matrices
/// of 64 KiB — at 356 µs serial against 147 µs with one matrix per task, and
/// 345 against 89 at the fastest samples. 64 KiB is one matrix there, sits
/// inside one core's L2, and is the task width its lane passes settled on.
/// Tasks are cut in destination rows rather than matrices so that the same
/// width splits the volume's axis-0 transpose — one `[64 x 4096]` matrix,
/// 4 MiB — into 64 tasks where whole-matrix tasks left it on one thread.
#[cfg(feature = "parallel")]
const PARALLEL_TRANSPOSE_TASK_BYTES: usize = 64 * 1024;

/// Bytes of batch below which the transpose stays on the calling thread.
///
/// The same probe's 32³ volume — 32 matrices of 16 KiB, 512 KiB in all — ran
/// *slower* spread over tasks (28.5 µs against 21.9 serial): at eleven
/// microseconds of work the runtime's spawn-and-join is the larger part. The
/// 64³ batch, 4 MiB, won by 2.4x to 3.9x. One mebibyte sits between the two;
/// the probe is the instrument that moves it.
#[cfg(feature = "parallel")]
const PARALLEL_TRANSPOSE_MIN_BYTES: usize = 1024 * 1024;

pub(super) fn transpose_complex_matrices<T>(
    source: &[Complex<T>],
    destination: &mut [Complex<T>],
    matrix_count: usize,
    rows: usize,
    columns: usize,
) -> Result<()>
where
    T: LaneScalar + Pod,
{
    #[expect(
        clippy::unnecessary_lazy_evaluations,
        reason = "Avoid eager LetoError drop on successful arithmetic; ADR 0027"
    )]
    let matrix_len = rows
        .checked_mul(columns)
        .ok_or_else(|| LetoError::Overflow {
            reason: "complex matrix element count",
        })?;
    #[expect(
        clippy::unnecessary_lazy_evaluations,
        reason = "Avoid eager LetoError drop on successful arithmetic; ADR 0027"
    )]
    let total_len = matrix_count
        .checked_mul(matrix_len)
        .ok_or_else(|| LetoError::Overflow {
            reason: "complex matrix batch element count",
        })?;
    validate_length("source", source.len(), total_len)?;
    validate_length("destination", destination.len(), total_len)?;

    if total_len == 0 {
        return Ok(());
    }

    if uses_register_complex_tiles(matrix_count, rows, columns)
        && transpose_hardware(source, destination, matrix_len, rows, columns)
    {
        return Ok(());
    }

    #[cfg(feature = "parallel")]
    if parallel_transpose_applies::<Complex<T>>(total_len) {
        transpose_in_tasks(source, destination, matrix_len, rows, columns);
        return Ok(());
    }
    for (source_matrix, destination_matrix) in source
        .chunks_exact(matrix_len)
        .zip(destination.chunks_exact_mut(matrix_len))
    {
        transpose_copy(source_matrix, destination_matrix, rows, columns)?;
    }
    Ok(())
}

/// Whether a validated batch of `total_len` elements is worth spreading over
/// the runtime's workers.
#[cfg(feature = "parallel")]
fn parallel_transpose_applies<E>(total_len: usize) -> bool {
    total_len.saturating_mul(core::mem::size_of::<E>()) >= PARALLEL_TRANSPOSE_MIN_BYTES
}

/// Transposes a validated batch over tasks of at least
/// [`PARALLEL_TRANSPOSE_TASK_BYTES`] of destination rows.
///
/// A destination row of the batch — one column of one source matrix — is a
/// contiguous run of `rows` elements, and the batch is `matrix_count *
/// columns` of them in order, so a task is a run of whole destination rows
/// and no two tasks write one element. A run that crosses a matrix boundary
/// transposes one matrix window at a time, and a batch of one large matrix
/// splits like any other. The lengths were validated by the caller, which is
/// what lets the per-window result be an invariant here rather than an error
/// to thread out of the closure.
#[cfg(feature = "parallel")]
fn transpose_in_tasks<T>(
    source: &[Complex<T>],
    destination: &mut [Complex<T>],
    matrix_len: usize,
    rows: usize,
    columns: usize,
) where
    T: LaneScalar + Pod,
{
    let element_bytes = core::mem::size_of::<Complex<T>>().max(1);
    let row_bytes = rows.saturating_mul(element_bytes).max(1);
    // Each destination row reads one source column. A byte budget alone cuts
    // tall matrices into one-column tasks, making separate workers fetch the
    // same source lines. Keep one line's worth of columns together: the 64³
    // Apollo attribution measures four rows/task at 31–33 us versus 42–44 us
    // for one, while raising the byte budget globally regresses wide moves.
    // Unaligned origins and ragged pitches can still share boundary lines.
    // Clipping to columns also bounds rows * source_columns by matrix_len.
    let source_columns = crate::infrastructure::cache::cached_cache_geometry()
        .cache_line_bytes()
        .div_ceil(element_bytes)
        .min(columns);
    let rows_per_task = (PARALLEL_TRANSPOSE_TASK_BYTES / row_bytes).max(source_columns);
    let task_len = rows_per_task * rows;
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Parallel, _, _>(
        destination,
        task_len,
        |index, task| {
            let mut destination_row = index * rows_per_task;
            let mut remaining = task;
            while !remaining.is_empty() {
                let matrix = destination_row / columns;
                let column = destination_row % columns;
                let width = (columns - column).min(remaining.len() / rows);
                let (window, rest) = core::mem::take(&mut remaining).split_at_mut(width * rows);
                let start = matrix * matrix_len + column;
                let span = (rows - 1) * columns + width;
                transpose_copy_strided(&source[start..start + span], columns, window, rows, width)
                    .expect("invariant: each window is whole destination rows of one matrix, validated above");
                destination_row += width;
                remaining = rest;
            }
        },
    );
}

#[inline]
fn validate_length(role: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        return Ok(());
    }
    Err(LetoError::StorageError {
        reason: format!(
            "complex matrix transpose {role} length {actual} does not match expected {expected}"
        ),
    })
}

#[inline]
const fn uses_register_complex_tiles(matrix_count: usize, rows: usize, columns: usize) -> bool {
    matrix_count >= REGISTER_TRANSPOSE_MIN_MATRICES
        && rows <= REGISTER_TRANSPOSE_MAX_MATRIX_SIDE
        && columns <= REGISTER_TRANSPOSE_MAX_MATRIX_SIDE
}

#[inline]
fn transpose_hardware<T>(
    source: &[Complex<T>],
    destination: &mut [Complex<T>],
    matrix_len: usize,
    rows: usize,
    columns: usize,
) -> bool
where
    T: LaneScalar + Pod,
{
    let minimum_side = rows.min(columns);
    if minimum_side >= 8
        && vectorize_hardware_lanes::<16, T, _>(ComplexTransposeKernel {
            source,
            destination: &mut *destination,
            matrix_len,
            rows,
            columns,
        })
        .is_some()
    {
        return true;
    }
    if minimum_side >= 4
        && vectorize_hardware_lanes::<8, T, _>(ComplexTransposeKernel {
            source,
            destination: &mut *destination,
            matrix_len,
            rows,
            columns,
        })
        .is_some()
    {
        return true;
    }
    minimum_side >= 2
        && vectorize_hardware_lanes::<4, T, _>(ComplexTransposeKernel {
            source,
            destination,
            matrix_len,
            rows,
            columns,
        })
        .is_some()
}

struct ComplexTransposeKernel<'a, T> {
    source: &'a [Complex<T>],
    destination: &'a mut [Complex<T>],
    matrix_len: usize,
    rows: usize,
    columns: usize,
}

impl<T> LaneKernel<T> for ComplexTransposeKernel<'_, T>
where
    T: LaneScalar + Pod,
{
    type Output = ();

    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        // Only the register side specializes storage; matrix dimensions stay
        // runtime values so shapes do not multiply the emitted kernels.
        match ComplexReg::<T, A>::COMPLEX_COUNT {
            2 => transpose_tiled::<T, A, 2>(self, &simd),
            4 => transpose_tiled::<T, A, 4>(self, &simd),
            8 => transpose_tiled::<T, A, 8>(self, &simd),
            _ => {
                unreachable!("invariant: requested hardware widths hold 2, 4 or 8 complex samples")
            }
        }
    }
}

#[inline(always)]
fn transpose_tiled<T, A, const SIDE: usize>(batch: ComplexTransposeKernel<'_, T>, simd: &Simd<T, A>)
where
    T: LaneScalar + Pod,
    A: SimdArch + SimdKernel<T>,
{
    let ComplexTransposeKernel {
        source,
        destination,
        matrix_len,
        rows,
        columns,
    } = batch;
    let full_rows = rows / SIDE * SIDE;
    let full_columns = columns / SIDE * SIDE;

    for (source_matrix, destination_matrix) in source
        .chunks_exact(matrix_len)
        .zip(destination.chunks_exact_mut(matrix_len))
    {
        for tile_row in (0..full_rows).step_by(SIDE) {
            for tile_column in (0..full_columns).step_by(SIDE) {
                let tile = load_tile::<T, A, SIDE>(
                    simd,
                    source_matrix,
                    columns,
                    tile_row * columns + tile_column,
                );
                store_tile(
                    simd,
                    destination_matrix,
                    rows,
                    tile_column * rows + tile_row,
                    &tile,
                );
            }
        }

        for row in 0..rows {
            let first_tail_column = if row < full_rows { full_columns } else { 0 };
            for column in first_tail_column..columns {
                destination_matrix[column * rows + row] = source_matrix[row * columns + column];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::uses_register_complex_tiles;

    #[test]
    fn selects_only_the_measured_small_matrix_regime() {
        assert!(uses_register_complex_tiles(256, 16, 16));
        assert!(!uses_register_complex_tiles(255, 16, 16));
        assert!(!uses_register_complex_tiles(256, 17, 16));
        assert!(!uses_register_complex_tiles(256, 16, 17));
    }
}
