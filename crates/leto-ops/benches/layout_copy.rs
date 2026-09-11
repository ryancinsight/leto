//! Caller-owned transpose-copy comparison for Apollo FFT axis passes.
//!
//! The manual candidates reproduce Apollo's current 32-by-32 gather and
//! scatter loops. The Leto candidates use the existing logical assignment
//! contract over transposed views. Inputs and output allocations are shared
//! between candidates so address placement cannot confound the comparison.

#![expect(
    clippy::unwrap_used,
    reason = "benchmark setup treats a violated precondition as a failure"
)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use leto::{
    transpose_copy, transpose_copy_strided, Array2, ArrayView2, ArrayViewMut2, Complex, Layout,
};
use leto_ops::ComplexLayout;
use std::hint::black_box;
use std::time::Duration;

const TRANSPOSE_TILE: usize = 32;
const SHAPES: [[usize; 2]; 4] = [[4_096, 16], [4_096, 64], [16_384, 16], [65_536, 4]];
const COMPLEX_BATCHES: [(usize, usize, usize); 2] = [(1_024, 4, 4), (256, 16, 16)];

fn input(shape: [usize; 2]) -> Array2<Complex<f64>> {
    Array2::from_shape_fn(shape, |[row, column]| {
        let linear = row * shape[1] + column;
        Complex::new(linear as f64 * 0.25 + 1.0, linear as f64 * -0.125)
    })
}

fn tiled_gather<T: Copy>(source: &[T], target: &mut [T], rows: usize, columns: usize) {
    for column_tile in (0..columns).step_by(TRANSPOSE_TILE) {
        let column_end = (column_tile + TRANSPOSE_TILE).min(columns);
        for row_tile in (0..rows).step_by(TRANSPOSE_TILE) {
            let row_end = (row_tile + TRANSPOSE_TILE).min(rows);
            for column in column_tile..column_end {
                for row in row_tile..row_end {
                    target[column * rows + row] = source[row * columns + column];
                }
            }
        }
    }
}

fn tiled_scatter<T: Copy>(source: &[T], target: &mut [T], rows: usize, columns: usize) {
    for column_tile in (0..columns).step_by(TRANSPOSE_TILE) {
        let column_end = (column_tile + TRANSPOSE_TILE).min(columns);
        for row_tile in (0..rows).step_by(TRANSPOSE_TILE) {
            let row_end = (row_tile + TRANSPOSE_TILE).min(rows);
            for column in column_tile..column_end {
                for row in row_tile..row_end {
                    target[row * columns + column] = source[column * rows + row];
                }
            }
        }
    }
}

fn validate_gather(
    source: &Array2<Complex<f64>>,
    target: &mut Array2<Complex<f64>>,
    shape: [usize; 2],
) {
    let transposed = source.transpose([1, 0]).unwrap();
    target.view_mut().assign(&transposed);
    let leto_values = target.view().data().to_vec();

    let source_view = source.view();
    let source_values = source_view.as_slice_memory_order().unwrap();
    let mut target_view = target.view_mut();
    let target_values = target_view.as_mut_slice_memory_order().unwrap();
    tiled_gather(source_values, target_values, shape[0], shape[1]);
    assert_eq!(leto_values, target.view().data());
}

fn validate_scatter(
    source: &Array2<Complex<f64>>,
    target: &mut Array2<Complex<f64>>,
    shape: [usize; 2],
) {
    let transposed = source.transpose([1, 0]).unwrap();
    target.view_mut().assign(&transposed);
    let leto_values = target.view().data().to_vec();

    let source_view = source.view();
    let source_values = source_view.as_slice_memory_order().unwrap();
    let mut target_view = target.view_mut();
    let target_values = target_view.as_mut_slice_memory_order().unwrap();
    tiled_scatter(source_values, target_values, shape[0], shape[1]);
    assert_eq!(leto_values, target.view().data());
}

fn bench_layout_copy(c: &mut Criterion) {
    for shape in SHAPES {
        let [rows, columns] = shape;
        let source = input(shape);
        let transposed = source.transpose([1, 0]).unwrap();
        let mut target = Array2::zeros([columns, rows]);
        validate_gather(&source, &mut target, shape);

        let source_view = source.view();
        let source_values = source_view.as_slice_memory_order().unwrap();
        let mut gather = c.benchmark_group(format!("layout_copy/gather/{rows}x{columns}"));
        gather.bench_function("leto_assign", |bencher| {
            bencher.iter(|| {
                target.view_mut().assign(black_box(&transposed));
                black_box(target.view().data());
            });
        });
        gather.bench_function("apollo_tiled", |bencher| {
            bencher.iter(|| {
                let mut target_view = target.view_mut();
                let target_values = target_view.as_mut_slice_memory_order().unwrap();
                tiled_gather(
                    black_box(source_values),
                    black_box(target_values),
                    rows,
                    columns,
                );
                black_box(target_view.data());
            });
        });
        gather.finish();

        let scratch = input([columns, rows]);
        let scratch_transposed = scratch.transpose([1, 0]).unwrap();
        let mut output = Array2::zeros(shape);
        validate_scatter(&scratch, &mut output, shape);
        let scratch_view = scratch.view();
        let scratch_values = scratch_view.as_slice_memory_order().unwrap();
        let mut scatter = c.benchmark_group(format!("layout_copy/scatter/{rows}x{columns}"));
        scatter.bench_function("leto_assign", |bencher| {
            bencher.iter(|| {
                output.view_mut().assign(black_box(&scratch_transposed));
                black_box(output.view().data());
            });
        });
        scatter.bench_function("apollo_tiled", |bencher| {
            bencher.iter(|| {
                let mut output_view = output.view_mut();
                let output_values = output_view.as_mut_slice_memory_order().unwrap();
                tiled_scatter(
                    black_box(scratch_values),
                    black_box(output_values),
                    rows,
                    columns,
                );
                black_box(output_view.data());
            });
        });
        scatter.finish();
    }
}

fn expected_batch<T: Copy + Default>(
    source: &[Complex<T>],
    matrix_count: usize,
    rows: usize,
    columns: usize,
) -> Vec<Complex<T>> {
    let matrix_len = rows * columns;
    let mut output = vec![Complex::default(); source.len()];
    for matrix in 0..matrix_count {
        let base = matrix * matrix_len;
        for row in 0..rows {
            for column in 0..columns {
                output[base + column * rows + row] = source[base + row * columns + column];
            }
        }
    }
    output
}

fn generic_batch_assign<T: Copy>(
    source: &[T],
    destination: &mut [T],
    matrix_len: usize,
    source_layout: Layout<2>,
    destination_layout: Layout<2>,
) {
    for (source_matrix, destination_matrix) in source
        .chunks_exact(matrix_len)
        .zip(destination.chunks_exact_mut(matrix_len))
    {
        let source_view = ArrayView2::try_new(source_layout, source_matrix).unwrap();
        let mut destination_view =
            ArrayViewMut2::try_new(destination_layout, destination_matrix).unwrap();
        destination_view.assign(&source_view);
    }
}

fn bench_complex_scalar<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    scalar: &str,
    matrix_count: usize,
    rows: usize,
    columns: usize,
    value: impl Fn(usize) -> Complex<T>,
) where
    T: ComplexLayout + Default + PartialEq + core::fmt::Debug,
{
    let matrix_len = rows * columns;
    let len = matrix_count * matrix_len;
    let source = (0..len).map(value).collect::<Vec<_>>();
    let expected = expected_batch(&source, matrix_count, rows, columns);
    let source_layout = Layout::f_contiguous([columns, rows]).unwrap();
    let destination_layout = Layout::c_contiguous([columns, rows]).unwrap();
    let parameter = format!("{scalar}/{matrix_count}x{rows}x{columns}");

    let mut provider_output = vec![Complex::default(); len];
    T::transpose_complex_matrices(&source, &mut provider_output, matrix_count, rows, columns)
        .unwrap();
    assert_eq!(provider_output, expected);
    group.bench_with_input(BenchmarkId::new("provider", &parameter), &(), |b, ()| {
        b.iter(|| {
            T::transpose_complex_matrices(
                black_box(&source),
                black_box(&mut provider_output),
                matrix_count,
                rows,
                columns,
            )
            .unwrap();
            black_box(provider_output[len - 1])
        });
    });

    let mut generic_output = vec![Complex::default(); len];
    generic_batch_assign(
        &source,
        &mut generic_output,
        matrix_len,
        source_layout,
        destination_layout,
    );
    assert_eq!(generic_output, expected);
    group.bench_with_input(BenchmarkId::new("generic", &parameter), &(), |b, ()| {
        b.iter(|| {
            generic_batch_assign(
                black_box(&source),
                black_box(&mut generic_output),
                matrix_len,
                source_layout,
                destination_layout,
            );
            black_box(generic_output[len - 1])
        });
    });
}

fn bench_complex_batches(c: &mut Criterion) {
    let mut group = c.benchmark_group("layout_copy/complex_batch");
    for (matrix_count, rows, columns) in COMPLEX_BATCHES {
        bench_complex_scalar(&mut group, "f32", matrix_count, rows, columns, |index| {
            Complex::new(index as f32 + 0.25, -(index as f32) - 0.5)
        });
        bench_complex_scalar(&mut group, "f64", matrix_count, rows, columns, |index| {
            Complex::new(index as f64 + 0.25, -(index as f64) - 0.5)
        });
    }
    group.finish();
}

/// Apollo's 64³ axis-0 transpose: one `[64 x 4096]` `Complex64` matrix, moved
/// as 64 windows of 64 destination rows.
const WINDOW_ROWS: usize = 64;
const WINDOW_COLUMNS: usize = 4_096;
const WINDOW_WIDTH: usize = 64;
/// Eight complex elements move the source rows off the 64 KiB pitch by two
/// cache lines, so the rows of one tile no longer share L1 and L2 sets while
/// the bytes moved stay the same.
const PITCH_PAD: usize = 8;

/// Transposes the matrix window by window on the calling thread, each window
/// through the strided kernel at the given source pitch.
fn transpose_windows(source: &[Complex<f64>], pitch: usize, destination: &mut [Complex<f64>]) {
    for (window, block) in destination
        .chunks_exact_mut(WINDOW_WIDTH * WINDOW_ROWS)
        .enumerate()
    {
        let start = window * WINDOW_WIDTH;
        let span = (WINDOW_ROWS - 1) * pitch + WINDOW_WIDTH;
        transpose_copy_strided(
            &source[start..start + span],
            pitch,
            block,
            WINDOW_ROWS,
            WINDOW_WIDTH,
        )
        .unwrap();
    }
}

/// The same bytes at the same shape as `WINDOW_*`, at a pitch of the column
/// count and at that pitch padded, on one thread and in the provider's tasks;
/// beside them the batch of 64 `[64 x 64]` matrices apollo's axis 1 moves,
/// which the provider spreads the same way and whose rows sit 1 KiB apart.
fn bench_window_pitch(c: &mut Criterion) {
    let mut group = c.benchmark_group("layout_copy/window_pitch");
    let len = WINDOW_ROWS * WINDOW_COLUMNS;
    let value = |index: usize| Complex::new(index as f64 + 0.25, -(index as f64) - 0.5);
    let dense = (0..len).map(value).collect::<Vec<_>>();
    let expected = expected_batch(&dense, 1, WINDOW_ROWS, WINDOW_COLUMNS);
    let padded_pitch = WINDOW_COLUMNS + PITCH_PAD;
    let padded = (0..WINDOW_ROWS * padded_pitch)
        .map(|index| {
            let (row, column) = (index / padded_pitch, index % padded_pitch);
            if column < WINDOW_COLUMNS {
                value(row * WINDOW_COLUMNS + column)
            } else {
                Complex::default()
            }
        })
        .collect::<Vec<_>>();
    let mut output = vec![Complex::default(); len];

    for (label, source, pitch) in [
        ("serial/pitch=4096", &dense, WINDOW_COLUMNS),
        ("serial/pitch=4104", &padded, padded_pitch),
    ] {
        transpose_windows(source, pitch, &mut output);
        assert_eq!(output, expected);
        group.bench_function(label, |b| {
            b.iter(|| {
                transpose_windows(black_box(source), pitch, black_box(&mut output));
                black_box(output[len - 1])
            });
        });
    }

    <f64 as ComplexLayout>::transpose_complex_matrices(
        &dense,
        &mut output,
        1,
        WINDOW_ROWS,
        WINDOW_COLUMNS,
    )
    .unwrap();
    assert_eq!(output, expected);
    group.bench_function("tasks/pitch=4096", |b| {
        b.iter(|| {
            <f64 as ComplexLayout>::transpose_complex_matrices(
                black_box(&dense),
                black_box(&mut output),
                1,
                WINDOW_ROWS,
                WINDOW_COLUMNS,
            )
            .unwrap();
            black_box(output[len - 1])
        });
    });

    let batch_expected = expected_batch(&dense, WINDOW_ROWS, WINDOW_ROWS, WINDOW_ROWS);
    <f64 as ComplexLayout>::transpose_complex_matrices(
        &dense,
        &mut output,
        WINDOW_ROWS,
        WINDOW_ROWS,
        WINDOW_ROWS,
    )
    .unwrap();
    assert_eq!(output, batch_expected);
    group.bench_function("tasks/batch=64x64x64", |b| {
        b.iter(|| {
            <f64 as ComplexLayout>::transpose_complex_matrices(
                black_box(&dense),
                black_box(&mut output),
                WINDOW_ROWS,
                WINDOW_ROWS,
                WINDOW_ROWS,
            )
            .unwrap();
            black_box(output[len - 1])
        });
    });
    group.finish();
}

/// The two move geometries a 3-D transform makes on a 64³ `Complex64` volume:
/// the C-order chain transposes `[64, 4096]` three times, and the pair that
/// keeps the rotated order transposes `[64, 4096]` twice going out and
/// `[4096, 64]` twice coming back. Four moves replaced six and bought four
/// percent, so the two geometries are timed here against each other, through
/// the provider (tasks) and through Leto's serial kernel.
fn bench_transpose_geometry(c: &mut Criterion) {
    const SIDE: usize = 64;
    let plane = SIDE * SIDE;
    let len = SIDE * plane;
    let source = (0..len)
        .map(|index| Complex::new(index as f64 + 0.25, -(index as f64) - 0.5))
        .collect::<Vec<_>>();
    let mut destination = vec![Complex::<f64>::default(); len];
    let mut group = c.benchmark_group("layout_copy/transpose_geometry");

    // The same permutation as the wide move, expressed as `SIDE` independent
    // `[SIDE, SIDE]` transposes of a strided window rather than one
    // `[SIDE, SIDE * SIDE]` matrix: `(x, y, z)` to `(y, z, x)` is, for each
    // `y`, the `[nx, nz]` window at stride `ny * nz` laid down contiguously.
    let expected_rotation = {
        let mut rotated = vec![Complex::<f64>::default(); len];
        for y in 0..SIDE {
            for z in 0..SIDE {
                for x in 0..SIDE {
                    rotated[(y * SIDE + z) * SIDE + x] = source[(x * SIDE + y) * SIDE + z];
                }
            }
        }
        rotated
    };
    let rotate_by_window = |destination: &mut [Complex<f64>]| {
        for y in 0..SIDE {
            let window = &source[y * SIDE..];
            let block = &mut destination[y * plane..(y + 1) * plane];
            transpose_copy_strided(window, plane, block, SIDE, SIDE).unwrap();
        }
    };
    rotate_by_window(&mut destination);
    assert_eq!(destination, expected_rotation, "windowed rotation");
    group.bench_function("serial/windowed/64x(64x64)", |b| {
        b.iter(|| {
            rotate_by_window(black_box(&mut destination));
            black_box(destination[len - 1])
        });
    });

    for (label, rows, columns) in [("wide/64x4096", SIDE, plane), ("tall/4096x64", plane, SIDE)] {
        let expected = expected_batch(&source, 1, rows, columns);
        <f64 as ComplexLayout>::transpose_complex_matrices(
            &source,
            &mut destination,
            1,
            rows,
            columns,
        )
        .unwrap();
        assert_eq!(destination, expected, "{label} provider");
        group.bench_function(format!("provider/{label}"), |b| {
            b.iter(|| {
                <f64 as ComplexLayout>::transpose_complex_matrices(
                    black_box(&source),
                    black_box(&mut destination),
                    1,
                    rows,
                    columns,
                )
                .unwrap();
                black_box(destination[len - 1])
            });
        });

        transpose_copy(&source, &mut destination, rows, columns).unwrap();
        assert_eq!(destination, expected, "{label} serial");
        group.bench_function(format!("serial/{label}"), |b| {
            b.iter(|| {
                transpose_copy(
                    black_box(&source),
                    black_box(&mut destination),
                    rows,
                    columns,
                )
                .unwrap();
                black_box(destination[len - 1])
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = layout_copy;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_millis(500))
        .without_plots();
    targets = bench_layout_copy, bench_complex_batches, bench_window_pitch, bench_transpose_geometry
}
criterion_main!(layout_copy);
