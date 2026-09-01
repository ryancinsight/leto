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
use eunomia::Pod;
use hermes_simd::LaneScalar;
use leto::{Array2, ArrayView2, ArrayViewMut2, Complex, Layout};
use leto_ops::transpose_complex_matrices;
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
    T: LaneScalar + Pod + Default + PartialEq + core::fmt::Debug,
{
    let matrix_len = rows * columns;
    let len = matrix_count * matrix_len;
    let source = (0..len).map(value).collect::<Vec<_>>();
    let expected = expected_batch(&source, matrix_count, rows, columns);
    let source_layout = Layout::f_contiguous([columns, rows]).unwrap();
    let destination_layout = Layout::c_contiguous([columns, rows]).unwrap();
    let parameter = format!("{scalar}/{matrix_count}x{rows}x{columns}");

    let mut provider_output = vec![Complex::default(); len];
    transpose_complex_matrices(&source, &mut provider_output, matrix_count, rows, columns).unwrap();
    assert_eq!(provider_output, expected);
    group.bench_with_input(BenchmarkId::new("provider", &parameter), &(), |b, ()| {
        b.iter(|| {
            transpose_complex_matrices(
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

criterion_group! {
    name = layout_copy;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_millis(500))
        .without_plots();
    targets = bench_layout_copy, bench_complex_batches
}
criterion_main!(layout_copy);
