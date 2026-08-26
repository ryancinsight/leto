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

use criterion::{criterion_group, criterion_main, Criterion};
use leto::{Array2, Complex};
use std::hint::black_box;
use std::time::Duration;

const TRANSPOSE_TILE: usize = 32;
const SHAPES: [[usize; 2]; 4] = [[4_096, 16], [4_096, 64], [16_384, 16], [65_536, 4]];

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

criterion_group! {
    name = layout_copy;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_millis(500))
        .without_plots();
    targets = bench_layout_copy
}
criterion_main!(layout_copy);
