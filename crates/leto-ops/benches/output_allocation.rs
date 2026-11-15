//! Cost of the output allocation in the allocating kernels.
//!
//! Each allocating kernel (`sum_axis`, `topk_axis`, `embedding_gather`,
//! `triangular`, `ray_line_integrals`) builds its output storage and then
//! overwrites every element. The zero-filled allocation pays one write pass
//! over the output before that overwrite; this bench measures the whole
//! allocating call, so a regression against a revision that skipped the
//! fill shows as a ratio of the call, not of the fill alone.

#![expect(
    clippy::unwrap_used,
    reason = "benchmark setup treats a violated precondition as a failure"
)]

use criterion::{criterion_group, criterion_main, Criterion};
use leto::{Array1, Array2, Array3};
use leto_ops::{
    embedding_gather, ray_line_integrals, sum_axis, topk_axis, triangular, TriangularMode,
};
use std::hint::black_box;
use std::time::Duration;

/// Strictly positive, deterministic, non-constant element values.
fn value(index: usize) -> f32 {
    f32::from(u8::try_from(index % 251).unwrap()) + 1.0
}

fn matrix(shape: [usize; 2]) -> Array2<f32> {
    let columns = shape[1];
    Array2::from_shape_fn(shape, |[row, column]| value(row * columns + column))
}

fn bench_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("output_allocation/sum_axis");
    for shape in [[262_144, 4], [1_024, 1_024]] {
        let input = matrix(shape);
        let view = input.view();
        group.bench_function(format!("{}x{}", shape[0], shape[1]), |bencher| {
            bencher.iter(|| black_box(sum_axis(black_box(&view), 1).unwrap()));
        });
    }
    group.finish();
}

fn bench_topk(c: &mut Criterion) {
    let input = matrix([4_096, 256]);
    let view = input.view();
    c.bench_function("output_allocation/topk_axis/4096x256_k4", |bencher| {
        bencher.iter(|| black_box(topk_axis(black_box(&view), 1, 4).unwrap()));
    });
}

fn bench_embedding(c: &mut Criterion) {
    let table = matrix([16_384, 64]);
    let indices = Array1::from_shape_fn([16_384], |[row]| {
        u32::try_from((row * 7_919) % 16_384).unwrap()
    });
    let (table_view, index_view) = (table.view(), indices.view());
    c.bench_function("output_allocation/embedding_gather/16384x64", |bencher| {
        bencher.iter(|| {
            black_box(embedding_gather(black_box(&table_view), black_box(&index_view)).unwrap())
        });
    });
}

fn bench_triangular(c: &mut Criterion) {
    let input = matrix([1_024, 1_024]);
    let view = input.view();
    c.bench_function("output_allocation/triangular/1024x1024", |bencher| {
        bencher.iter(|| black_box(triangular(black_box(&view), TriangularMode::Lower, 0).unwrap()));
    });
}

fn bench_ray_integrals(c: &mut Criterion) {
    let field = Array3::from_shape_fn([64, 64, 64], |[x, y, z]| value(x + 64 * (y + 64 * z)));
    let rays = Array2::from_shape_fn([16_384, 6], |[ray, component]| match component {
        0 => -10.0,
        1 => f32::from(u8::try_from(ray % 64).unwrap()),
        2 => f32::from(u8::try_from(ray / 64 % 64).unwrap()),
        3 => 1.0,
        _ => 0.0,
    });
    let (field_view, ray_view) = (field.view(), rays.view());
    c.bench_function("output_allocation/ray_line_integrals/16384", |bencher| {
        bencher.iter(|| {
            black_box(
                ray_line_integrals(
                    black_box(&field_view),
                    [0.0; 3],
                    [1.0; 3],
                    black_box(&ray_view),
                    1.0,
                )
                .unwrap(),
            )
        });
    });
}

criterion_group! {
    name = output_allocation;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_reduction, bench_topk, bench_embedding, bench_triangular, bench_ray_integrals
}
criterion_main!(output_allocation);
