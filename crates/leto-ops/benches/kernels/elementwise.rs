//! Elementwise map throughput and parallel crossover.

use super::*;

fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_add");
    let len = 1usize << 16;
    let a = Array::from_shape_vec([len], pinned_values(len, 1.0)).unwrap();
    let b = Array::from_shape_vec([len], pinned_values(len, 0.5)).unwrap();
    group.bench_function("contiguous_64k", |bencher| {
        bencher.iter_batched(
            || Array::zeros([len]),
            |mut out| {
                leto_ops::binary_map::<AddOp, f64, 1>(
                    black_box(&a.view()),
                    black_box(&b.view()),
                    &mut out.view_mut(),
                )
                .unwrap();
                out
            },
            BatchSize::LargeInput,
        );
    });

    let n = 256usize;
    let sq_a = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0)).unwrap();
    let sq_b = Array::from_shape_vec([n, n], pinned_values(n * n, 0.5)).unwrap();
    group.bench_function("transposed_256x256", |bencher| {
        bencher.iter_batched(
            || Array::zeros([n, n]),
            |mut out| {
                let at = sq_a.transpose([1, 0]).unwrap();
                leto_ops::binary_map::<AddOp, f64, 2>(
                    black_box(&at),
                    black_box(&sq_b.view()),
                    &mut out.view_mut(),
                )
                .unwrap();
                out
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("contiguous_256x256", |bencher| {
        bencher.iter_batched(
            || Array::zeros([n, n]),
            |mut out| {
                leto_ops::binary_map::<AddOp, f64, 2>(
                    black_box(&sq_a.view()),
                    black_box(&sq_b.view()),
                    &mut out.view_mut(),
                )
                .unwrap();
                out
            },
            BatchSize::LargeInput,
        );
    });

    let strided_lhs_storage =
        Array::from_shape_vec([n * 2, n * 2], pinned_values(n * n * 4, 1.0)).unwrap();
    let strided_lhs = strided_lhs_storage
        .view()
        .slice_with::<2>(&[
            SliceArg::range(None, None, 2),
            SliceArg::range(None, None, 2),
        ])
        .unwrap();
    group.bench_function("strided_step2_lhs_256x256", |bencher| {
        bencher.iter_batched(
            || Array::zeros([n, n]),
            |mut out| {
                leto_ops::binary_map::<AddOp, f64, 2>(
                    black_box(&strided_lhs),
                    black_box(&sq_b.view()),
                    &mut out.view_mut(),
                )
                .unwrap();
                out
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_parallel_crossover(c: &mut Criterion) {
    // Sweep bandwidth-bound `add` across the LLC-residency crossover to
    // validate the working-set-vs-L3 parallel gate, instantiated at every
    // scalar the gate serves: the gate scales the working set by
    // `size_of::<T>()`, so a sweep at one scalar validates one row of that
    // model. Working set = 3·N·size_of::<T>(); on a 36 MiB L3 the crossover
    // sits near N ≈ 1.5M for f64 and N ≈ 3M for f32. Run under default
    // features for the gate's decision, `--no-default-features` for an
    // all-serial baseline, and with the gate temporarily forced parallel to
    // locate the true crossover during threshold calibration.
    //
    // A same-binary three-arm probe (gate / sequential slice loop / naive
    // scoped-thread add, 2026-09-01) put the naive-parallel break-even at
    // 1.57M for f64 and between 2.1M and 3.1M for f32 — the byte model, not
    // an element count; the in-place unary crossover near 1M elements for
    // both scalars does not transfer to this three-stream path.
    let mut group = c.benchmark_group("parallel_crossover");
    crossover_rows::<f64>(&mut group, "f64");
    crossover_rows::<f32>(&mut group, "f32");
    group.finish();
}

fn crossover_rows<T: leto_ops::Scalar + eunomia::FloatElement>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    scalar: &str,
) {
    // Exact in every scalar the gate serves, unlike a fractional factor, so the
    // f32 and f64 rows add identical values and differ only in width.
    let pinned = |len: usize, offset: usize| -> Vec<T> {
        (0..len).map(|i| T::from_count(i % 997 + offset)).collect()
    };
    for &n in &[
        524_288usize,
        1_048_576,
        1_572_864,
        2_097_152,
        3_145_728,
        4_194_304,
        8_388_608,
    ] {
        let a = Array::from_shape_vec([n], pinned(n, 1)).unwrap();
        let b = Array::from_shape_vec([n], pinned(n, 3)).unwrap();
        group.bench_function(format!("add/{scalar}/{}k", n / 1024), |bencher| {
            bencher.iter_batched(
                || Array::<T, _, 1>::zeros([n]),
                |mut out| {
                    leto_ops::binary_map::<AddOp, T, 1>(
                        black_box(&a.view()),
                        black_box(&b.view()),
                        &mut out.view_mut(),
                    )
                    .unwrap();
                    out
                },
                BatchSize::LargeInput,
            );
        });
    }
}
