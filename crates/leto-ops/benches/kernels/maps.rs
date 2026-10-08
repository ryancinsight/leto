//! Unary map, tile geometry, reductions and zip throughput.

use super::*;

fn bench_unary_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("unary_map");
    let len = 1usize << 16;
    let input = Array::from_shape_vec([len], pinned_values(len, 1.0)).unwrap();
    group.bench_function("map_into_contiguous_64k", |bencher| {
        bencher.iter_batched(
            || Array::zeros([len]),
            |mut out| {
                map_into(black_box(&input.view()), &mut out.view_mut(), |value| {
                    value + 0.5
                })
                .unwrap();
                out
            },
            BatchSize::LargeInput,
        );
    });

    // Typed scalar-add into caller-owned output. Bandwidth-bound, so the
    // intensity-aware gate must keep a 64k  fill (1 MB working set) serial
    // rather than parallelizing it into a slowdown (cf. the raw  above,
    // which stays eager as a compute-bound default).
    group.bench_function("scalar_add_into_64k", |bencher| {
        bencher.iter_batched(
            || Array::zeros([len]),
            |mut out| {
                scalar_map_into::<AddOp, f64, 1>(
                    black_box(&input.view()),
                    0.5,
                    &mut out.view_mut(),
                )
                .unwrap();
                out
            },
            BatchSize::LargeInput,
        );
    });

    let n = 256usize;
    let square = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0)).unwrap();
    let transposed = square.transpose([1, 0]).unwrap();
    group.bench_function("map_into_transposed_256x256", |bencher| {
        bencher.iter_batched(
            || Array::zeros([n, n]),
            |mut out| {
                map_into(black_box(&transposed), &mut out.view_mut(), |value| {
                    value + 0.5
                })
                .unwrap();
                out
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_runtime_tile_geometry(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime_tile_geometry");
    let n = 256usize;
    let input = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0)).unwrap();
    let transposed = input.transpose([1, 0]).unwrap();
    let narrow = CacheGeometry::with_cache_line_bytes(64).expect("64-byte policy is valid");
    let wide = CacheGeometry::with_cache_line_bytes(128).expect("128-byte policy is valid");

    for (name, geometry) in [("line64", narrow), ("line128", wide)] {
        group.bench_function(name, |bencher| {
            bencher.iter_batched(
                || Array::zeros([n, n]),
                |mut output| {
                    map_into_with_cache_geometry(
                        black_box(&transposed),
                        &mut output.view_mut(),
                        |value| value + 0.5,
                        geometry,
                    )
                    .unwrap();
                    output
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions");
    let len = 1usize << 16;
    let a = Array::from_shape_vec([len], pinned_values(len, 1.0)).unwrap();
    group.bench_function("sum_64k", |bencher| {
        bencher.iter(|| sum(black_box(&a.view())));
    });
    group.bench_function("norm_l2_64k", |bencher| {
        bencher.iter(|| norm_l2(black_box(&a.view())).unwrap());
    });
    group.bench_function("norm_l1_64k", |bencher| {
        bencher.iter(|| norm_l1(black_box(&a.view())).unwrap());
    });
    group.bench_function("norm_max_64k", |bencher| {
        bencher.iter(|| norm_max(black_box(&a.view())).unwrap());
    });
    // Scalar-fold reference series: the exact pre-0.17.0 dense-path body for
    // norm_l1/norm_max, kept as the in-run before-number for the hermes
    // abs-reduction routing.
    group.bench_function("norm_l1_64k_scalar_ref", |bencher| {
        let data = a.view();
        bencher.iter(|| {
            let slice = black_box(data.as_slice_memory_order().unwrap());
            slice.iter().fold(0.0f64, |acc, &x| acc + x.abs())
        });
    });
    group.bench_function("norm_max_64k_scalar_ref", |bencher| {
        let data = a.view();
        bencher.iter(|| {
            let slice = black_box(data.as_slice_memory_order().unwrap());
            slice
                .iter()
                .fold(0.0f64, |acc, &x| if x.abs() > acc { x.abs() } else { acc })
        });
    });

    let n = 256usize;
    let square = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0)).unwrap();
    group.bench_function("sum_contiguous_256x256", |bencher| {
        bencher.iter(|| sum(black_box(&square.view())));
    });
    let transposed = square.transpose([1, 0]).unwrap();
    group.bench_function("sum_transposed_256x256", |bencher| {
        bencher.iter(|| sum(black_box(&transposed)));
    });
    group.bench_function("norm_l2_transposed_256x256", |bencher| {
        bencher.iter(|| norm_l2(black_box(&transposed)).unwrap());
    });

    let reversed = square
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();
    group.bench_function("sum_reverse_last_axis_256x256", |bencher| {
        bencher.iter(|| sum(black_box(&reversed)));
    });

    let strided_storage =
        Array::from_shape_vec([n * 2, n * 2], pinned_values(n * n * 4, 1.0)).unwrap();
    let strided = strided_storage
        .view()
        .slice_with::<2>(&[
            SliceArg::range(None, None, 2),
            SliceArg::range(None, None, 2),
        ])
        .unwrap();
    group.bench_function("sum_strided_step2_256x256", |bencher| {
        bencher.iter(|| sum(black_box(&strided)));
    });
    group.bench_function("norm_l2_reverse_last_axis_256x256", |bencher| {
        bencher.iter(|| norm_l2(black_box(&reversed)).unwrap());
    });
    group.finish();
}

fn bench_zip(c: &mut Criterion) {
    let mut group = c.benchmark_group("zip");
    let n = 256usize;
    let src = Array::from_shape_vec([n, n], pinned_values(n * n, 0.5)).unwrap();
    group.bench_function("zip_mut_with_transposed_256x256", |bencher| {
        bencher.iter_batched(
            || Array::from_shape_vec([n, n], pinned_values(n * n, 1.0)).unwrap(),
            |mut out| {
                let transposed = src.transpose([1, 0]).unwrap();
                zip_mut_with(out.view_mut(), &transposed, |o, &s| *o += s).unwrap();
                out
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}
