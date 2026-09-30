//! Elementwise, unary, reduction, zip and oracle bench families.

use super::pinned_values;
use criterion::{BatchSize, Criterion};
use leto::{Array, SliceArg};
use leto_ops::{
    add, dot, map_into, map_into_with_cache_geometry, matmul, norm_l1, norm_l2, norm_max,
    scalar_map_into, sum, unary_map, zip_mut_with, AddOp, CacheGeometry, ExpOp,
};
use std::hint::black_box;

pub(super) fn bench_elementwise(c: &mut Criterion) {
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

pub(super) fn bench_parallel_crossover(c: &mut Criterion) {
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

fn crossover_rows<T: leto_ops::Scalar>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    scalar: &str,
) {
    // Exact in every scalar the gate serves, unlike a fractional factor, so the
    // f32 and f64 rows add identical values and differ only in width.
    let pinned = |len: usize, offset: usize| -> Vec<T> {
        (0..len).map(|i| T::from_usize(i % 997 + offset)).collect()
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

pub(super) fn bench_unary_map(c: &mut Criterion) {
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

pub(super) fn bench_runtime_tile_geometry(c: &mut Criterion) {
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

pub(super) fn bench_reductions(c: &mut Criterion) {
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

pub(super) fn bench_zip(c: &mut Criterion) {
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

/// Leto-only matmul and reverse-reduction baselines (removed external comparison).
pub(super) fn bench_oracle_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("oracle_compare");

    for &n in &[64usize, 128, 256] {
        let lhs_values = pinned_values(n * n, 1.0e-3);
        let rhs_values = pinned_values(n * n, 2.0e-3);
        let leto_lhs = Array::from_shape_vec([n, n], lhs_values).unwrap();
        let leto_rhs = Array::from_shape_vec([n, n], rhs_values).unwrap();

        group.bench_function(format!("matmul_leto_{n}x{n}"), |bencher| {
            bencher.iter_batched(
                || Array::zeros([n, n]),
                |mut out| {
                    matmul(
                        black_box(&leto_lhs.view()),
                        black_box(&leto_rhs.view()),
                        &mut out.view_mut(),
                    )
                    .unwrap();
                    out
                },
                BatchSize::LargeInput,
            );
        });
    }

    let reduce_n = 256usize;
    let reduce_values = pinned_values(reduce_n * reduce_n, 1.0);
    let leto_reduce = Array::from_shape_vec([reduce_n, reduce_n], reduce_values).unwrap();
    let leto_reversed = leto_reduce
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();

    group.bench_function("sum_reverse_leto_256x256", |bencher| {
        bencher.iter(|| sum(black_box(&leto_reversed)));
    });
    group.bench_function("norm_l2_reverse_leto_256x256", |bencher| {
        bencher.iter(|| norm_l2(black_box(&leto_reversed)).unwrap());
    });
    group.finish();
}

/// Leto-only parity baselines across the elementwise, unary, reduction, and
/// vector-dot families (removed external comparison). Same pinned f64 inputs feed
/// the leto side; criterion reports median + CI per side.
pub(super) fn bench_parity_oracle(c: &mut Criterion) {
    let mut group = c.benchmark_group("parity_oracle");
    let len = 1usize << 16;
    let a_values = pinned_values(len, 1.0);
    let b_values = pinned_values(len, 0.5);

    let leto_a = Array::from_shape_vec([len], a_values).unwrap();
    let leto_b = Array::from_shape_vec([len], b_values).unwrap();

    group.bench_function("add_leto_64k", |bencher| {
        bencher.iter_batched(
            || Array::zeros([len]),
            |mut out| {
                add(
                    black_box(&leto_a.view()),
                    black_box(&leto_b.view()),
                    &mut out.view_mut(),
                )
                .unwrap();
                out
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("exp_leto_64k", |bencher| {
        bencher.iter(|| unary_map(ExpOp, black_box(&leto_a.view())).unwrap());
    });

    group.bench_function("sum_leto_64k", |bencher| {
        bencher.iter(|| sum(black_box(&leto_a.view())));
    });

    group.bench_function("dot_leto_64k", |bencher| {
        bencher.iter(|| dot(black_box(&leto_a.view()), black_box(&leto_b.view())).unwrap());
    });

    // Seeded random constructors (leto-native).
    use leto_ops::{normal_with_seed, uniform_with_seed};

    group.bench_function("uniform_leto_64k", |bencher| {
        bencher.iter(|| {
            uniform_with_seed(
                black_box([len]),
                black_box(-2.0),
                black_box(5.0),
                black_box(42),
            )
            .unwrap()
        });
    });

    group.bench_function("normal_leto_64k", |bencher| {
        bencher.iter(|| {
            normal_with_seed(
                black_box([len]),
                black_box(1.0),
                black_box(2.0),
                black_box(42),
            )
            .unwrap()
        });
    });

    group.finish();
}

