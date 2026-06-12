//! Criterion baselines for the leto-ops hot kernels.
//!
//! These baselines are the prerequisite gate for the cache-aware tiling work
//! (atlas ADR 0002 leto slice): per `performance_engineering`, no change is
//! labeled an optimization without a recorded baseline comparison. Inputs are
//! pinned; report median + CI from criterion's standard output.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use leto::{Array, SliceArg};
use leto_ops::{map_into, matmul, norm_l1, norm_l2, norm_max, sum, zip_mut_with, AddOp};
use nalgebra::DMatrix;
use ndarray::Array2 as NdArray2;
use std::hint::black_box;

fn pinned_values(len: usize, scale: f64) -> Vec<f64> {
    // Deterministic, non-trivial values (no RNG: reproducible inputs).
    (0..len).map(|i| (i as f64 * 0.731 + 1.0) * scale).collect()
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for &n in &[64usize, 256] {
        let a = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0e-3)).unwrap();
        let b = Array::from_shape_vec([n, n], pinned_values(n * n, 2.0e-3)).unwrap();
        group.bench_function(format!("dense_{n}x{n}"), |bencher| {
            bencher.iter_batched(
                || Array::zeros([n, n]),
                |mut out| {
                    matmul(
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
    group.finish();
}

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
    group.finish();
}

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
                zip_mut_with(&mut out.view_mut(), &transposed, |o, &s| *o += s).unwrap();
                out
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_oracle_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("oracle_compare");
    let n = 128usize;
    let lhs_values = pinned_values(n * n, 1.0e-3);
    let rhs_values = pinned_values(n * n, 2.0e-3);
    let leto_lhs = Array::from_shape_vec([n, n], lhs_values.clone()).unwrap();
    let leto_rhs = Array::from_shape_vec([n, n], rhs_values.clone()).unwrap();
    let ndarray_lhs = NdArray2::from_shape_vec((n, n), lhs_values.clone()).unwrap();
    let ndarray_rhs = NdArray2::from_shape_vec((n, n), rhs_values.clone()).unwrap();
    let nalgebra_lhs = DMatrix::from_row_slice(n, n, &lhs_values);
    let nalgebra_rhs = DMatrix::from_row_slice(n, n, &rhs_values);

    group.bench_function("matmul_leto_128x128", |bencher| {
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
    group.bench_function("matmul_ndarray_128x128", |bencher| {
        bencher.iter(|| black_box(&ndarray_lhs).dot(black_box(&ndarray_rhs)));
    });
    group.bench_function("matmul_nalgebra_128x128", |bencher| {
        bencher.iter(|| black_box(&nalgebra_lhs) * black_box(&nalgebra_rhs));
    });

    let reduce_n = 256usize;
    let reduce_values = pinned_values(reduce_n * reduce_n, 1.0);
    let leto_reduce = Array::from_shape_vec([reduce_n, reduce_n], reduce_values.clone()).unwrap();
    let leto_reversed = leto_reduce
        .view()
        .slice_with::<2>(&[SliceArg::All, SliceArg::range(None, None, -1)])
        .unwrap();
    let ndarray_reduce = NdArray2::from_shape_vec((reduce_n, reduce_n), reduce_values).unwrap();
    let ndarray_reversed = ndarray_reduce.slice(ndarray::s![.., ..;-1]);

    group.bench_function("sum_reverse_leto_256x256", |bencher| {
        bencher.iter(|| sum(black_box(&leto_reversed)));
    });
    group.bench_function("sum_reverse_ndarray_256x256", |bencher| {
        bencher.iter(|| black_box(&ndarray_reversed).sum());
    });
    group.bench_function("norm_l2_reverse_leto_256x256", |bencher| {
        bencher.iter(|| norm_l2(black_box(&leto_reversed)).unwrap());
    });
    group.bench_function("norm_l2_reverse_ndarray_256x256", |bencher| {
        bencher.iter(|| {
            black_box(&ndarray_reversed)
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
        });
    });
    group.finish();
}

criterion_group! {
    name = kernels;
    config = Criterion::default().sample_size(20).without_plots();
    targets = bench_matmul, bench_elementwise, bench_unary_map, bench_reductions, bench_zip, bench_oracle_compare
}
criterion_main!(kernels);
