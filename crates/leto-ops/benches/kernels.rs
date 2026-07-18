//! Criterion baselines for the leto-ops hot kernels.
//!
//! These baselines are the prerequisite gate for the cache-aware tiling work
//! (atlas ADR 0002 leto slice): per `performance_engineering`, no change is
//! labeled an optimization without a recorded baseline comparison. Inputs are
//! pinned; report median + CI from criterion's standard output.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use leto::{Array, SliceArg};
use leto_ops::{
    add, bunch_kaufman, dot, map_into, matmul, norm_l1, norm_l2, norm_max, schur, sum, unary_map,
    zip_mut_with, AddOp, ExpOp,
};
use leto_ops::{
    cholesky_decompose, eigenvalues, lu_decompose, matexp, matpow, qr_decompose, singular_values,
    svd_via_bidiagonal,
};
use leto_ops::{spmm, CsrMatrix};
use ndarray::{Array1 as NdArray1, Array2 as NdArray2};
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

    // Leto vs ndarray only (nalgebra removed).
    for &n in &[64usize, 128, 256] {
        let lhs_values = pinned_values(n * n, 1.0e-3);
        let rhs_values = pinned_values(n * n, 2.0e-3);
        let leto_lhs = Array::from_shape_vec([n, n], lhs_values.clone()).unwrap();
        let leto_rhs = Array::from_shape_vec([n, n], rhs_values.clone()).unwrap();
        let ndarray_lhs = NdArray2::from_shape_vec((n, n), lhs_values.clone()).unwrap();
        let ndarray_rhs = NdArray2::from_shape_vec((n, n), rhs_values.clone()).unwrap();

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
        group.bench_function(format!("matmul_ndarray_{n}x{n}"), |bencher| {
            bencher.iter(|| black_box(&ndarray_lhs).dot(black_box(&ndarray_rhs)));
        });
    }

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

/// Broad leto-vs-ndarray oracle comparison across the elementwise, unary,
/// reduction, and vector-dot families (the completeness-harness performance
/// companion to `bench_oracle_compare`, which owns matmul and reverse
/// reductions). Same pinned f64 inputs feed both sides; criterion reports
/// median + CI per side so the ratio is a recorded empirical comparison.
fn bench_parity_oracle(c: &mut Criterion) {
    let mut group = c.benchmark_group("parity_oracle");
    let len = 1usize << 16;
    let a_values = pinned_values(len, 1.0);
    let b_values = pinned_values(len, 0.5);

    let leto_a = Array::from_shape_vec([len], a_values.clone()).unwrap();
    let leto_b = Array::from_shape_vec([len], b_values.clone()).unwrap();
    let nd_a = NdArray1::from_vec(a_values.clone());
    let nd_b = NdArray1::from_vec(b_values.clone());

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
    group.bench_function("add_ndarray_64k", |bencher| {
        bencher.iter(|| black_box(&nd_a) + black_box(&nd_b));
    });

    group.bench_function("exp_leto_64k", |bencher| {
        bencher.iter(|| unary_map(ExpOp, black_box(&leto_a.view())).unwrap());
    });
    group.bench_function("exp_ndarray_64k", |bencher| {
        bencher.iter(|| black_box(&nd_a).mapv(f64::exp));
    });

    group.bench_function("sum_leto_64k", |bencher| {
        bencher.iter(|| sum(black_box(&leto_a.view())));
    });
    group.bench_function("sum_ndarray_64k", |bencher| {
        bencher.iter(|| black_box(&nd_a).sum());
    });

    group.bench_function("dot_leto_64k", |bencher| {
        bencher.iter(|| dot(black_box(&leto_a.view()), black_box(&leto_b.view())).unwrap());
    });
    group.bench_function("dot_ndarray_64k", |bencher| {
        bencher.iter(|| black_box(&nd_a).dot(black_box(&nd_b)));
    });

    // Seeded random constructors
    use leto_ops::{normal_with_seed, uniform_with_seed};
    use ndarray_rand::rand_distr::{Normal, Uniform};
    use ndarray_rand::RandomExt;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

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
    group.bench_function("uniform_ndarray_64k", |bencher| {
        bencher.iter(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            NdArray1::random_using(black_box(len), black_box(Uniform::new(-2.0, 5.0)), &mut rng)
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
    group.bench_function("normal_ndarray_64k", |bencher| {
        bencher.iter(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            NdArray1::random_using(
                black_box(len),
                black_box(Normal::new(1.0, 2.0).unwrap()),
                &mut rng,
            )
        });
    });

    group.finish();
}

fn bench_linalg_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_compare");

    // Leto-only benchmarks (removed nalgebra comparison).
    for &n in &[32usize, 64] {
        let values = pinned_values(n * n, 1.0e-3);
        let leto_mat = Array::from_shape_vec([n, n], values.clone()).unwrap();

        group.bench_function(format!("schur_leto_{n}x{n}"), |bencher| {
            bencher.iter(|| black_box(schur(black_box(&leto_mat.view())).unwrap()));
        });

        // Bunch-Kaufman requires symmetric matrix.
        let mut sym_values = values.clone();
        for i in 0..n {
            for j in 0..n {
                sym_values[i * n + j] = values[if i < j { i * n + j } else { j * n + i }];
            }
        }
        let leto_sym = Array::from_shape_vec([n, n], sym_values).unwrap();

        group.bench_function(format!("bunch_kaufman_leto_{n}x{n}"), |bencher| {
            bencher.iter(|| black_box(bunch_kaufman(black_box(&leto_sym.view())).unwrap()));
        });
    }

    group.finish();
}

/// Leto-only decomposition baselines (removed nalgebra comparison).
fn bench_decomposition_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomposition_compare");

    for &n in &[32usize, 64] {
        let values = pinned_values(n * n, 1.0e-3);
        let leto_mat = Array::from_shape_vec([n, n], values.clone()).unwrap();

        group.bench_function(format!("lu_leto_{n}x{n}"), |b| {
            b.iter(|| black_box(lu_decompose(black_box(&leto_mat.view())).unwrap()))
        });

        group.bench_function(format!("qr_leto_{n}x{n}"), |b| {
            b.iter(|| black_box(qr_decompose(black_box(&leto_mat.view())).unwrap()))
        });

        group.bench_function(format!("svd_leto_{n}x{n}"), |b| {
            b.iter(|| black_box(svd_via_bidiagonal(black_box(&leto_mat.view())).unwrap()))
        });

        group.bench_function(format!("singular_values_leto_{n}x{n}"), |b| {
            b.iter(|| black_box(singular_values(black_box(&leto_mat.view())).unwrap()))
        });

        group.bench_function(format!("eig_leto_{n}x{n}"), |b| {
            b.iter(|| black_box(eigenvalues(black_box(&leto_mat.view())).unwrap()))
        });

        group.bench_function(format!("matexp_leto_{n}x{n}"), |b| {
            b.iter(|| black_box(matexp(black_box(&leto_mat.view())).unwrap()))
        });

        group.bench_function(format!("matpow_leto_{n}x{n}"), |b| {
            b.iter(|| black_box(matpow(black_box(&leto_mat.view()), 8).unwrap()))
        });

        // Cholesky needs SPD: build AᵀA + nI.
        let mut spd = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += values[k * n + i] * values[k * n + j];
                }
                spd[i * n + j] = acc + if i == j { n as f64 } else { 0.0 };
            }
        }
        let leto_spd = Array::from_shape_vec([n, n], spd).unwrap();
        group.bench_function(format!("cholesky_leto_{n}x{n}"), |b| {
            b.iter(|| black_box(cholesky_decompose(black_box(&leto_spd.view())).unwrap()))
        });
    }

    group.finish();
}

/// Sparse vs dense matrix product on a deliberately sparse operand: with the
/// matrix ~5% dense, the CSR `spmm` does `O(nnz·k)` work where dense `matmul`
/// does `O(n²·k)`, so the sparse path is expected ~order-of-magnitude faster.
/// The one-time `from_dense` compression is excluded from the timed region (the
/// sparse workflow compresses once and reuses); the dense matmul is the baseline.
fn bench_sparse_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_compare");
    let n = 256usize;
    let k = 32usize;

    // Deterministic ~5%-dense n×n matrix (one nonzero in ~20 entries).
    let mut dense_a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if (i * 7 + j * 13) % 20 == 0 {
                dense_a[i * n + j] = ((i + j) % 7 + 1) as f64;
            }
        }
    }
    let a = Array::from_shape_vec([n, n], dense_a).unwrap();
    let b = Array::from_shape_vec([n, k], pinned_values(n * k, 1.0e-3)).unwrap();
    let csr = CsrMatrix::from_dense(&a.view());

    group.bench_function("dense_matmul_256sq_x32", |bencher| {
        bencher.iter_batched(
            || Array::zeros([n, k]),
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
    group.bench_function("sparse_spmm_256sq_x32_5pct", |bencher| {
        bencher.iter(|| spmm(black_box(&csr), black_box(&b.view())).unwrap());
    });
    group.finish();
}

criterion_group! {
    name = kernels;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_millis(500))
        .without_plots();
    targets = bench_matmul, bench_elementwise, bench_unary_map, bench_reductions, bench_zip, bench_oracle_compare, bench_parity_oracle, bench_linalg_compare, bench_decomposition_compare, bench_sparse_compare
}
criterion_main!(kernels);
