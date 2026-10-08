//! Oracle and library comparison benchmarks.

use super::*;

/// Leto-only matmul and reverse-reduction baselines (removed external comparison).
fn bench_oracle_compare(c: &mut Criterion) {
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
fn bench_parity_oracle(c: &mut Criterion) {
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

fn bench_linalg_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_compare");

    // Leto-only benchmarks (removed external comparison).
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

/// Leto-only decomposition baselines (removed external comparison).
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
            b.iter(|| black_box(svd_decompose(black_box(&leto_mat.view())).unwrap()))
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
/// matrix ~5% dense, the CSR  does  work where dense
/// does , so the sparse path is expected ~order-of-magnitude faster.
/// The one-time  compression is excluded from the timed region (the
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
