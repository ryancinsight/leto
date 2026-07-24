//! Criterion baselines for the leto-ops hot kernels.
//!
//! These baselines are the prerequisite gate for cache-aware kernel work
//! (Atlas ADR 0002 Leto slice): per , no change is
//! labeled an optimization without a recorded baseline comparison. Inputs are
//! pinned; report median + CI from Criterion's standard output. Each hot-kernel
//! family includes a C-dense case and a prepared non-unit-stride view so view
//! construction and allocation do not contaminate the timed operation.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use leto::{Array, SliceArg};
use leto_ops::{
    add, bunch_kaufman, dot, map_into, matmul, norm_l1, norm_l2, norm_max, scalar_map_into, schur,
    sum, unary_map, zip_mut_with, AddOp, ExpOp,
};
use leto_ops::{
    cholesky_decompose, eigenvalues, lu_decompose, matexp, matpow, qr_decompose, singular_values,
    svd_via_bidiagonal, udu_decompose,
};
use leto_ops::{csc_spmv_into, spmm, spmv_into, CscMatrix, CsrMatrix};
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

    let n = 256usize;
    let strided_lhs_storage =
        Array::from_shape_vec([n * 2, n * 2], pinned_values(n * n * 4, 1.0e-3)).unwrap();
    let strided_lhs = strided_lhs_storage
        .view()
        .slice_with::<2>(&[
            SliceArg::range(None, None, 2),
            SliceArg::range(None, None, 2),
        ])
        .unwrap();
    let rhs = Array::from_shape_vec([n, n], pinned_values(n * n, 2.0e-3)).unwrap();
    group.bench_function("strided_step2_lhs_256x256", |bencher| {
        bencher.iter_batched(
            || Array::zeros([n, n]),
            |mut out| {
                matmul(
                    black_box(&strided_lhs),
                    black_box(&rhs.view()),
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
    // Sweep bandwidth-bound   across the LLC-residency crossover to
    // validate the working-set-vs-L3 parallel gate. Working set = 3·N·8 bytes, so
    // on a 36 MiB L3 the gate's crossover sits near N ≈ 1.5M. Run under default
    // features for the gate's decision,  for an all-serial
    // baseline, and with the gate temporarily forced parallel to locate the true
    // crossover during threshold calibration.
    let mut group = c.benchmark_group("parallel_crossover");
    for &n in &[524_288usize, 1_048_576, 2_097_152, 4_194_304, 8_388_608] {
        let a = Array::from_shape_vec([n], pinned_values(n, 1.0)).unwrap();
        let b = Array::from_shape_vec([n], pinned_values(n, 0.5)).unwrap();
        group.bench_function(format!("add_{}k", n / 1024), |bencher| {
            bencher.iter_batched(
                || Array::zeros([n]),
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
    }
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
                zip_mut_with(&mut out.view_mut(), &transposed, |o, &s| *o += s).unwrap();
                out
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

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

/// LU scaling instrument. The factorization is a rank-1 (BLAS-2) trailing
/// update; on a large-L3 host it stays cache-resident (and fast) until the
/// working set exceeds the LLC (n ≈ 1200 at 36 MiB), so a blocked (BLAS-3)
/// rewrite only pays past that size — an investigated but not-yet-shipped
/// optimization (gap_audit ).
fn bench_lu_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu_scaling");
    for &n in &[128usize, 256, 512] {
        let mat = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0e-3)).unwrap();
        group.bench_function(format!("lu_{n}x{n}"), |b| {
            b.iter(|| black_box(lu_decompose(black_box(&mat.view())).unwrap()))
        });
    }
    group.finish();
}

/// Banded CSR matrix with  nonzeros per interior row — the
/// structure of a 1-D stencil / discretized-PDE operator, the canonical Krylov
/// SpMV workload. Column indices are strictly increasing within each row (CSR
/// invariant); the diagonal is heavy so the operand is well-scaled.
fn banded_csr(n: usize, half_bw: usize) -> CsrMatrix<f64> {
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptr = Vec::with_capacity(n + 1);
    row_ptr.push(0);
    for i in 0..n {
        let lo = i.saturating_sub(half_bw);
        let hi = (i + half_bw + 1).min(n);
        for j in lo..hi {
            col_indices.push(j);
            values.push(if j == i {
                (2 * half_bw + 1) as f64
            } else {
                -1.0
            });
        }
        row_ptr.push(values.len());
    }
    CsrMatrix::from_parts(values, col_indices, row_ptr, n, n).expect("banded CSR is valid")
}

/// SpMV  scaling instrument.  is a 7-point-stencil banded operator
/// (the per-iteration kernel of every Krylov solve).  stays L2-resident
/// (isolates per-nonzero instruction overhead);  spills past the LLC
/// (memory-bandwidth-bound).  is timed with a reused output buffer so
/// the measurement is the kernel, not allocation.
fn bench_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv");
    for &n in &[4096usize, 65536, 1 << 20] {
        let a = banded_csr(n, 3);
        let x = Array::from_shape_vec([n], pinned_values(n, 1.0e-3)).unwrap();
        let mut y = vec![0.0f64; n];
        group.bench_function(format!("banded7_{n}"), |bencher| {
            bencher.iter(|| {
                spmv_into(
                    black_box(&a),
                    black_box(&x.view()),
                    black_box(y.as_mut_slice()),
                )
                .unwrap();
            });
        });
    }
    group.finish();
}

/// Banded CSC matrix — the column-major analogue of banded_csr, same 1-D
/// stencil structure (row indices strictly increasing within each column).
fn banded_csc(n: usize, half_bw: usize) -> CscMatrix<f64> {
    let mut values = Vec::new();
    let mut row_indices = Vec::new();
    let mut col_ptr = Vec::with_capacity(n + 1);
    col_ptr.push(0);
    for j in 0..n {
        let lo = j.saturating_sub(half_bw);
        let hi = (j + half_bw + 1).min(n);
        for i in lo..hi {
            row_indices.push(i);
            values.push(if i == j {
                (2 * half_bw + 1) as f64
            } else {
                -1.0
            });
        }
        col_ptr.push(values.len());
    }
    CscMatrix::from_parts(values, row_indices, col_ptr, n, n).expect("banded CSC is valid")
}

/// CSC SpMV  scaling instrument — the scatter-add, column-major
/// analogue of bench_spmv across the same L2/L3/DRAM size ladder.
fn bench_csc_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("csc_spmv");
    for &n in &[4096usize, 65536, 1 << 20] {
        let a = banded_csc(n, 3);
        let x = Array::from_shape_vec([n], pinned_values(n, 1.0e-3)).unwrap();
        let mut y = vec![0.0f64; n];
        group.bench_function(format!("banded7_{n}"), |bencher| {
            bencher.iter(|| {
                csc_spmv_into(
                    black_box(&a),
                    black_box(&x.view()),
                    black_box(y.as_mut_slice()),
                )
                .unwrap();
            });
        });
    }
    group.finish();
}

/// Row-major SPD matrix  (well-conditioned, positive-definite), the
/// input Cholesky requires.
fn spd_values(n: usize) -> Vec<f64> {
    let values = pinned_values(n * n, 1.0e-3);
    let mut spd = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += values[k * n + i] * values[k * n + j];
            }
            spd[i * n + j] = acc + if i == j { n as f64 } else { 0.0 };
        }
    }
    spd
}

/// Cholesky scaling instrument. The  factorization is dominated by the
/// Cholesky–Crout inner-product reduction; these cache-resident sizes isolate
/// that reduction's throughput (scalar vs SIMD-dispatched).
fn bench_cholesky_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_scaling");
    for &n in &[128usize, 256, 512] {
        let spd = Array::from_shape_vec([n, n], spd_values(n)).unwrap();
        group.bench_function(format!("cholesky_{n}x{n}"), |b| {
            b.iter(|| black_box(cholesky_decompose(black_box(&spd.view())).unwrap()))
        });
    }
    group.finish();
}

/// QR scaling instrument. The Householder panel reflector's rank-1 apply
/// dominates at O(m·n²). Square sizes
/// below  (256) run the *entire* apply through the within-panel
/// scalar loops; n=256 crosses into the blocked compact-WY path, so the
/// SIMD-dispatch win concentrates at n<256.
fn bench_qr_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_scaling");
    for &n in &[64usize, 128, 192, 256] {
        let mat = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0e-3)).unwrap();
        group.bench_function(format!("qr_{n}x{n}"), |b| {
            b.iter(|| black_box(qr_decompose(black_box(&mat.view())).unwrap()))
        });
    }
    group.finish();
}

/// SVD factor-path scaling instrument.  accumulates the U/V
/// orthogonal factors by applying the bidiagonalization reflectors — a per-row
///  (reduction) + axpy over full-dimension contiguous slices, O(dim³) total.
fn bench_svd_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_scaling");
    for &n in &[64usize, 128, 192] {
        let mat = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0e-3)).unwrap();
        group.bench_function(format!("svd_{n}x{n}"), |b| {
            b.iter(|| black_box(svd_via_bidiagonal(black_box(&mat.view())).unwrap()))
        });
    }
    group.finish();
}

/// UDUᵀ scaling instrument. The symmetric-indefinite factorization's inner work
/// is a per-entry weighted dot ;  is
/// loop-invariant across the -loop, so hoisting it and reducing via
/// is both an algorithmic (O(n³) recompute) and a SIMD win. SPD input is a safe
/// symmetric subset (no zero pivots).
fn bench_udu_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("udu_scaling");
    for &n in &[64usize, 128, 256] {
        let sym = Array::from_shape_vec([n, n], spd_values(n)).unwrap();
        group.bench_function(format!("udu_{n}x{n}"), |b| {
            b.iter(|| black_box(udu_decompose(black_box(&sym.view())).unwrap()))
        });
    }
    group.finish();
}

criterion_group! {
    name = kernels;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_millis(500))
        .without_plots();
    targets = bench_matmul, bench_elementwise, bench_parallel_crossover, bench_unary_map, bench_reductions, bench_zip, bench_oracle_compare, bench_parity_oracle, bench_linalg_compare, bench_decomposition_compare, bench_lu_scaling, bench_sparse_compare, bench_spmv, bench_csc_spmv, bench_cholesky_scaling, bench_qr_scaling, bench_svd_scaling, bench_udu_scaling
}
criterion_main!(kernels);
