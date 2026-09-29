//! Linear-algebra, decomposition and sparse bench families.

use super::pinned_values;
use criterion::{BatchSize, Criterion};
use leto::Array;
use leto_ops::{
    bunch_kaufman, cholesky_decompose, csc_spmv_into, eigenvalues, full_piv_lu, lu_decompose,
    matmul, matexp, matpow, qr_decompose, schur, singular_values, spmm, spmv_into, svd_decompose,
    udu_decompose, CscMatrix, CsrMatrix,
};
use std::hint::black_box;

pub(super) fn bench_linalg_compare(c: &mut Criterion) {
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
pub(super) fn bench_decomposition_compare(c: &mut Criterion) {
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
pub(super) fn bench_sparse_compare(c: &mut Criterion) {
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
pub(super) fn bench_lu_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu_scaling");
    for &n in &[128usize, 256, 512] {
        let mat = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0e-3)).unwrap();
        group.bench_function(format!("lu_{n}x{n}"), |b| {
            b.iter(|| black_box(lu_decompose(black_box(&mat.view())).unwrap()))
        });
    }
    group.finish();
}

/// Full-pivoting LU scaling instrument. Complete pivoting's trailing-update
/// axpy (`LETO-DECOMP-AXPY-FOLLOWUPS-2026-09-27`) is a long slice at every
/// step (`n - k - 1` down to `0`, same shape as `bench_lu_scaling`'s
/// partial-pivot update), so this isolates the SIMD-dispatch win on the
/// complete-pivoting path specifically.
pub(super) fn bench_full_piv_lu_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_piv_lu_scaling");
    for &n in &[128usize, 256, 512] {
        let mat = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0e-3)).unwrap();
        group.bench_function(format!("full_piv_lu_{n}x{n}"), |b| {
            b.iter(|| black_box(full_piv_lu(black_box(&mat.view())).unwrap()))
        });
    }
    group.finish();
}

/// Bunch-Kaufman scaling instrument. Both the 1x1 and 2x2 pivot trailing
/// updates (`LETO-DECOMP-AXPY-FOLLOWUPS-2026-09-27`) are long-slice axpys
/// over the symmetric trailing block, same shape as Cholesky's converted
/// updates.
pub(super) fn bench_bunch_kaufman_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("bunch_kaufman_scaling");
    for &n in &[128usize, 256, 512] {
        let values = pinned_values(n * n, 1.0e-3);
        let mut sym = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                sym[i * n + j] = values[if i < j { i * n + j } else { j * n + i }];
            }
        }
        let mat = Array::from_shape_vec([n, n], sym).unwrap();
        group.bench_function(format!("bunch_kaufman_{n}x{n}"), |b| {
            b.iter(|| black_box(bunch_kaufman(black_box(&mat.view())).unwrap()))
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
pub(super) fn bench_spmv(c: &mut Criterion) {
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
pub(super) fn bench_csc_spmv(c: &mut Criterion) {
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
pub(super) fn bench_cholesky_scaling(c: &mut Criterion) {
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
pub(super) fn bench_qr_scaling(c: &mut Criterion) {
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
pub(super) fn bench_svd_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_scaling");
    for &n in &[64usize, 128, 192] {
        let mat = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0e-3)).unwrap();
        group.bench_function(format!("svd_{n}x{n}"), |b| {
            b.iter(|| black_box(svd_decompose(black_box(&mat.view())).unwrap()))
        });
    }
    group.finish();
}

/// UDUᵀ scaling instrument. The symmetric-indefinite factorization's inner work
/// is a per-entry weighted dot ;  is
/// loop-invariant across the -loop, so hoisting it and reducing via
/// is both an algorithmic (O(n³) recompute) and a SIMD win. SPD input is a safe
/// symmetric subset (no zero pivots).
pub(super) fn bench_udu_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("udu_scaling");
    for &n in &[64usize, 128, 256] {
        let sym = Array::from_shape_vec([n, n], spd_values(n)).unwrap();
        group.bench_function(format!("udu_{n}x{n}"), |b| {
            b.iter(|| black_box(udu_decompose(black_box(&sym.view())).unwrap()))
        });
    }
    group.finish();
}

/// Operator-chain instrument for the core elementwise operator tier (ADR 0004).
///
/// Each pair is the same expression in the two receiver forms, timed in one run
/// so the comparison is immune to cross-run drift on a hybrid P/E-core host:
/// `borrowed` re-borrows every intermediate and allocates one array per binary
/// operator (n-1 for n terms); `owned` lets the middle temporary be consumed and
/// reused, allocating once regardless of term count. Operand construction is
/// outside the timed region; the closure returns its result so the chain cannot
/// be elided.
///
/// Two sizes because the forms differ in allocation, and allocation cost is
/// size-regime-dependent: 64x64 (32 KiB/array) keeps every intermediate in the
/// allocator's cache and cache-resident, so the timing reflects the traversal
/// itself; 256x256 (512 KiB/array) is the regime where each freed intermediate
/// returns to the OS and the borrowed form additionally pays first-touch page
/// faults. The 64x64 pair is the conservative comparison; 256x256 shows the
/// larger but noisier end (read its CI, not just its median).
pub(super) fn bench_operator_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("operator_chain");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(1));

    for &n in &[64usize, 256] {
        let a = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0)).unwrap();
        let b = Array::from_shape_vec([n, n], pinned_values(n * n, 0.5)).unwrap();
        let d = Array::from_shape_vec([n, n], pinned_values(n * n, 0.25)).unwrap();

        group.bench_function(format!("borrowed_3term_{n}x{n}"), |bencher| {
            bencher.iter(|| &(black_box(&a) + black_box(&b)) + black_box(&d));
        });
        group.bench_function(format!("owned_3term_{n}x{n}"), |bencher| {
            bencher.iter(|| black_box(&a) + black_box(&b) + black_box(&d));
        });
    }

    // Term-count scaling at the cache-resident size: the borrowed form's
    // allocation count grows with n, the owned form's does not.
    let n = 64usize;
    let a = Array::from_shape_vec([n, n], pinned_values(n * n, 1.0)).unwrap();
    let b = Array::from_shape_vec([n, n], pinned_values(n * n, 0.5)).unwrap();
    let d = Array::from_shape_vec([n, n], pinned_values(n * n, 0.25)).unwrap();
    let e = Array::from_shape_vec([n, n], pinned_values(n * n, 0.125)).unwrap();
    group.bench_function("borrowed_5term_64x64", |bencher| {
        bencher.iter(|| {
            &(&(&(black_box(&a) + black_box(&b)) + black_box(&d)) + black_box(&e)) + black_box(&a)
        });
    });
    group.bench_function("owned_5term_64x64", |bencher| {
        bencher
            .iter(|| black_box(&a) + black_box(&b) + black_box(&d) + black_box(&e) + black_box(&a));
    });
    group.finish();
}

