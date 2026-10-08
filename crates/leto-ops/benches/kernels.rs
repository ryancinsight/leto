//! Criterion baselines for the leto-ops hot kernels.
//!
//! These baselines are the prerequisite gate for cache-aware kernel work
//! (Atlas ADR 0002 Leto slice): per the stack's performance-engineering rule,
//! no change is labeled an optimization without a recorded baseline
//! comparison. Inputs are pinned; report median + CI from Criterion's
//! standard output.
//!
//! Read the CI, not just the median, and check the host is quiet first: a
//! concurrent build elsewhere in the stack widens these intervals past the
//! effects being measured. A run with four rustc processes alongside it put
//! matmul/dense_64x64 at [18.5, 24.1, 30.4] us -- a +/-25% spread, wide
//! enough to hide or invent any realistic kernel change. Each hot-kernel
//! family includes a C-dense case and a prepared non-unit-stride view so view
//! construction and allocation do not contaminate the timed operation.

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use leto::{Array, SliceArg};
use leto_ops::{
    add, bunch_kaufman, cached_cache_geometry, dot, full_piv_lu, map_into,
    map_into_with_cache_geometry, matmul, matmul_with_tile_policy, norm_l1, norm_l2, norm_max,
    scalar_map_into, schur, sum, unary_map, zip_mut_with, AddOp, CacheGeometry, ExpOp,
    MatmulTilePolicy,
};
use leto_ops::{
    cholesky_decompose, eigenvalues, lu_decompose, matexp, matpow, qr_decompose, singular_values,
    svd_decompose, udu_decompose,
};
use leto_ops::{csc_spmv_into, spmm, spmv_into, CscMatrix, CsrMatrix};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn pinned_values(len: usize, scale: f64) -> Vec<f64> {
    // Deterministic, non-trivial values (no RNG: reproducible inputs).
    (0..len).map(|i| (i as f64 * 0.731 + 1.0) * scale).collect()
}

mod chains;
mod elementwise;
mod maps;
mod matmul;
mod oracle;
mod scaling;

criterion_group! {
    name = kernels;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_millis(500))
        .without_plots();
    targets = matmul::bench_matmul, elementwise::bench_elementwise, chains::bench_operator_chain, elementwise::bench_parallel_crossover, maps::bench_unary_map, maps::bench_runtime_tile_geometry, maps::bench_reductions, maps::bench_zip, oracle::bench_oracle_compare, oracle::bench_parity_oracle, oracle::bench_linalg_compare, oracle::bench_decomposition_compare, scaling::bench_lu_scaling, scaling::bench_full_piv_lu_scaling, scaling::bench_bunch_kaufman_scaling, oracle::bench_sparse_compare, scaling::bench_spmv, scaling::bench_csc_spmv, scaling::bench_cholesky_scaling, scaling::bench_qr_scaling, scaling::bench_svd_scaling, scaling::bench_udu_scaling
}
criterion_main!(kernels);
