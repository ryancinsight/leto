# leto-ops criterion baselines

Harness: `crates/leto-ops/benches/kernels.rs` (`cargo bench -p leto-ops`).
Methodology: criterion, sample_size 20, median + 95% CI; pinned deterministic
inputs (no RNG); default features (`std`, `simd`, `parallel`); f64.
Machine class: Windows 11 x86_64 dev workstation (AVX2-class), 2026-06-11,
leto 0.11.0. Baselines gate the cache-aware tiling work (atlas ADR 0002 leto
slice): a statistically significant regression against these blocks merge.

| Benchmark | Baseline (pre row-walk) | After row-walk (0.11.1) | Change |
| --- | --- | --- | --- |
| matmul/dense_64x64 | 28.42 µs | unchanged (untouched kernel) | — |
| matmul/dense_256x256 | 2.376 ms | unchanged (untouched kernel) | — |
| elementwise_add/contiguous_64k | 13.85 µs | 13.59 µs | no change (p = 0.56) |
| elementwise_add/transposed_256x256 | 1.206 ms | 50.98 µs → 49.19 µs | **−95.9% (23.7×), p < 0.05** |
| reductions/sum_64k | 3.607 µs | unchanged (untouched; ±9% run-to-run noise observed and reversed on rerun) | — |
| reductions/norm_l2_64k | 28.06 µs | unchanged (untouched) | — |

## Observations (drive the optimization backlog)

- **Row-walk traversal landed (0.11.1)**: the strided elementwise paths now
  compute offsets once per innermost row (`RowMajorTraversal`) and walk the
  last axis by stride increments, eliminating per-element div/mod index
  decomposition and per-element offset products. Measured: transposed add
  1.206 ms → 49–51 µs (−95.9%, 23.7×, p < 0.05) with contiguous unchanged.
  The residual ~3.6× gap vs contiguous (49 µs vs 13.6 µs) is genuine
  cache-line behavior of column-stride walks — the remaining L1-tile
  blocking item targets it.
- matmul 256³ at ~2.38 ms ≈ 14 GFLOP/s (2·n³/t): memory-bound at this size
  on this machine class; blocking (L1/L2 tiles from themis `CacheLevel`)
  is the standard remedy and is backlogged behind these baselines.
- norm_l2 is ~7.8× slower than sum over the same data: sum dispatches
  through hermes SIMD `sum_slice`; the norm fold is a scalar loop —
  a Stage C2 hermes-coverage item (fused square-accumulate).
