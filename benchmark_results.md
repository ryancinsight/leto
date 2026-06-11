# leto-ops criterion baselines

Harness: `crates/leto-ops/benches/kernels.rs` (`cargo bench -p leto-ops`).
Methodology: criterion, sample_size 20, median + 95% CI; pinned deterministic
inputs (no RNG); default features (`std`, `simd`, `parallel`); f64.
Machine class: Windows 11 x86_64 dev workstation (AVX2-class), 2026-06-11,
leto 0.11.0. Baselines gate the cache-aware tiling work (atlas ADR 0002 leto
slice): a statistically significant regression against these blocks merge.

| Benchmark | Median | 95% CI |
| --- | --- | --- |
| matmul/dense_64x64 | 28.42 µs | [27.74, 29.03] µs |
| matmul/dense_256x256 | 2.376 ms | [2.322, 2.429] ms |
| elementwise_add/contiguous_64k | 13.85 µs | [13.28, 14.50] µs |
| elementwise_add/transposed_256x256 | 1.206 ms | [1.179, 1.236] ms |
| reductions/sum_64k | 3.607 µs | [3.554, 3.664] µs |
| reductions/norm_l2_64k | 28.06 µs | [27.81, 28.28] µs |

## Observations (drive the optimization backlog)

- **Strided traversal is the dominant cache problem**: transposed elementwise
  add over the same 65 536 elements is ~87× slower than contiguous
  (1.206 ms vs 13.85 µs). The strided fallback recomputes `index_from_flat`
  + `offset_of` per element and strides column-wise through rows — L1/L2
  unfriendly. Cache-aware tiling (process in L1-sized blocks along the
  fast axis) is the targeted fix; these rows are its before-numbers.
- matmul 256³ at ~2.38 ms ≈ 14 GFLOP/s (2·n³/t): memory-bound at this size
  on this machine class; blocking (L1/L2 tiles from themis `CacheLevel`)
  is the standard remedy and is backlogged behind these baselines.
- norm_l2 is ~7.8× slower than sum over the same data: sum dispatches
  through hermes SIMD `sum_slice`; the norm fold is a scalar loop —
  a Stage C2 hermes-coverage item (fused square-accumulate).
