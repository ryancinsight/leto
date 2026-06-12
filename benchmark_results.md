# leto-ops criterion baselines

Harness: `crates/leto-ops/benches/kernels.rs` (`cargo bench -p leto-ops`).
Methodology: criterion, sample_size 20, median + 95% CI; pinned deterministic
inputs (no RNG); default features; f64. Machine class: Windows 11 x86_64 dev
workstation (AVX2-class). These baselines gate optimization work (atlas ADR
0002 leto slice): a statistically significant regression in a touched kernel
blocks merge, and no change is labeled an optimization without a recorded
comparison.

## Current state (full sweep, 0.17.0, 2026-06-12)

| Benchmark | Median | Note |
| --- | --- | --- |
| matmul/dense_64x64 | 28.1 µs | i-k-j kernel, hermes AXPY rows (0.16.0); within noise of 27.4 µs |
| matmul/dense_256x256 | 1.529 ms | hermes AXPY unit-stride rows (0.16.0) |
| elementwise_add/contiguous_64k | 15.8 µs | hermes SIMD slice path |
| elementwise_add/transposed_256x256 | 34.8 µs | line-tiled (0.14.4) |
| unary_map/map_into_contiguous_64k | 13.0 µs | dense slice path |
| unary_map/map_into_transposed_256x256 | 23.4 µs | line-tiled (0.15.0) |
| reductions/sum_64k | 3.44 µs | hermes `sum_slice` |
| reductions/norm_l2_64k | 4.67 µs | hermes dot via `dot_slice` (0.11.3) |
| reductions/norm_l1_64k | 4.069 µs | hermes abs-sum (0.17.0); scalar ref 34.174 µs |
| reductions/norm_max_64k | 5.293 µs | hermes abs-max (0.17.0); scalar ref 39.961 µs |
| reductions/sum_transposed_256x256 | 4.48 µs | dense memory-order slice → hermes sum (0.16.0) |
| reductions/norm_l2_transposed_256x256 | 4.67 µs | dense memory-order slice → hermes dot |
| reductions/sum_reverse_last_axis_256x256 | 26.1 µs | unit-magnitude stride; row-walk by design |
| reductions/norm_l2_reverse_last_axis_256x256 | 25.0 µs | non-dense; row-walk fallback |
| zip/zip_mut_with_transposed_256x256 | 40.7 µs | line-tiled (0.16.1); closure-opaque body limits further gain |

## Measured optimization history

| Change | Benchmark | Before → After | Delta |
| --- | --- | --- | --- |
| Row-walk strided maps (0.11.1) | elementwise transposed 256² | 1.206 ms → ~50 µs | **−95.9% (23.7×)** |
| Row-walk strided reductions (0.11.2) | first strided reduction baselines | — | baselines |
| Hermes dot norms (0.11.3) | norm_l2 64k / dense transposed | 28.1 µs → 5.5 µs | **−80%** |
| Row-walk zip/scan/map_inplace (0.13.1) | zip transposed 256² | 553.4 µs → 55.9 µs | **−89.9% (9.9×)** |
| Line micro-tiling, binary (0.14.4) | elementwise transposed 256² | 50.7 µs → 28.4 µs | **−43.5%** |
| Line micro-tiling, unary (0.15.0) | map_into transposed 256² | ~50 µs class → 23.4 µs | tiled level |
| Hermes AXPY matmul rows (0.16.0) | matmul dense 256² | 2.210 ms → 1.529 ms | **−31%** |
| Sum memory-order fast path (0.16.0) | sum transposed 256² | 44.9 µs → 4.48 µs | **−90% (10×)** |
| Line micro-tiling, zip (0.16.1) | zip_mut_with transposed 256² | 47.6 µs → 40.7 µs | **−14.5%** |
| Hermes abs-reductions (0.17.0) | norm_l1 64k / norm_max 64k | 34.174 µs → 4.069 µs / 39.961 µs → 5.293 µs | **−88.1% / −86.8%** |

Cumulative on the headline case (elementwise transposed 256²): 1.206 ms →
~35 µs ≈ **35–42×** depending on run; residual vs contiguous is ~2.2×
(large-stride TLB/prefetch behavior — revisit only with profile evidence).

## Rejected optimization candidates (do not retry without a changed model)

- **Const-generic dense matmul blocking (0.14.3 audit)**: ROW_TILE=16 /
  SHARED_TILE=32 / COL_TILE=32 over dense row-major views regressed
  `64x64` to ~48.5 µs and `256x256` to ~3.37 ms vs the ~28 µs / ~2.25 ms
  baselines. Reverted.
- **Generic `Scalar::mul_add` matmul accumulation hook (0.14.3 audit)**:
  regressed `64x64` to ~245.6 µs and `256x256` to ~12.5 ms. Reverted.
- Constraint recorded in backlog Stage C2: matmul SIMD work waits on a
  hermes scalar-AXPY / fused row-update provider; leto must not emulate one
  with temporary allocation.

## Open measured targets

- `zip_mut_with` transposed now tiled at 40.7 µs; the residual vs the binary
  map (~28 µs) is the opaque `FnMut` body — no further structural target
  without an op-ZST zip variant, which no caller currently needs.
- Truly non-dense strided reductions (reverse-axis cases) still row-walk;
  tiling them needs per-lane partial accumulators — different shape from
  the map case.
- matmul: AXPY gate closed (0.16.0, −31% at 256²); revisiting blocking is
  permitted only on top of the AXPY row kernel (changed model) with
  criterion evidence.
