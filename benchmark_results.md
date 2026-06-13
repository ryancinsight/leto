# leto-ops criterion baselines

Harness: `crates/leto-ops/benches/kernels.rs` (`cargo bench -p leto-ops`).
Methodology: criterion, sample_size 20, median + 95% CI; pinned deterministic
inputs (no RNG); default features; f64. Machine class: Windows 11 x86_64 dev
workstation (AVX2-class). These baselines gate optimization work (atlas ADR
0002 leto slice): a statistically significant regression in a touched kernel
blocks merge, and no change is labeled an optimization without a recorded
comparison.

## Current state (oracle matrix expansion, 0.19.5, 2026-06-13)

0.19.3 keeps the row-blocked Hermes AXPY matmul contraction and changes only
the output zeroing phase: dense output storage and unit-stride output rows are
filled through slices before strided fallback. 0.19.0 changes reverse-last-axis
whole-array reductions by borrowing physical
unit-stride row slices and feeding them to the dense slice reducers. Current
hot matmul kernels do not call topology detection; row-blocking uses a fixed
const-generic 32-row block chosen to fit 32 f64 output rows plus one RHS row
inside the conservative 256 KiB L2 fallback at the 256-column benchmark shape.

| Benchmark | Median | Note |
| --- | --- | --- |
| matmul/dense_64x64 | 22.536 µs | row-blocked Hermes AXPY rows (0.18.1), ~−19.8% vs recorded 28.1 µs table baseline |
| matmul/dense_256x256 | 1.4016 ms | row-blocked Hermes AXPY rows (0.18.1), ~−8.3% vs recorded 1.529 ms table baseline |
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
| reductions/sum_reverse_last_axis_256x256 | 5.203 µs | borrowed unit-stride row slices (0.19.0), −21.56% median in criterion run |
| reductions/norm_l2_reverse_last_axis_256x256 | 9.615 µs | borrowed unit-stride row slices (0.19.0), −18.00% median in criterion run |
| zip/zip_mut_with_transposed_256x256 | 40.7 µs | line-tiled (0.16.1); closure-opaque body limits further gain |

## Oracle comparison gate (ndarray / nalgebra, 0.19.5, 2026-06-13)

Methodology: `cargo bench -p leto-ops --bench kernels --all-features
"oracle_compare/matmul_(leto|ndarray|nalgebra)_(64|128|256)x(64|128|256)"
-- --sample-size 10`; criterion medians + 95% CI; deterministic f64 inputs;
same process and machine class as the current baselines above.
Evidence tier: empirical benchmark comparison, not a proof.

| Benchmark | Median | Oracle conclusion |
| --- | --- | --- |
| oracle_compare/matmul_leto_64x64 | 21.443 µs | slower than ndarray/nalgebra |
| oracle_compare/matmul_ndarray_64x64 | 11.566 µs | oracle baseline |
| oracle_compare/matmul_nalgebra_64x64 | 12.241 µs | oracle baseline |
| oracle_compare/matmul_leto_128x128 | 127.63 µs | slower than ndarray/nalgebra |
| oracle_compare/matmul_ndarray_128x128 | 106.16 µs | oracle baseline |
| oracle_compare/matmul_nalgebra_128x128 | 108.70 µs | oracle baseline |
| oracle_compare/matmul_leto_256x256 | 2.4357 ms | slower than ndarray/nalgebra |
| oracle_compare/matmul_ndarray_256x256 | 606.89 µs | oracle baseline |
| oracle_compare/matmul_nalgebra_256x256 | 848.08 µs | oracle baseline |
| oracle_compare/sum_reverse_leto_256x256 | 4.7805 µs | faster than ndarray |
| oracle_compare/sum_reverse_ndarray_256x256 | 6.0717 µs | oracle baseline |
| oracle_compare/norm_l2_reverse_leto_256x256 | 9.3496 µs | faster than ndarray |
| oracle_compare/norm_l2_reverse_ndarray_256x256 | 30.877 µs | oracle baseline |

Result: reverse-last-axis reductions satisfy current ndarray performance parity
on this benchmark shape. Dense matmul does not: Leto is ~1.85x slower than
ndarray at 64x64, ~1.20x slower at 128x128, and ~4.01x slower at 256x256 by
median. Replacement claims must exclude dense matmul performance parity until
the open target below is closed.

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
| Row-blocked matmul (0.18.1) | matmul dense 64² / 256² | 28.1 µs → 22.536 µs / 1.529 ms → 1.4016 ms | **~−19.8% / ~−8.3%** |
| Reverse-row reduction slices (0.19.0) | sum / norm_l2 reverse-last-axis 256² | 6.517 µs → 5.203 µs / 11.775 µs → 9.615 µs | **−21.56% / −18.00% median in criterion run** |

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
- **Remove dense row-block zero-skip branch (0.19.2 audit)**: 128x128 oracle
  measurements were within noise, while the canonical `matmul/dense_256x256`
  run became unstable and showed a regressed median. Reverted; do not retry
  branch removal without a branch-miss profile showing the zero check is the
  bound.
- **Packed RHS columns + `Scalar::dot_slice` (0.19.3 audit)**: packs RHS once
  and computes each output through contiguous SIMD dot hooks, but regressed
  `oracle_compare/matmul_leto_128x128` to 242.96 µs. Reverted; allocation plus
  dot-call granularity loses to the zero-copy row-AXPY kernel.
- **Inline scalar row update in the row-block path (0.19.3 audit)**: replacing
  Hermes AXPY with a generic inlined loop regressed
  `oracle_compare/matmul_leto_128x128` to 203.28 µs. Reverted; keep the Hermes
  row update until a fused multi-row provider exists.
- **Hermes `tiled_gemm` f64 dense path (0.19.4 audit)**: wiring the existing
  Hermes row-major tiled GEMM facade into Leto's dense C-contiguous matmul path
  was value-correct but regressed `oracle_compare/matmul_leto_128x128` to
  317.46 µs. Reverted; the current Hermes tiled GEMM surface is not the f64
  replacement kernel for Leto dense matmul.
- **Raise/disable parallel row-block scheduling for small dense matmul
  (0.19.4 audit)**: rejected. All-features row-block parallelism beat the
  serial-SIMD build for the current oracle sizes: 128x128 144.15 µs vs
  170.25 µs, and 64x64 21.759 µs vs 23.665 µs.
- **`MATMUL_ROW_BLOCK=16` (0.19.5 audit)**: rejected. Focused matmul tests
  passed, but the release benchmark process ended with
  `STATUS_ACCESS_VIOLATION`; no source change retained.
- **First-shared-row output initialization (0.19.5 audit)**: rejected. The path
  skipped the separate output-zero pass for row-blocked matmul and initialized
  each row from the first shared row before AXPY accumulation. Focused matmul
  tests passed, but `matmul/dense_64x64` regressed to 26.807 µs median and the
  release benchmark process ended with `STATUS_ACCESS_VIOLATION`; no source
  change retained.
- Constraint recorded in backlog Stage C2: matmul SIMD work waits on a
  hermes scalar-AXPY / fused row-update provider; leto must not emulate one
  with temporary allocation.

## Open measured targets

- `zip_mut_with` transposed now tiled at 40.7 µs; the residual vs the binary
  map (~28 µs) is the opaque `FnMut` body — no further structural target
  without an op-ZST zip variant, which no caller currently needs.
- Truly non-dense strided reductions with |last-axis stride| > 1 still
  row-walk; tiling them needs per-lane partial accumulators — different shape
  from the unit-stride row-slice case closed in 0.19.0.
- matmul: fixed row-blocking on top of AXPY is closed (0.18.1). Remaining
  work is topology-adaptive tile sizing across row/block/column dimensions,
  not another unmeasured rewrite of the current row-block kernel.
- matmul output zeroing now uses dense and unit-stride row slice fills before
  the strided fallback. This is a memory-efficiency cleanup in the initialization
  phase, not a parity claim; the contraction bottleneck remains open.
- dense matmul oracle parity: 64x64, 128x128, and 256x256 Leto medians remain
  slower than ndarray/nalgebra. Next kernel work targets RHS packing,
  row/block/column micro-kernel shape, and cache-geometry selection; no
  replacement-performance claim is valid until this comparison is closed.
  Rejected paths now include local row-loop rewrites, existing Hermes tiled
  GEMM, reduced parallelism, smaller row blocks, and first-row initialization.
  The next candidate needs a new fused multi-row row-update provider or an
  allocation-controlled packing API with reusable scratch.
