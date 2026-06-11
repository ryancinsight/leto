# leto-ops criterion baselines

Harness: `crates/leto-ops/benches/kernels.rs` (`cargo bench -p leto-ops`).
Methodology: criterion, sample_size 20, median + 95% CI; pinned deterministic
inputs (no RNG); default features (`std`, `simd`, `parallel`); f64.
Machine class: Windows 11 x86_64 dev workstation (AVX2-class), 2026-06-11.
Baselines gate the cache-aware tiling work (atlas ADR 0002 leto slice): a
statistically significant regression in a touched kernel blocks merge.

| Benchmark | Baseline (pre row-walk, 0.11.0) | Row-walk maps (0.11.1) | Row-walk reductions (0.11.2) | Hermes dot norm (0.11.3) | Change |
| --- | --- | --- | --- | --- | --- |
| matmul/dense_64x64 | 28.42 µs | unchanged (untouched kernel) | 29.80 µs | 28.34 µs | no attributed change; untouched kernel |
| matmul/dense_256x256 | 2.376 ms | unchanged (untouched kernel) | 2.401 ms | 2.245 ms | no attributed change; untouched kernel |
| elementwise_add/contiguous_64k | 13.85 µs | 13.59 µs | 14.85 µs | 15.11 µs | no attributed change; untouched in 0.11.3 |
| elementwise_add/transposed_256x256 | 1.206 ms | 50.98 µs → 49.19 µs | 55.30 µs | 50.65 µs | 0.11.1: **−95.9% (23.7×), p < 0.05** |
| reductions/sum_64k | 3.607 µs | unchanged (untouched; ±9% run-to-run noise observed and reversed on rerun) | 3.789 µs | 3.653 µs | no change (p = 0.97 vs 0.11.2 sample) |
| reductions/norm_l2_64k | 28.06 µs | unchanged (untouched) | 28.07 µs | 5.508 µs | **−80.0%, p < 0.05** |
| reductions/sum_transposed_256x256 | not measured | not measured | 40.73 µs | 40.24 µs | no change (p = 0.67) |
| reductions/norm_l2_transposed_256x256 | not measured | not measured | 28.67 µs | 5.550 µs | **−80.7%, p < 0.05** |
| reductions/sum_reverse_last_axis_256x256 | not measured | not measured | 30.55 µs | 31.36 µs | no change (p = 0.60) |
| reductions/norm_l2_reverse_last_axis_256x256 | not measured | not measured | 30.21 µs | 30.82 µs | no change (p = 0.15); non-dense negative stride still row-walks |

| zip/zip_mut_with_transposed_256x256 | 553.4 µs (pre row-walk, 0.13.0) | 55.9 µs (0.13.1) | **−89.9% (9.9×), p < 0.05** |

## Rejected Optimization Candidates

- **Const-generic dense matmul blocking (0.14.3 audit, not shipped)**:
  candidate tile shape `ROW_TILE=16`, `SHARED_TILE=32`, `COL_TILE=32` routed
  only dense row-major views through a zero-allocation blocked path. Criterion
  measured `matmul/dense_64x64` at 46.169-50.686 µs and
  `matmul/dense_256x256` at 3.3176-3.4166 ms, regressing the retained
  baselines (~28.34 µs and ~2.245 ms). Source reverted.
- **Generic `Scalar::mul_add` matmul accumulation hook (0.14.3 audit, not
  shipped)**: candidate routed matmul accumulation through a trait hook using
  native `f32`/`f64::mul_add`. Criterion measured `matmul/dense_64x64` at
  232.66-255.99 µs and `matmul/dense_256x256` at 11.346-13.303 ms. Source
  reverted.

## Observations (drive the optimization backlog)

- **Row-walk policy complete (0.13.1)**: every strided fallback (binary,
  unary map/mapv/map_inplace, all four zips, whole-array reductions/norms,
  scan lanes) routes through `RowMajorTraversal`. The serial zip fallback —
  previously per-element with two offset products — measured 553.4 µs →
  55.9 µs on the transposed 256×256 case. Remaining strided cost is the
  L1-tile blocking item.

- **Row-walk traversal landed (0.11.1)**: the strided elementwise paths now
  compute offsets once per innermost row (`RowMajorTraversal`) and walk the
  last axis by stride increments, eliminating per-element div/mod index
  decomposition and per-element offset products. Measured: transposed add
  1.206 ms → 49–51 µs (−95.9%, 23.7×, p < 0.05) with contiguous unchanged.
  The residual ~3.6× gap vs contiguous (49 µs vs 13.6 µs) is genuine
  cache-line behavior of column-stride walks — the remaining L1-tile
  blocking item targets it.
- **Whole-array strided reductions/norms use row-walk traversal (0.11.2)**:
  transposed and reverse-last-axis reductions are now separately measured.
  The new criterion cases establish first baselines for strided whole-array
  reductions: transposed `sum` 40.73 µs, transposed `norm_l2` 28.67 µs,
  reverse-last-axis `sum` 30.55 µs, and reverse-last-axis `norm_l2` 30.21 µs.
- **Dense L2/Frobenius norm uses Hermes dot (0.11.3)**: the contiguous fast
  path computes `Σ x²` through the generic `Scalar::dot_slice` hook. Native
  f32/f64 dispatch through Hermes; f16/bf16 retain the existing scalar
  fallback. Measured: `norm_l2_64k` 28.07 µs → 5.508 µs (−80.0%,
  p < 0.05), and dense-memory transposed `norm_l2` 28.67 µs → 5.550 µs
  (−80.7%, p < 0.05). Reverse-last-axis views are not dense memory slices and
  remain on the row-walk fallback.
- matmul 256³ at ~2.38 ms ≈ 14 GFLOP/s (2·n³/t): memory-bound at this size
  on this machine class; blocking (L1/L2 tiles from themis `CacheLevel`)
  is the standard remedy and is backlogged behind these baselines.
- norm_l2 now dispatches dense f32/f64 square accumulation through Hermes dot.
  Remaining norm work is the non-dense strided path and any future Hermes
  fused square-accumulate kernel that avoids dot self-alias dispatch overhead.
