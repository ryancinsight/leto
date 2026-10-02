# leto-ops Criterion Baselines

Benchmark baselines and optimization history for `crates/leto-ops/benches/kernels.rs`.
These gate optimization work: a statistically significant regression in a
touched kernel blocks merge, and no change is labeled an optimization
without a recorded comparison (engineering_gates: performance gate).

## Harness and methodology

Harness: `crates/leto-ops/benches/kernels.rs` (`cargo bench -p leto-ops`).
Methodology: Criterion, sample size 10 except for the six operator-chain cases,
which use 50; median + 95% CI; pinned deterministic inputs (no RNG); default
features; f64. Historical rows retain their recorded sample size. Machine class: Windows 11 x86_64 dev
workstation (AVX2-class). These baselines gate optimization work (Atlas ADR
0002 leto slice): a statistically significant regression in a touched kernel
blocks merge, and no change is labeled an optimization without a recorded
comparison.

### Bounded full runs

Run the complete Criterion suite with `python scripts/bench.py`, or one binary
with the same committed per-binary ceiling, for example
`python scripts/bench.py col_piv_qr`. The runner reads the `leto-ops` benchmark
target inventory from Cargo metadata and requires it to match
`.config/bench.toml`; adding or removing a `[[bench]]` entry therefore requires
updating its case and timing model in the same change. Named group overrides
record source-level sample-size and timing changes. The validator computes
nominal Criterion time from each group's case count and warm-up plus measurement
time. The five target ceilings sum to 297 seconds under a 300-second suite
deadline; each target also reserves its configured warm-cache build and process
allowance.

The runner preserves Cargo's target directory, profiles, pinned Rust toolchain,
and member `.cargo/config.toml` environment. Inside the Atlas stack it invokes
Cargo from a temporary config root containing the stack configuration with only
the generated development `[patch]` overlay removed; this keeps the committed
lock in standalone form and the shared build cache and profile hashes
unchanged. It passes no target-directory or profile command-line override.
Every Cargo child has a finite absolute deadline. Each target reserves part of
its ceiling for process-tree termination and captured-output cleanup, and that
cleanup remains inside the 300-second suite deadline. A Windows tree-kill
failure stops the run rather than falling back to killing only the parent.
Standard benchmark output remains attached to the terminal so Criterion
medians and confidence intervals can be recorded directly.

The runner requires Python 3.11 or newer and uses only the standard library.
Run its value and process-tree tests from scripts/ with
python -m unittest discover -s tests/bench -t tests -p 'test_*.py'; CI runs the
same suite. On Windows, child processes start suspended and enter a Job Object
before resuming. Cleanup uses bounded completion-port waits and verifies the
kernel's active-process count; it does not wait on the Job Object handle or
leave descendants behind after a successful parent exit.

A completed run proves that the declared binaries finished inside the committed
wall-clock budgets and supplies empirical Criterion estimates for that host and
revision. It does not establish value correctness, cross-host performance, or a
speedup by itself. Performance comparisons still require identical inputs and
features, a recorded machine class and concurrent load, and median plus
confidence-interval comparison against the relevant baseline revision. A
timeout is a benchmark or production-path defect; do not increase the budget or
reduce the workload to make it pass.

## Current state (fused multi-row AXPY, 0.19.7, 2026-06-13)

0.19.7 keeps the row-blocked matmul contraction and replaces repeated per-row
Hermes AXPY dispatches with Hermes fused multi-row AXPY for positive-stride
output row blocks. 0.19.0 changes reverse-last-axis whole-array reductions by
borrowing physical unit-stride row slices and feeding them to the dense slice
reducers. Current hot matmul kernels do not call topology detection;
row-blocking uses a fixed const-generic 32-row block chosen to fit 32 f64
output rows plus one RHS row inside the conservative 256 KiB L2 fallback at
the 256-column benchmark shape.

| Benchmark | Median | Note |
| --- | --- | --- |
| matmul/dense_64x64 | 17.430 µs | oracle comparison median; fused multi-row Hermes AXPY (0.19.7) |
| matmul/dense_256x256 | 1.0631 ms | oracle comparison median; fused multi-row Hermes AXPY (0.19.7) |
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

## Topology policy evaluation (2026-08-08)

The post-repair quiet comparison used the existing Criterion harness with
all-features, deterministic f64 inputs, identical prepared strided operands,
an output-equivalence preflight, alternating target order, and full-output
checksums outside each timed interval. The automatic selector used
`cached_cache_geometry()`, matching the production topology cache. On the
observed host, L2 was 3 MiB and the selector chose 16 output rows; the explicit
control chose the retained 32-row specialization.

| Benchmark | Median | 95% CI | Result |
| --- | ---: | ---: | --- |
| `matmul/wide_policy_auto_64x64x4096` | 329.68 µs | 327.85–333.38 µs | value-equivalent adaptive 16-row route |
| `matmul/wide_policy_fixed32_64x64x4096` | 352.80 µs | 305.31–419.24 µs | value-equivalent retained production route |

The intervals overlap substantially, so this run is inconclusive under current
host variance and does not establish an adaptive win or regression. Production
convenience APIs therefore remain fixed at 32 rows; the explicit
`MatmulTilePolicy` seam is retained for future hardware-specific experiments.
This is policy-evaluation evidence, not a global performance ranking.

The dense C×C route now shares the policy-aware row-block/tiled-GEMM path rather
than using the removed `serial_cc_matmul`/`parallel_cc_matmul` bypass. A focused
64×64 fixed-1 versus fixed-32 differential test passed with identical output.
A separate route-coverage run measured `matmul/dense_64x64` at `6.0371 µs`
(`[5.6937–6.3841]`) and `matmul/dense_256x256` at `116.35 µs`
(`[115.89–117.16]`). These observations confirm execution of the dense route;
they are not claimed as a cross-run optimization or parity result.

## Contiguous and non-unit-stride coverage (2026-07-23)

Commands: `rustup run nightly cargo bench --locked -p leto-ops --bench kernels
"contiguous_256x256|strided_step2" -- --noplot` and `rustup run nightly cargo
bench --locked -p leto-ops --bench kernels
"matmul/(dense_256x256|strided_step2_lhs_256x256)" -- --noplot`. Criterion
uses the current sample-size-10, 500 ms warm-up/measurement configuration;
inputs are prepared before timing and the logical workload is 256×256 f64 in
every row.

| Benchmark | Median | 95% CI | Coverage |
| --- | ---: | ---: | --- |
| elementwise_add/contiguous_256x256 | 11.796 µs | 11.175–12.282 µs | C-dense binary map |
| elementwise_add/strided_step2_lhs_256x256 | 49.229 µs | 47.403–50.473 µs | non-unit-stride binary fallback |
| reductions/sum_contiguous_256x256 | 3.6693 µs | 3.5824–3.7651 µs | C-dense whole-array sum |
| reductions/sum_strided_step2_256x256 | 34.150 µs | 33.013–35.310 µs | non-unit-stride row walk |
| matmul/dense_256x256 | 407.01 µs | 331.81–449.28 µs | C-dense contraction |
| matmul/strided_step2_lhs_256x256 | 297.46 µs | 276.48–316.65 µs | fallback copy plus contraction |

The matmul step-2 row also measured 217.13 µs [209.40, 228.93] in the prior
coverage run. Concurrent Cargo builds and outliers make that row unsuitable
for a speedup claim; rerun on a quiet, counterbalanced host before changing a
production kernel.

## Non-unit-stride reduction audit (2026-07-23)

The quiet baseline command was `rustup run nightly cargo bench --locked
-p leto-ops --bench kernels --all-features
"reductions/sum_(contiguous_256x256|strided_step2_256x256)" -- --noplot`.
The candidate used one generic, order-preserving four-way loop in the existing
zero-copy fallback and was removed after the controlled comparison.

| Benchmark | Baseline median | Candidate median | Result |
| --- | ---: | ---: | --- |
| reductions/sum_contiguous_256x256 | 4.1184 µs | 4.6830 µs | candidate control regression in run |
| reductions/sum_strided_step2_256x256 | 28.849 µs | 27.793 µs | `p = 0.06`; no significant improvement |

The candidate did not justify a production change. The retained fallback
borrows the source storage and performs no materialization; the added
value-semantic test covers the selected step-2 logical values. After removing
the candidate, the unchanged implementation measured `28.226 µs`
[27.481, 28.889] in a 20-sample run; an intervening 10-sample run measured
`31.633 µs` [30.099, 33.701]. This spread is recorded as host/run variance,
not as a production regression or speedup claim.

## Dense matmul parity audit (2026-07-23)

Commands: `rustup run nightly cargo bench --locked -p leto-ops --bench
kernels --all-features
"oracle_compare/matmul_(leto|ndarray)_(64x64|128x128|256x256)" -- --noplot`
and a no-default-feature Leto comparison with `--features std`. The first run
used the current sample-size-10, 500 ms Criterion configuration on a quiet
host. The 64×64 threshold comparison was rerun with 20 samples, 1 s warm-up,
and 2 s measurement. These are empirical measurements, not proof of a global
ranking.

| Benchmark | Median | 95% CI | Result |
| --- | ---: | ---: | --- |
| oracle_compare/matmul_leto_64x64 (parallel) | 23.597 µs | 17.659–29.520 µs | slower than ndarray |
| oracle_compare/matmul_ndarray_64x64 | 12.770 µs | 11.437–14.460 µs | oracle baseline |
| oracle_compare/matmul_leto_128x128 (parallel) | 123.63 µs | 117.41–130.97 µs | near ndarray; intervals overlap |
| oracle_compare/matmul_ndarray_128x128 | 113.07 µs | 104.34–120.96 µs | oracle baseline |
| oracle_compare/matmul_leto_256x256 (parallel) | 233.60 µs | 202.28–253.87 µs | 4.08× faster than ndarray median |
| oracle_compare/matmul_ndarray_256x256 | 952.54 µs | 935.68–981.42 µs | oracle baseline |
| oracle_compare/matmul_leto_64x64 (serial) | 27.483 µs | 26.478–28.271 µs | 16.5% slower than parallel |
| oracle_compare/matmul_leto_128x128 (serial) | 223.69 µs | 222.72–224.43 µs | 80.9% slower than parallel |
| oracle_compare/matmul_leto_256x256 (serial) | 1.8522 ms | 1.8208–1.9032 ms | 7.93× slower than parallel |

The current parallel threshold is retained. `cargo flamegraph` could not
profile this Windows session because `dtrace` was unavailable and the
`blondie` fallback required administrator rights. No production optimization
claim is made; a future tile or packing change needs a working profile and a
value-preserving benchmark win.

## Historical oracle comparison gate (ndarray / nalgebra, 0.19.7, 2026-06-13)

Methodology: `cargo bench -p leto-ops --bench kernels --all-features
"oracle_compare/matmul_(leto|ndarray|nalgebra)_(64|128|256)x(64|128|256)"
-- --sample-size 10`; criterion medians + 95% CI; deterministic f64 inputs;
same process and machine class as the current baselines above.
Evidence tier: empirical benchmark comparison, not a proof.

| Benchmark | Median | Oracle conclusion |
| --- | --- | --- |
| oracle_compare/matmul_leto_64x64 | 17.430 µs | slower than ndarray/nalgebra; improved vs 0.19.5 |
| oracle_compare/matmul_ndarray_64x64 | 8.4923 µs | oracle baseline |
| oracle_compare/matmul_nalgebra_64x64 | 8.7752 µs | oracle baseline |
| oracle_compare/matmul_leto_128x128 | 108.98 µs | slower than ndarray/nalgebra; improved vs 0.19.5 |
| oracle_compare/matmul_ndarray_128x128 | 66.527 µs | oracle baseline |
| oracle_compare/matmul_nalgebra_128x128 | 62.935 µs | oracle baseline |
| oracle_compare/matmul_leto_256x256 | 1.0631 ms | slower than ndarray/nalgebra; improved vs 0.19.5 |
| oracle_compare/matmul_ndarray_256x256 | 495.95 µs | oracle baseline |
| oracle_compare/matmul_nalgebra_256x256 | 505.35 µs | oracle baseline |
| oracle_compare/sum_reverse_leto_256x256 | 4.7805 µs | faster than ndarray |
| oracle_compare/sum_reverse_ndarray_256x256 | 6.0717 µs | oracle baseline |
| oracle_compare/norm_l2_reverse_leto_256x256 | 9.3496 µs | faster than ndarray |
| oracle_compare/norm_l2_reverse_ndarray_256x256 | 30.877 µs | oracle baseline |

Historical result: reverse-last-axis reductions satisfy the recorded ndarray
performance parity on this benchmark shape. The dense matmul rows are retained
as the 0.19.7 historical baseline; the current parity audit above supersedes
their conclusion for the present default-feature build.

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
| Fused multi-row Hermes AXPY (0.19.7) | oracle matmul dense 64² / 128² / 256² | 21.443 µs → 17.430 µs / 127.63 µs → 108.98 µs / 2.4357 ms → 1.0631 ms | **−18.7% / −14.6% / −56.4% by median** |
| Batched Hermes row-panel AXPY (post-0.19.7) | oracle matmul dense 128² | 212.64 µs → 98.853 µs on the local themis-0.9 stack | **−53.5% median** |

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
- **Hermes column-chunk `axpy_rows` RHS reuse (post-0.19.7 audit)**: rejected.
  The loop loaded each RHS SIMD chunk once and applied it across the row block.
  Hermes AXPY tests and Leto focused matmul tests passed, but oracle matmul
  regressed to 29.913 µs / 671.87 µs / 9.0392 ms at 64x64/128x128/256x256 and
  the benchmark process ended with `STATUS_ACCESS_VIOLATION`. Hermes was
  restored to the row-major `axpy_rows` loop in `96c871b`; Leto remains pinned
  to the measured-good `efac045`.
- **`MATMUL_ROW_BLOCK=64` (post-0.19.7 audit)**: rejected. Focused matmul tests
  passed, but oracle matmul regressed versus 0.19.7 to 22.085 µs / 187.79 µs /
  2.1801 ms at 64x64/128x128/256x256. The 32-row block remains the measured
  best row-block geometry among tested 16/32/64 variants.
- **Row-block fused-branch and alpha-buffer hoist (post-0.19.7 audit)**:
  rejected. Focused matmul tests passed, but 128x128 oracle timing showed no
  statistically significant Leto improvement (275.97 µs median in the current
  dependency state) and the benchmark process ended with
  `STATUS_ACCESS_VIOLATION`. No source change retained.
- **Generic 4x4 registered dense tile (post-0.19.7 audit)**: rejected.
  Focused matmul tests passed, but the array-accumulator implementation
  regressed 64x64/128x128/256x256 to 75.684 µs / 695.06 µs / 6.2823 ms.
  A follow-up attempt to skip the zeroing pass for the same overwrite kernel
  regressed further or stayed unstable. No source change retained.
- **Broad depth-batched row-panel AXPY policy (post-0.19.7 audit)**:
  rejected. Hermes `axpy_rows_batch` is retained, but routing all medium+
  dense row-blocks through it regressed the 64x64/128x128/256x256 Leto-only
  oracle rows to 21.695 µs / 126.73 µs / 1.8273 ms in the local themis-0.9
  stack. Leto retains the measured 128-row gate only.
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
  **Superseded 2026-08-28** by the dense matmul parity closure below: the
  serial kernel measures ahead of ndarray at every oracle shape.
- dense matmul oracle parity — CLOSED 2026-08-28, premise superseded by
  measurement. The recorded deficit (64² 17.4 vs 8.5 µs, 128² 109 vs 66.5,
  256² 1063 vs 496) predates the re-landed dense `T::tiled_gemm` route and
  the Hermes lane-throughput overhaul (hermes ADR 017 + the dispatch fix);
  re-measured at HEAD `f527685` with a pinned same-binary external probe
  (leto path-dep beside ndarray 0.16 / nalgebra 0.34 — outside the repo, per
  the no-external-comparator dependency policy; best of 24 blocks per call,
  thread pinned per core type, three-engine value agreement asserted below
  1e-6):
  serial (leto-ops without `parallel`), leto/ndarray on the pinned P-core:
  64² 12.1/15.8 µs = 0.77x, 128² 92.2/117.3 = 0.79x, 256² 764/898 = 0.85x,
  512² 6574/7759 = 0.85x — the Leto serial kernel is 15–23 % ahead at every
  oracle shape, and 2–3x ahead of nalgebra. E-core: parity within noise
  except 128² at 1.28x, the one cell not won. The default (parallel) entry
  is 1.8x ahead at 64² and up to 15x at 512² against the references'
  single-threaded execution. The packing-scratch / register-micro-kernel
  lever is retired with the gap: re-open only on a regression of the
  `oracle_compare/matmul_leto_*` medians or a fresh external re-comparison,
  not on the superseded numbers above. The rejected-candidate list stays
  binding for any reopened work. Limits: one AVX2 host, one value
  distribution, references at default features (matrixmultiply runtime
  detection); the E-core 128² cell is recorded, not disputed.

## Decomposition kernel SIMD dispatch (2026-07-20)

LU, Hessenberg reduction, the SVD values-path, Francis QR, and the shared
Householder primitive already routed inner sweeps through the SIMD `Scalar`
ops (`dot_slice`/`axpy_slice`); three hot O(n³) kernels had not been
converted. Meta-pattern found: a scalar loop-carried **reduction** (`dot`)
does not autovectorize and converting it to `dot_slice` wins; a scalar
**axpy** already autovectorizes at the SSE2 baseline, so converting it only
wins when the slice is provably long (full-dimension), and loses (extra
cross-crate call + assert + Result) on short/shrinking slices.

Shipped:
- Cholesky–Crout inner product → `dot_slice`: **−49%/−72%/−65%** at
  n=128/256/512 (`bench_cholesky_scaling`), 2–3.5×; 15 QR/Cholesky value
  tests pass.
- SVD bidiagonal U/V accumulation (`dot` + paired long-slice axpy, both
  converted): **−52%/−49%/−38%** at n=64/128/192 (`bench_svd_scaling`),
  1.6–2.1×; 13 SVD tests pass — confirms the "provably long axpy" exception.
- udu weighted-dot, hoisted loop-invariant `w[k]` plus `dot_slice`:
  **−44%/−62%/−69%** at n=64/128/256 (`bench_udu_scaling`), 1.8–3.2×; 3 UDU
  tests pass — largest per-decomposition win measured.

Rejected (do not retry without a changed model):
- QR panel reflector apply (`qr/decompose.rs`): axpys over short,
  shrinking (~n/2, 32-col blocked panel) trailing slices. `axpy_slice`
  measured **+9–18%** at n=64/128/192/256 (`bench_qr_scaling`, p=0.00). Kept
  scalar.

Open: full_piv_lu / bunch_kaufman trailing-update axpys are LU-style long slices.
col_piv_qr pivot-norm down-dating is tracked by PR #290.

Cross-crate lead, not a leto item: hermes's CSR SpMV scalar remainder
(`hermes-simd-core/src/sparse/spmv.rs`) re-checks a gather bound the SIMD
body above it already trusts; short rows (nnz < LANE_COUNT) pay it in full.
Report to hermes, not tracked here.

## SpMV bounds-check elision (Krylov kernel, 2026-07-20)

`spmv_slice_into` (every Krylov iteration's CSR matvec) carried three
per-nonzero bounds checks the compiler could not prove away. Collapsing the
row loop to `row_ptr.windows(2)` zip `y`, slicing each row's value/column
runs, reduces `O(nnz)` checks to `O(nrows)`. Pure refactor — same traversal
order, bitwise-identical output.

- CSR: **−14% (n=4096) / −19% (n=65536) / −27% (n=1<<20, CI 19–34%)** vs
  `spmv_pre` (clean-host criterion, p=0.00); DRAM-bound case rose
  ~12.6→~18 GB/s.
- CSC (same elision, scatter-add): **−24% (n=4096) / −16% (n=65536)**
  (`bench_csc_spmv`).

Rejected: an unsafe `xs.get_unchecked(col)` on the remaining data-dependent
gather. The cache-resident n=4096 case (least bandwidth-sensitive) showed no
change (p=0.29) — no demonstrated benefit for the added unsafe/miri burden.
Not shipped.

Blocked lever, not yet filed: narrowing `CsrMatrix`'s `col_indices`/
`row_ptr` from `usize` to `u32` would halve index traffic on DRAM-bound
SpMV (the dominant term), but it is a public-API format change — file as a
[major] backlog item with an ADR before starting.

Concurrent builds invalidate wall-clock evidence when run-to-run variance
exceeds the derived noise bound. Use deterministic instruction/cache counters
for attribution, or isolate a wall-clock run on cores or a host-level benchmark
lease; record host load with the result.

## Blocked LU cache-resident regression (2026-07-20)

`lu_decompose` is unblocked (BLAS-2, rank-1 SIMD axpy trailing update); a
right-looking blocked (BLAS-3) LU (64-col panel, unit-lower solve, matmul
trailing update) was implemented and verified correct (`P·A=L·U` at n=200)
but measured **slower** at tested sizes: LU@256 988 µs → 1.65 ms; @512
neutral. Cause: this host's 36 MiB L3 keeps LU matrices cache-resident to
n≈1200, so unblocked SIMD axpy runs at cache bandwidth and the blocked
version's panel-extraction/allocation overhead dominates before the L3
threshold. Reverted — never ship a regression.

Re-open only with: (a) a gate on `working_set > l3_bytes` (the cache-aware
threshold the parallel policy already uses) so cache-resident sizes never
regress, (b) trailing-update copies eliminated via matmul into strided
views, (c) the win verified past the LLC using a valid isolated wall-clock run
or deterministic counters. `lu_scaling` and a large-n `P·A=L·U`
reconstruction test are retained as coverage.

## Layout-copy / assign kernel (Apollo FFT gather/scatter, 2026-08-26)

Apollo's non-contiguous 2-D/3-D FFT axis passes use caller-owned scratch and
cache-tiled gather/scatter loops; Leto's `assign` was value-correct over the
same transposed views but converted every linear position to an N-D index
with checked per-element lookup.

Pre-change baseline (locked Criterion, identical source/output allocations,
view construction outside timed closures), Leto `assign` vs. Apollo's tiled
loop:

| Shape | Direction | Leto `assign` | Apollo tiled loop | Ratio |
|---|---:|---:|---:|---:|
| 4096×16 | gather | 753.62 µs [747.54, 759.14] | 26.010 µs [25.747, 26.432] | 29.0× |
| 4096×16 | scatter | 747.83 µs [743.04, 752.47] | 28.851 µs [28.386, 29.279] | 25.9× |
| 4096×64 | gather | 3.1190 ms [3.0997, 3.1340] | 165.06 µs [163.29, 167.65] | 18.9× |
| 4096×64 | scatter | 3.7923 ms [3.7586, 3.8246] | 174.19 µs [171.51, 177.33] | 21.8× |
| 16384×16 | gather | 3.0940 ms [3.0847, 3.1056] | 202.27 µs [200.96, 205.07] | 15.3× |
| 16384×16 | scatter | 3.1699 ms [3.1543, 3.1857] | 172.02 µs [169.38, 174.31] | 18.4× |
| 65536×4 | gather | 3.0278 ms [3.0168, 3.0511] | 143.89 µs [141.43, 145.81] | 21.0× |
| 65536×4 | scatter | 3.0087 ms [2.9903, 3.0232] | 153.19 µs [149.14, 156.62] | 19.6× |

This established a Leto traversal gap, not an allocation or Apollo
FFT-arithmetic gap. Accepted target: one validated assignment kernel shared
by owned arrays and mutable views, with a tiled rank-2 transpose route and a
structural bounds-elided fallback for other strided layouts.

Post-change (same binary, no unsafe code, no transient allocation):

| Shape | Direction | Leto candidate | Apollo tiled loop | Comparison |
|---|---:|---:|---:|---:|
| 4096×16 | gather | 23.526 µs [23.299, 23.739] | 25.947 µs [25.459, 26.290] | Leto 9.3% lower; disjoint |
| 4096×16 | scatter | 28.290 µs [27.963, 28.711] | 28.580 µs [28.169, 29.075] | overlap |
| 4096×64 | gather | 174.46 µs [172.36, 176.63] | 175.21 µs [173.20, 176.34] | overlap |
| 4096×64 | scatter | 172.26 µs [168.98, 177.58] | 176.97 µs [174.52, 180.29] | overlap |
| 16384×16 | gather | 156.80 µs [155.95, 158.24] | 166.85 µs [164.27, 169.95] | Leto 6.0% lower; disjoint |
| 16384×16 | scatter | 163.07 µs [160.69, 166.01] | 185.15 µs [183.65, 186.87] | Leto 11.9% lower; disjoint |
| 65536×4 | gather | 142.88 µs [140.11, 146.23] | 149.88 µs [147.80, 152.86] | Leto 4.7% lower; disjoint |
| 65536×4 | scatter | 151.54 µs [149.00, 155.67] | 150.74 µs [148.57, 154.28] | overlap |

4/8 rows disjoint favoring Leto, 4/8 overlap (inconclusive, not a
regression). This establishes layout-copy throughput only; full FFT
behavior and steady-state allocation verification is Apollo's (tracked by
backlog `LETO-FFT-LAYOUT-THROUGHPUT`).
