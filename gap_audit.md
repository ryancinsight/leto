# Leto Gap Audit: ndarray / nalgebra Replacement for Atlas

## 2026-09-01 Apollo complex matrix-batch transpose

Apollo's retained 3-D FFT transpose phase exposed a distinct layout workload:
hundreds of adjacent 4x4 or 16x16 complex matrices. A broad Hermes route was
rejected because it regressed large rectangular 2-D matrices by 5-52% in the
consumer attribution instrument. The accepted operation therefore lives in
`leto-ops`, selects exact-width register tiles only at 256 or more matrices
whose sides are at most 16, and retains Leto's generic tiled assignment for
every other shape or unsupported hardware width. Leto core remains independent
of Hermes.

The bounded `layout_copy/complex_batch` Criterion instrument constructs and
validates input/output storage outside timing, reuses the same buffers, returns
a computed output sample through `black_box`, and pairs the provider with the
unchanged generic assignment in one binary. Across two independently launched
runs on the local Windows AVX2 workstation, provider median reductions were
86.7-88.8% (`f32`) and 88.9-89.8% (`f64`) for 1,024x4x4, and 28.3-53.3%
(`f32`) and 26.1-30.5% (`f64`) for 256x16x16. In the second run, provider and
generic median 95% confidence intervals were respectively 9.453-9.584 us and
71.189-73.286 us (`f32`, 1,024x4x4), 8.041-8.341 us and 71.258-75.256 us
(`f64`, 1,024x4x4), 14.558-15.560 us and 32.313-33.234 us (`f32`, 256x16x16),
and 22.273-25.176 us and 31.769-33.659 us (`f64`, 256x16x16). Every interval
pair is disjoint. A warmed global-allocator census records zero allocations and
zero reallocations for both scalar types. These are layout-operation results,
not full FFT or cross-machine evidence; Apollo integration owns that closure.

See [ADR 0027](docs/adr/0027-hermes-complex-batch-transpose.md).

## 2026-08-26 Apollo FFT layout-copy baseline

Apollo's non-contiguous 2-D and 3-D FFT axis passes use caller-owned scratch
and cache-tiled gather/scatter loops. Leto's existing caller-owned `assign`
contract is value-correct for the same transposed views, but it converts every
linear position to an N-dimensional index and performs checked source and
destination lookup per element.

The locked `layout_copy` Criterion instrument reuses identical source and
output allocations between candidates and keeps view construction outside the
timed closures. Median and 95% confidence intervals on the default Windows
workstation are:

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

Every confidence interval is disjoint. This establishes a Leto traversal
implementation gap, not an allocation result or an Apollo FFT-arithmetic gap.
The accepted implementation target is one validated assignment kernel shared
by owned arrays and mutable views, with a tiled rank-2 transpose route and a
structural bounds-elided fallback for other strided layouts.

The candidate implements that target without unsafe code or transient
allocation. Same-binary post-change medians are:

| Shape | Direction | Leto candidate | Apollo tiled loop | Descriptive comparison |
|---|---:|---:|---:|---:|
| 4096×16 | gather | 23.526 µs [23.299, 23.739] | 25.947 µs [25.459, 26.290] | Leto median 9.3% lower; intervals disjoint |
| 4096×16 | scatter | 28.290 µs [27.963, 28.711] | 28.580 µs [28.169, 29.075] | intervals overlap |
| 4096×64 | gather | 174.46 µs [172.36, 176.63] | 175.21 µs [173.20, 176.34] | intervals overlap |
| 4096×64 | scatter | 172.26 µs [168.98, 177.58] | 176.97 µs [174.52, 180.29] | intervals overlap |
| 16384×16 | gather | 156.80 µs [155.95, 158.24] | 166.85 µs [164.27, 169.95] | Leto median 6.0% lower; intervals disjoint |
| 16384×16 | scatter | 163.07 µs [160.69, 166.01] | 185.15 µs [183.65, 186.87] | Leto median 11.9% lower; intervals disjoint |
| 65536×4 | gather | 142.88 µs [140.11, 146.23] | 149.88 µs [147.80, 152.86] | Leto median 4.7% lower; intervals disjoint |
| 65536×4 | scatter | 151.54 µs [149.00, 155.67] | 150.74 µs [148.57, 154.28] | intervals overlap |

Each candidate is a separate Criterion function, so these are independent
samples within one binary rather than paired observations. Relative medians
are descriptive; overlapping confidence intervals are inconclusive, and this
instrument does not establish formal non-inferiority. Four rows have disjoint
intervals favoring Leto and four overlap. The manual candidate's absolute time
also moved between binaries, consistent with code placement and host variance,
so the stored entry baseline establishes only that checked logical indexing was
the dominant prior cost, not an exact cross-binary improvement percentage. This
establishes layout-copy throughput only. Apollo integration still must verify
full FFT behavior and steady-state allocation.

## 2026-08-13 Convolution provider closure

The generic regular and transposed convolution family is complete at provider
default `e525d8dd5ee52d12de0bf61987e8af6bf896700f`. PRs #78, #79, and #80
delivered the forward, backward, transposed backward, and canonical parameter
identity changes. Hosted run `31663241086` passed formatting, minimal-feature
compilation, warning-denied Clippy, configured Nextest, doctests, and
documentation at that exact head. Coeus default
`aabdec67a0f5baa415c4abb6dded69db41b2f2d6` consumes the provider directly and
deletes its former host convolution loops; hosted run `31672329963` passes at
that exact consumer head.

The remaining 33 Leto Rustdoc broken/private-link warnings predate this family
and are recorded in the provider backlog. No convolution-specific warning,
compatibility adapter, fallback, or performance claim remains open. Hephaestus
owns accelerator execution and its separate device-gate evidence.

## 2026-08-08 Fallible plain mutable iteration

- **Finding:** Leto had logical stride-aware read-only iteration and checked
  `indexed_iter_mut`, but no plain mutable iterator for callers that do not
  need the logical index. The existing `Array::iter_mut` remains the
  contiguous-only compatibility surface and cannot serve arbitrary strides.
- **Resolution:** add `ElementIterMut` plus `Array::try_iter_mut` and
  `ArrayViewMut::try_iter_mut`. The plain iterator delegates to the existing
  injectivity-checked `IndexedIterMut`, so positive, negative, and transposed
  layouts remain zero-copy while zero-stride or otherwise aliased layouts are
  rejected before any `&mut T` escapes. No infallible mutable `IntoIterator`
  is added because its trait contract cannot report layout rejection.
- **Evidence target:** logical-order strided mutation, double-ended/exact-size
  behavior inherited from the shared iterator, alias rejection, formatting,
  warning-denied compile/test gates, doctests, and Rustdoc. This closes the
  basic mutable-iteration ergonomics gap; broader tensor batch/channel layout
  work remains separate.

## 2026-08-08 Borrowed iterator ergonomics

- **Resolution:** `&Array` and `&ArrayViewMut` now implement
  `IntoIterator` using the existing logical, stride-aware `ElementIter`.
  Owned-array and mutable-view callers can use idiomatic read-only `for`
  traversal without materializing a contiguous copy.
- **Safety boundary:** mutable iteration remains the fallible
  `indexed_iter_mut` contract. Its preflight validates storage reachability
  and rejects non-injective/zero-stride layouts before yielding any `&mut T`;
  no infallible mutable `IntoIterator` is introduced.
- **Evidence:** the core iteration suite covers owned-array, transposed,
  mutable-view, indexed mutable, double-ended, empty, and alias-rejection
  cases; focused Leto Nextest passes 14/14. This closes iterator ergonomics
  within the broader tensor batch/channel layout item, which remains open.

## 2026-08-04 Provider-owned CPU cross-entropy gap

- **Finding:** a downstream consumer currently owns stable softmax,
  mean-cross-entropy, and its logit gradient because Leto exposes no borrowed
  classification-loss contract.
- **Impact:** CPU backend selection does not identify the implementation owner;
  the consumer stages probabilities in an owned vector and duplicates provider
  validation and arithmetic.
- **Resolution:** ADR 0023 assigns stable mean forward and additive backward to
  one scalar-generic Leto operation family over borrowed views and caller-owned
  outputs. Exact-head Rust verification passed and PR #94 merged as `c743a60`.
  Accelerator execution and consumer dispatch remain in their owning
  repositories.
- **Re-open trigger:** the provider contract, conformance evidence, and exact-head
  merge are complete.

## 2026-07-29 Coeus convolution provider gap

- **Finding:** Coeus cannot delete its CPU convolution kernels until Leto owns
  regular and transposed forward/backward convolution contracts.
- **Resolution:** Leto owns regular and transposed N-dimensional
  forward/additive-backward leaves, with checked preflight validation,
  borrowed inputs, caller-owned outputs, and one monomorphized implementation
  per operation across scalar and rank dimensions. The lightweight `leto`
  domain owns parameter vocabulary shared with accelerator planners; ADR 0019
  owns the contract.
- **Evidence:** focused Nextest passes 21 exact value-semantic and
  failure-atomicity contracts across f32, f64, F16, and Bf16, including 1-D,
  2-D, and 3-D transposed forward/backward convolution, strided views, and
  output-padding gradient semantics; package test targets compile warning-free
  on Rust 1.97 GNU;
  doctests pass 11/11 with one existing ignored case; and all 196 applicable
  minor-release SemVer checks pass.
- **Residual:** Coeus has not yet cut CPU dispatch over to these APIs.
  Warning-denied Clippy collection is blocked by mixed MSYS2/rustup artifacts
  in the shared target cache, not by a Leto diagnostic. Rustdoc completes but
  reports 33 pre-existing broken-link warnings.

## 2026-08-08 Topology-adaptive matmul policy

- **Resolution:** `leto-ops::MatmulTilePolicy` consumes the existing
  `CacheGeometry` contract and selects only bounded, power-of-two row-block
  instantiations already supported by the matmul kernel. The policy budgets
  one quarter of detected L2 per output-row footprint, caps the common route at
  32 rows, and downsizes unusually wide rows without allocating or changing
  arithmetic order.
- **Evidence tier:** pure policy tests cover the measured 32-row common shapes,
  wide-row down-sizing, empty/tiny inputs, and fallback geometry. The explicit
  policy seam is threaded through serial and parallel row-block paths; dense C×C
  inputs no longer bypass it. The legacy `serial_cc_matmul` /
  `parallel_cc_matmul` fast path was removed so dense and generic layouts share
  the policy-aware row-block/tiled-GEMM implementation. A 64×64 fixed-1 versus
  fixed-32 differential test is value-equivalent. **Benchmark (2026-08-08):**
  dense route medians were `6.0371 µs` at 64×64 and `116.35 µs` at 256×256 in
  one run; these are route-coverage observations, not a cross-run speedup
  claim. The alternating-order, checksum-consuming, value-preserving
  64×64×4096 strided comparison on a host with 3 MiB L2 selected 16 rows
  automatically (`329.68 µs`, 95% CI `327.85–333.38 µs`) versus fixed 32 rows
  (`352.80 µs`, 95% CI `305.31–419.24 µs`). The intervals overlap substantially,
  so host variance prevents a reliable adaptive policy ranking. **Status:**
  implementation and route coverage delivered; adaptive performance ranking
  remains inconclusive. Fixed 32 remains the production policy and the explicit
  adaptive selector remains available for future hardware-specific evidence.

## 2026-07-23 Contiguous and non-unit-stride benchmark coverage

- **Finding:** the canonical `leto-ops` Criterion harness had selected
  transposed/reversed cases, but it did not isolate a C-dense baseline from a
  genuinely non-unit-stride fallback for each of elementwise binary mapping,
  whole-array reduction, and matrix multiplication. The production dispatch
  predicates already distinguish these layouts; the missing evidence was
  benchmark coverage, not an established kernel defect.
- **Resolution:** `crates/leto-ops/benches/kernels.rs` now prepares identical
  logical 256×256 C-dense and step-2 views outside the timed closures. New
  rows cover elementwise add, whole-array sum, and matmul with a step-2 LHS;
  the existing transposed/reverse cases remain for their distinct layout
  contracts. No production kernel or API changed.
- **Evidence:** locked default-feature Criterion runs on the default Windows
  workstation reported elementwise C-dense `11.796 µs` [11.175, 12.282] vs
  step-2 `49.229 µs` [47.403, 50.473], and sum C-dense `3.6693 µs`
  [3.5824, 3.7651] vs step-2 `34.150 µs` [33.013, 35.310]. A matched matmul
  run reported dense `407.01 µs` [331.81, 449.28] vs step-2 `297.46 µs`
  [276.48, 316.65].
- **Limit:** the matmul step-2 row also measured `217.13 µs` [209.40, 228.93]
  in an earlier run, with outliers in both runs; this host was concurrently
  building Cargo targets. The matmul numbers are therefore coverage evidence,
  not a speedup claim. A quiet-host counterbalanced rerun is required before
  any production optimization or topology decision.
- **Verification:** `cargo check --locked -p leto-ops --benches`, warning-denied
  all-target Clippy, `cargo nextest run --locked -p leto-ops --all-features`
  (306/306), `cargo test --doc --locked -p leto-ops --all-features` (8/8),
  warning-denied package Rustdoc, package format check, and `git diff --check`
  pass. The benchmark target is the only source file changed.

## 2026-07-23 Non-unit-stride reduction audit

- **Finding:** the existing whole-array reduction fallback already walks the
  borrowed view directly, so it performs no copy or temporary allocation. Its
  remaining cost is scalar loop/index work for a non-unit last-axis stride;
  the canonical `sum_strided_step2_256x256` case measured `28.849 µs`
  [28.408, 29.110] on a quiet host.
- **Experiment:** an order-preserving four-way generic loop was evaluated in
  the canonical reduction module. It measured `27.793 µs` [27.052, 28.853]
  versus the baseline with `p = 0.06`, which is not a significant improvement.
  The contiguous control moved from `4.1184 µs` [4.0946, 4.1298] to
  `4.6830 µs` [4.4823, 4.8297] in the candidate run, so the candidate was
  removed rather than retained as speculative optimization.
- **Resolution:** retain the existing zero-copy row-walk implementation and
  add `whole_reduction_preserves_non_unit_stride_values` to pin the selected
  logical values. No allocation, scalar-type fork, or operation-specific
  duplicate was introduced.
- **Limit:** no call-stack profile was available on this Windows session;
  future strided reduction work requires a working profiler or an independent
  measured kernel model before changing the production traversal. A post-revert
  20-sample run of the unchanged implementation measured `28.226 µs`
  [27.481, 28.889], while an intervening 10-sample run measured `31.633 µs`
  [30.099, 33.701]; this run-to-run spread prevents attribution of small
  deltas to the removed candidate.

## 2026-07-22 Sparse LU native-view boundary

- **Finding:** `SparseLuSolver::solve` only accepted `&[T]` and returned
  `Vec<T>`. `CFDrs` therefore copied its native `Array1` right-hand side into
  a temporary `Vec`, and copied the provider's `Vec` result back into a new
  `Array1` on every direct solve.
- **Resolution:** add one provider-owned `solve_view` method over
  `ArrayView1`, route the legacy slice method through it, and migrate the
  `CFDrs` consumer to the native view/result contract. The dense-backed LU
  algorithm and matrix storage remain unchanged.
- **Evidence target:** provider and consumer value-semantic direct-solve
  regressions, warning-denied package gates, configured Nextest, doctest,
  Rustdoc, and public-surface SemVer classification. Allocation reduction is
  established by the ownership/data-flow audit; no runtime allocation profile
  is claimed by this change.

## 2026-07-20 Decomposition SIMD-Dispatch Gap (Cholesky shipped)

- **Finding:** LU, Hessenberg reduction, the SVD values-path, Francis QR, and the
  shared Householder primitive all route their inner sweeps through the SIMD
  `Scalar` ops (`dot_slice`/`axpy_slice`), but three hot O(n³) decomposition
  kernels were missed by that conversion and still ran hand-rolled scalar loops.
  Measurement note: `schur` is the slowest decomposition (348 µs @32² vs LU 2.8 µs)
  but it is a red herring — its Francis inner kernels are already SIMD; the cost is
  Q-accumulation/iterations, not a scalar loop.
- **Shipped (Cholesky):** `cholesky_decompose`'s Cholesky–Crout inner product
  (`cholesky.rs:50`) was a scalar loop-carried reduction that never autovectorizes.
  Routed through `dot_slice` (both operands already contiguous; same dispatch
  `solve_in_place` already used). **−49% / −72% / −65% at n=128/256/512**
  (`bench_cholesky_scaling`), a 2–3.5× win; reduction reorder within the
  differential oracle's tolerance (15 QR/Cholesky value tests pass).
- **Meta-pattern (from Cholesky + QR):** a scalar **reduction** (dot; loop-carried
  FP dependency → does *not* autovectorize) converts to `dot_slice` as a big win
  (Cholesky, 2–3.5×). A scalar **axpy** (no loop-carried dependency → *already*
  autovectorizes at the SSE2 baseline, inlined) does *not*: `axpy_slice` adds a
  cross-crate `hermes_simd::axpy` call + `assert` + `Result` per call, which loses
  to the inlined SSE2 loop for short slices. Convert reductions; leave
  already-vectorizing axpys scalar unless the slices are provably long.
- **Follow-ups (disjoint increments):**
  - QR panel reflector apply (`qr/decompose.rs:144-164`) — **investigated →
    regression, not converted.** The within-panel `w += vᵣ·row_r` / `row_r −= vᵣ·w`
    are axpys over short (trail ~n/2, shrinking; 32-col blocked panels) slices;
    `axpy_slice` measured **+9–18% at n=64/128/192/256** (`bench_qr_scaling`, clean
    ~6-proc, p=0.00). Kept scalar; QR authors had already tuned the blocked-path
    crossover (`BLOCK_MIN_ROWS`). `bench_qr_scaling` added as coverage.
  - SVD factor-path U/V accumulation (`bidiagonal/reduce.rs`, `apply_reflectors_right`)
    — **shipped.** Both the `dot` reduction and the paired axpy converted (`dot_slice`
    + `axpy_slice`): the slices are full-dimension (long), so per the meta-pattern the
    axpy pays here too (unlike QR). **−52% / −49% / −38% at n=64/128/192**
    (`bench_svd_scaling`, ~1.6–2.1×; 13 SVD tests pass). Confirms the meta-pattern's
    "provably long" axpy exception.
  - udu weighted-dot (`udu/decompose.rs`) — **shipped.** Hoisted the loop-invariant
    `w[k] = u[j][k]·d[k]` (shared by the pivot `dj` and every `u[i][j]`) and reduced
    both through `dot_slice`. Stacks the algorithmic O(n³) recompute drop with the
    SIMD reduction: **−44% / −62% / −69% at n=64/128/256** (`bench_udu_scaling`,
    ~1.8–3.2×; 3 UDU tests pass) — the largest per-decomposition win so far.
  - Secondary remaining (lower-traffic): full_piv_lu / bunch_kaufman trailing
    updates are axpys (LU-style long slices — profile against the meta-pattern
    before converting); col_piv_qr pivot-norm down-dating (a different, non-SIMD fix).
- **Cross-crate lead (hermes):** the CSR SpMV *scalar remainder*
  (`hermes-simd-core/src/sparse/spmv.rs:149`) re-checks the gather bound
  `x[cols[j]]` that the SIMD body 8 lines above already trusts (unchecked gather on
  the `Validated<Csr>` invariant). Short rows (nnz < LANE_COUNT) run wholly through
  it — ~10-30% on short-row SpMV via `get_unchecked` (foundational: backs every
  sparse solver). The `SellP` fallback (`spmv.rs:341`) has the same shape.
  hermes/eunomia f32/f64 SIMD hot paths are otherwise verified hand-written quality
  (hardware FMA, 4-way accumulators, bounds-check-free inner loops, F16C for f16).

## 2026-07-20 SpMV Bounds-Check Elision (Krylov Kernel)

- **Finding:** `spmv_slice_into` (the CSR matrix–vector kernel every Krylov
  iteration runs) indexed `values[p]`/`col_indices[p]` by a range whose bound
  `row_ptr[i+1] ≤ nnz` the compiler cannot prove, so each nonzero carried three
  bounds-check branches (plus the data-dependent `xs[col]` gather). Profiling a
  banded 7-point-stencil SpMV: 0.46 ns/nnz (n=4096, L2), 0.57 ns/nnz (n=65536,
  L3), 1.27 ns/nnz (n=1<<20, DRAM ≈ 12.6 GB/s — well below bandwidth, i.e.
  ILP/latency-bound, not bandwidth-saturated → real headroom).
- **Resolution (shipped):** iterate rows through `row_ptr.windows(2)` zipped with
  `y`, slicing each row's value/column runs, so `O(nnz)` element checks collapse
  to one `O(nrows)` slice check. Same nonzero traversal order → bitwise-identical
  results (pure refactor; `spmv_matches_closed_form_and_overwrites_output`
  unchanged). Restoring prefetch/ILP: **−14% (n=4096) / −19% (n=65536) / −27%
  (n=1<<20, wider CI 19–34%)** (clean-host criterion vs `spmv_pre`, p=0.00); the DRAM-bound case rises
  ~12.6 → ~18 GB/s. (A first reading showed −39% at n=1<<20; the clean re-measurement's
  −27% with its wider CI is the figure of record — memory-bound throughput has
  ~15% run-to-run variance on this host.)
- **Residual (measured — no clear win, not pursued):** the last per-nonzero check
  is the data-dependent gather `xs[col]`. The CSR invariant (`col < ncols`,
  enforced by `from_parts` and every constructor) plus `spmv_into`'s
  `xs.len() == ncols` check prove it in-bounds, so `xs.get_unchecked(col)` is
  sound. Measured twice: both DRAM-bound runs were contaminated by concurrent
  workspace builds saturating memory bandwidth (apparent +835%/+17% is contention,
  not code), but the **cache-resident n=4096 case — least bandwidth-sensitive —
  showed no change (p=0.29)**, indicating the residual check is not the limiter.
  Per "escalate to `get_unchecked` only on a *measured* shortfall," the unchecked
  gather is **not shipped** — no demonstrated benefit, and it adds an `unsafe` +
  miri burden. The safe elision is the optimum for this format. **Lesson:**
  memory-bound benchmarks are invalid under concurrent builds — gate measurement
  on a quiet host (rustc/cargo process count ≈ 0).
- **Blocked lever:** narrowing `col_indices`/`row_ptr` from `usize` to `u32`
  would halve index traffic (the dominant term for DRAM-bound SpMV: 8 B index vs
  8 B value per nonzero), but it is a public-API format change on `CsrMatrix`
  that collides with a peer's in-flight sparse-LU/SpGEMM work on the same format.
  Deferred until that settles ([major], needs an ADR).
- **Sibling (CSC, shipped):** `csc_spmv` carried the same unelided per-nonzero
  `values[p]`/`row_indices[p]` checks around a scatter-add. Same elision (slice
  each column's runs, zip with `col_ptr.windows(2)`; `y.fill` for zeroing) —
  **−24% (n=4096) / −16% (n=65536)** (criterion, `bench_csc_spmv`, clean at
  cache/L3). The gain is larger than CSR's because the residual per-nonzero work
  is a costlier `y[i]` scatter, so its bounds check dominated more. The DRAM-bound
  size could not be measured cleanly (concurrent-build contention); the change
  only removes work, so it cannot regress.

## 2026-07-20 Blocked LU Cache-Resident Regression

- **Finding:** `lu_decompose` is unblocked (BLAS-2, rank-1 SIMD `axpy` trailing
  update). Profiling: LU @256 = 988 µs vs matmul @256 = 404 µs — 2.4× the time at
  ⅓ the FLOPs (~7× lower FLOP rate), the classic BLAS-2 vs BLAS-3 gap.
- **Investigated:** implemented a right-looking blocked (BLAS-3) LU (64-column
  panel factored unblocked, unit-lower solve for `U12`, trailing update
  `A22 −= L21·U12` via matmul). Correct — `P·A = L·U` reconstruction verified at
  n=200 — but **slower at the tested sizes**: LU @256 988 µs → 1.65 ms, @512
  neutral (criterion, non-overlapping CIs at 256).
- **Cause:** this host's 36 MiB L3 keeps LU matrices cache-resident to n ≈ 1200
  (`3·n²·8 < L3`), so the unblocked SIMD `axpy` runs at cache bandwidth and the
  blocked version's overhead (panel-extraction copies, per-panel allocations,
  small rectangular matmuls) dominates. Blocking's cache-reuse benefit only
  materializes once the matrix exceeds the LLC.
- **Decision:** reverted — never ship a regression. Kept the `lu_scaling`
  benchmark and a large-`n` `P·A=L·U` reconstruction test as coverage. A future
  blocked LU should (a) gate on `working_set > l3_bytes` (the cache-aware
  threshold already used by the parallel policy) so it never regresses
  cache-resident sizes, (b) eliminate the trailing-update copies via matmul into
  strided views, and (c) verify the win at `n` past the LLC on a quiet host.

## 2026-07-18 Eunomia 0.4 Provider Refresh

- **Resolution:** the lock advances from Eunomia 0.2.0 `6f431f2d` to 0.4.0
  `49dc115`, so Leto consumes the canonical sub-byte conversion kernel and
  corrected reduced-format constants from Eunomia's default branch.
- **Evidence tier:** dependency-resolution identity plus warning-denied
  all-target/all-feature Clippy, configured Nextest 593/593, doctests 9/9,
  and warning-denied rustdoc.
- **Residual:** this refresh changes no Leto source or public API. External
  nalgebra/ndarray test and benchmark oracles remain tracked by
  `LETO-EXTERNAL-ORACLE-1`.

## 2026-07-18 Eunomia Complex Oracle Ownership

- **Finding:** commit `0178665` restored a workspace-level `num-complex`
  dependency while `leto-ops` test oracles still imported that representation
  directly, recreating a second complex vocabulary beside Eunomia.
- **Resolution:** bind migration, eigenvalue, and Schur test values
  to `eunomia::{Complex, Complex32, Complex64}` and delete the restored direct
  dependency. Direct manifest/source and production graph residue are zero.
- **Evidence tier:** compile-time type ownership; warning-denied
  all-target/all-feature Clippy; Nextest 305/305; doctest 8/8;
  warning-denied rustdoc; and 196/196 applicable SemVer checks.
- **Tracked residual:** external nalgebra/ndarray test and benchmark oracles
  remain in 14 and six files respectively. They do not enter the production
  graph. `LETO-EXTERNAL-ORACLE-1` requires equivalent independent evidence
  before their removal and deletes obsolete comparison benchmark rows.

## 2026-07-17 CFDrs Sparse Direct Factorization Gap

- **Open upstream item**: `LETO-SPARSE-DIRECT-1`.
- **Observed provider surface**: Leto 0.38 owns CSR storage, sparse products,
  CG, and GMRES, but exposes no sparse direct factorization or reusable sparse
  LU factors.
- **Consumer requirement**: CFDrs calls `DirectSparseSolver` only after its
  GMRES tiers stagnate, break down, or exhaust their iteration budget. Replacing
  that tier with another GMRES invocation removes failure-mode independence;
  retaining `rsparse` is therefore the correct boundary until Leto supplies a
  real direct implementation and CFDrs passes differential conformance.
- **Ownership decision**: sparse factorization belongs with Leto's CSR
  representation. A CFDrs-local wrapper, dense materialization, or iterative
  fallback would preserve the dependency/API gap instead of closing it.
- **Required evidence tier**: authoritative algorithm specification plus
  native-precision generic implementation, value-semantic and differential
  tests, and the downstream direct-after-GMRES contract regression. This audit
  records the gap; it does not claim the algorithm is implemented.

## 2026-07-15 provider default-branch convergence

Leto retained revision-qualified and path-patched first-party dependencies,
which created duplicate source identities for downstream Hephaestus and Apollo
graphs. The manifest now follows Mnemosyne, Moirai, Hermes, Eunomia, and Themis
default branches. `leto`/`leto-ops` focused fmt, warning-denied Clippy, locked
nextest, and rustdoc gates pass; the locked provider-duplicate scan is empty.
Evidence tier: locked dependency-resolution plus value-semantic package tests.
Residual downstream work: Hephaestus and Apollo lock convergence.

## Layer Boundary Decision (proposed, [arch])

Leto owns the non-differentiable array substrate: layout/strides, storage,
views, slicing, broadcasting, elementwise binary/unary math, reductions,
matmul (incl. batched), shape ops (concat/pad/split), dense linear algebra,
scaled dot-product attention on CPU, and narrow CPU CSR sparse-dense parity
kernels. Coeus owns autodiff, NN orchestration, optimizer fusion, and backend
selection; Hephaestus owns accelerator attention. Apollo owns transform
kernels. FFT stays in Apollo; Coeus already routes `fft_1d` there.

## B. Gaps vs nalgebra (linear algebra)

Apollo's nalgebra removal is complete; remaining gaps are forward-looking
for Coeus/consumer needs, not blocking any current consumer. nalgebra's
documented decomposition surface includes Schur/Hessenberg, symmetric
eigendecomposition, SVD, LU, QR, and Cholesky; Leto only admits the subset with
a named Atlas consumer driver.

| Gap | nalgebra counterpart | Status |
| --- | --- | --- |
| Symmetric eigensolver | `SymmetricEigen` | **Closed** — `symmetric_eigen_jacobi` (+ tolerance variant), generic over `T: RealScalar`, native precision, Jacobi rotations |
| Symmetric eigenvalues-only path | `SymmetricEigen::eigenvalues` / eigenvalue access | **Closed in 0.14.0** — `symmetric_eigenvalues_jacobi` (+ tolerance variant) shares the Jacobi diagonalization kernel through a monomorphized `RotationTarget`; the eigenvalues-only path uses a zero-sized no-vector target and avoids eigenvector storage |
| LU / solve / inverse / determinant | `LU`, `try_inverse` | **Closed** — `lu_decompose`, `solve`, `det`, and `inv`, generic over `T: RealScalar`; CFDrs dense solver driver |
| QR + least squares | `QR` | **Closed** — Householder `qr_decompose` and `solve_least_squares`; CFDrs least-squares driver |
| Cholesky | `Cholesky` | **Closed** — SPD `cholesky_decompose` and solve; CFDrs SPD driver |
| Thin full-rank SVD | `SVD` subset | **Closed; performance-updated in 0.35.0** — `svd_decompose` routes to bidiagonal QR (`svd_via_bidiagonal`) for tall/square/wide full-rank matrices and rejects rank-deficient inputs explicitly |
| Rank-deficient singular values | `SVD::singular_values` subset | **Closed; accuracy-updated in 0.34.3** — `singular_values` uses implicit-shift bidiagonal QR, so it avoids `AᵀA` and returns zero/tiny singular values without squaring the condition number |
| Full rank-revealing SVD / pseudoinverse | `SVD`, pseudo-inverse helpers | **Closed in 0.20.0** — ADR 0005 one-sided Jacobi SVD (`svd_rank_revealing`) plus rank-deficient `pinv`; verifies reconstruction, orthonormal `V`, nalgebra singular-value/pseudoinverse parity, and Moore-Penrose identities |
| Norms (L1/L2/Frobenius) | `norm`, `norm_squared` | **Closed** — `NormKind` ZSTs with `norm_l1`, `norm_l2`, and `norm_max` |
| Non-symmetric eigenvalues | `eigenvalues`, `complex_eigenvalues` | **Closed in 0.20.0** — ADR 0006 shifted complex QR after Hessenberg reduction; verifies exact spectra and nalgebra `complex_eigenvalues` parity |
| Real Schur vectors/form | `Schur` | Open — [major], eigenvalue spectrum is delivered; Schur vectors/quasi-triangular form require a consumer driver |
| UDU / LDLᵀ | `UDU` | **Closed in 0.21.0 for unpivoted UDU** — `udu_decompose` / `MatrixDecompose::udu`; verifies reconstruction, determinant/solve/inverse parity, and zero-pivot rejection. Pivoted Bunch-Kaufman remains open |
| Small fixed-size matrix/vector types | `Matrix3`, `Vector3` | **Closed for current consumer need** — Gaia/Kwavers driver added `FixedVector`, `FixedMatrix`, `Point3`, `Vector3`, `UnitVector3`, and `Isometry3`; CFDrs driver added Serde-backed geometry values for serialized velocity/parameter storage; broader nalgebra geometry remains consumer-driven |

Policy: linalg routines enter leto-ops only with a named consumer driver and
a differential oracle (ndarray-linalg/nalgebra as dev-dependency oracle, per
the existing ndarray-oracle pattern).

Driver-attribution correction (2026-06-15): the LU/QR/Cholesky/SVD "CFDrs
…driver" attributions above are aspirational, not actual. CFDrs (HEAD `0f578e1a`)
does **not** depend on `leto`/`leto-ops` today — it uses `nalgebra` 0.33 and
`ndarray` directly. These decompositions are validated by the nalgebra/ndarray
differential oracle (`oracle_parity.rs`) and stand on general-parity grounds;
they are not currently exercised by a live Atlas consumer. A real CFDrs
migration to leto dense linalg remains an unstarted, separately tracked item.

## D. Residual Risk Register

Update 2026-07-04 (layout serde rank gap). `Layout<N>` no longer derives
Serde over `[usize; N]`/`[isize; N]`; it serializes slices and validates
decoded vector lengths against `N` before rebuilding arrays. Evidence tier:
compile-time validation plus a rank-33 value-semantic serde roundtrip test.
Downstream Kwavers `kwavers-gpu` WGPU/CUDA-provider feature checks now compile
through the fixed Leto provider graph.

Update 2026-07-05 (CR-4 scalar SSOT rebind complete). The `leto_ops::Scalar`
trait is now bound as `pub trait Scalar: NumericElement` with only `from_usize`
and default-bodied slice kernels retained. `RealScalar: Scalar + FloatElement`.
The local branch was rebased onto `origin/main` (PR #30, 47 commits ahead),
resolving merge conflicts in `scalar.rs`, `lib.rs`, `array.rs`, and
`sparse/mod.rs`. The old standalone `Scalar` methods (`ZERO/ONE/add/sub/mul/div/
bitand/bitor/bitxor/count_ones/to_f64`) are all inherited from `NumericElement`.
`RealScalar` inherits transcendental methods from `FloatElement`. No Leto
compatibility shims were added. Evidence tier: type-level supertrait encoding
plus empirical package verification. `rustup run nightly cargo check -p
leto-ops --all-features`, `rustup run nightly cargo fmt --package leto-ops
--check`, `rustup run nightly cargo clippy -p leto-ops --all-targets
--all-features -- -D warnings`, and `rustup run nightly cargo nextest run -p
leto-ops --all-features` (271/271) pass. Clippy also reports the pre-existing
upstream `hermes-simd-core::sparse::ValidatedData::new_unchecked` dead-code
warning while exiting successfully for the `leto-ops` gate. Downstream consumer
verification (kwavers-math, cfd-math, ritk-registration) pending — consumers
that explicitly name removed Leto UFCS items should import Eunomia traits
directly.

Update 2026-07-02 (scalar SSOT audit). Leto-side scalar ownership is narrowed:
`leto_ops::Scalar` now extends `eunomia::NumericElement`, and `RealScalar`
extends `eunomia::FloatElement`. Evidence tier: type-level supertrait encoding
plus value-semantic tests. Residual downstream fallout is deliberately not
handled by Leto compatibility shims. Apollo source audit found no explicit
removed Leto scalar UFCS references in the checked tree. Coeus imports
`leto_ops::Scalar`/`RealScalar` in its Leto dispatch layer; explicit
`<T as Scalar>::from_f64` hits found by source search are Coeus' own scalar
trait, not Leto's. If Apollo/Coeus later hit `<T as leto_ops::Scalar>::ZERO`,
`ONE`, or `<T as leto_ops::RealScalar>::from_f64`, update the consumer to use
`eunomia::NumericElement` / `eunomia::FloatElement` directly. Follow-up
2026-07-02: platform-sized scalar support moved upstream into Eunomia's sealed
`NumericElement` primitive set, and Leto re-enabled `Scalar` for `isize`/`usize`
through that supertrait. Remaining consumer fallout still belongs in Apollo or
Coeus imports/call sites, not in Leto aliases or forwarding impls.

Update 2026-06-23 (matmul offset-routing audit). Deep safety/contention/memory
audit of the highest-unsafe-density paths (`view.rs` aliasing, storage
exception-safety, `matrix.rs` parallel matmul). Conclusions:
- **Storage (`infrastructure/storage/mnemosyne.rs`): clean.** No leak, double-free,
  UAF, misalignment, ZST/zero-length mishandling, or overflow gap; the
  `MnemosyneInitGuard` drops the initialized prefix and frees exactly once with
  `mem::forget` on success. One non-defect noted: the `if !ptr.is_null()` guards
  are dead branches (`allocate_raw` returns `dangling()` for size 0 and panics on
  failure, never null) — correctness-clarity only, left as-is.
- **`matrix.rs` parallel matmul: no data race on output** — every `for_each_index`
  task writes disjoint output rows/batches (`validate_matmul` rejects zero-stride
  output aliasing). Two real perf/memory defects found and **fixed this cycle**:
  (1) batched/offset-subview matmul fell to the allocating fallback because the
  routing predicates pinned `offset == 0`; relaxed to `is_c_dense`/`is_f_dense`
  (kernels already honor the layout offset) → no per-batch scratch/copy-back,
  fast kernels run; (2) per-batch `Mutex` poll replaced with a relaxed
  `AtomicBool` early-out.
- **RESIDUAL RISK / follow-on [patch]: CLOSED 2026-06-23.** `batched_matmul`'s
  parallel closure previously materialized `from_raw_parts_mut(out_ptr, out_len)`
  over the **full** output buffer per task — runtime-disjoint writes but UB under
  Stacked/Tree Borrows (N concurrent full-range `&mut`). Fixed: each task now
  borrows only its batch's physical span (`Layout::min_max_offsets` → `[lo, hi]`)
  with the offset rebased into that sub-slice, so concurrent `&mut` slices never
  overlap. A disjointness guard (`|out_batch_stride| ≥ per-matrix bounding span`,
  plus non-empty matrices) gates the parallel path; an interleaved-batch output
  (batch stride < span → overlapping bounding boxes) falls through to the
  sequential loop, which reborrows one batch at a time and is unconditionally
  sound. Evidence tier: Tree-Borrows soundness reasoning (each task's `&mut`
  range is provably disjoint) + value-semantic tests — new interleaved-output
  (vs C-contiguous reference) and empty-output boundary tests, plus the batched
  differential/parity oracles vs ndarray (407 workspace tests). Miri remains
  unavailable in this Windows env (moirai inline-asm/platform), so soundness is
  by reasoning, not a Miri run. A full-surface paranoid sweep confirmed this was
  the **only** full-buffer-`&mut`-per-task site: every other parallel kernel
  (`parallel_dot/cc/outer`, row-blocked, chunked map) builds disjoint per-row /
  per-block / per-chunk slices.
- **Hygiene:** removed the stale orphaned `target_ag/` second target tree (1.5 GiB,
  5 days untouched, gitignored, not the configured target dir) — it had filled
  the disk and violates the single-`CARGO_TARGET_DIR` rule.

Update 2026-06-15 (v0.24.0): §A indexed zip parity, the Stage A1
consumer-driven nalgebra surface, Stage C2 dense norm SIMD coverage, and
Stage C3 unary/binary/zip column-walk line micro-tiling are closed through
symmetric eigenvalues-only, LU, QR, Cholesky, norms, full-rank thin SVD,
rank-deficient singular values, rank-revealing SVD/pseudoinverse,
non-symmetric eigenvalues, Hessenberg/bidiagonal/full-pivot/column-pivot
reductions, unpivoted UDU, variance/std, quantile/median, and
covariance/correlation reductions, Hermes-backed dense reductions, and
cache-line tiled strided elementwise traversal. The optional themis topology
dependency is wired through `leto_ops::CacheGeometry`; dense matmul now has a
measured fixed row-block kernel backed by Hermes fused multi-row AXPY;
reverse-last-axis whole-array reductions now borrow unit-stride physical row
slices. Current ndarray/nalgebra oracle tests cover LU solve/determinant/
inverse, symmetric eigenvalues, Cholesky lower factors, singular values, and
reverse-last-axis reductions. Criterion oracle comparison shows reverse
reductions faster than ndarray, but dense matmul is still slower than
ndarray/nalgebra: Leto 17.430 µs / 108.98 µs / 1.0631 ms for
64x64/128x128/256x256 vs ndarray 8.4923 µs / 66.527 µs / 495.95 µs and
nalgebra 8.7752 µs / 62.935 µs / 505.35 µs. Topology-adaptive tile sizing and
non-unit truly strided reductions remain open. See CHANGELOG and the two ADRs
in `docs/adr/`. Remaining work is cross-cutting: Apollo internal FFT-kernel
migration, the themis-0.9 re-pin cascade, dense matmul oracle performance
parity, Schur vectors, pivoted symmetric-indefinite factorization, matrix
functions, and any consumer-driven fixed-size/geometry decisions.

## Leto rank-deficient singular-values parity [patch]
- Performed: split `leto-ops::singular_values` from the full-vector `svd_decompose` path. Singular-values-only now diagonalizes the smaller Gram matrix and maps near-zero eigenvalues to zero singular values for finite rank-deficient inputs.
- Architecture effect: Leto matches the common nalgebra singular-values surface for rank-deficient matrices without fabricating null-space singular vectors. `svd_decompose` keeps explicit rank-deficient rejection until a rank-revealing SVD contract exists.
- Evidence tier: value-semantic tall/wide rank-deficient singular-value tests. No machine-checked proof was performed.

## Leto wide thin SVD parity [patch]
- Performed: generalized `leto-ops::svd_decompose` and `singular_values` from tall/square full-column-rank inputs to all full-rank thin SVD shapes. Wide full-row-rank matrices now use `A A^T`, then derive right singular vectors via `V = A^T U Σ^-1`.
- Architecture effect: Leto closes the current wide-matrix SVD nalgebra-parity gap without a second API or downstream Apollo-specific adapter. Rank-deficient inputs remain explicit errors until a rank-revealing SVD contract is implemented.
- Evidence tier: value-semantic reconstruction and orthonormality tests. No machine-checked proof was performed.

## Leto 64² singular-values disparity — root-caused as algorithmic (ADR 0012)
- Performed: profiler-free phase attribution of `singular_values` 64² vs nalgebra
  (bidiag 1.72×, values-sweep 2.25×, total 1.92×). Ruled out — by direct
  experiment — convergence (92 Givens steps = 1.44/value), bounds checks
  (`get_unchecked` left timing unchanged → already elided), trait dispatch
  (f64 ops `#[inline(always)]`), and hermes `dot` dispatch (`target-cpu=native`;
  the values sweep is pure-scalar `VEC=false` and never calls `dot`).
- Root cause (corrected): nalgebra 0.32 uses the **same** implicit-shift Givens
  sweep (verified in its `svd.rs`), so the gap is a per-step/per-element
  **implementation constant**, NOT algorithmic (an earlier note wrongly framed it
  as Givens-vs-dqds — see ADR 0012's correction). dqds (0√+1÷ vs Givens' 2√+2÷)
  is a *theoretical* lever that would beat both. A full implementation (block
  splitting + in-place sweep + rank-deficiency gate to Givens) was built and
  passes all 17 differential tests, but a **clean same-session A/B** (criterion,
  nalgebra-anchored) measured it at **−1.3% vs Givens — a statistical tie, no
  win**: the dmin-fraction shift plateaus at ~300 sweeps (vs Givens' 92 steps),
  cancelling dqds's per-element saving. A win needs the full dlasq4 cased shift
  (~130 sweeps), and even then bidiag (1.72× nalgebra, unchanged) caps parity.
  Reverted per the ship-only-on-measured-win DoD; scoped [major] in ADR 0012.
- Residual risk: 64² values-only stays ~1.9× nalgebra; the per-step constant was
  narrowed (convergence/bounds/dispatch/inlining ruled out) but not isolated to a
  single cause. Correctness and accuracy are unaffected (Givens path retained).
- Evidence tier: criterion + differential suite + phase/step attribution.

## 2026-08-13 leto-ops criterion baselines (relocated from benchmark_results.md)

Relocated from the repository root: measurement baselines are audit state, not
a root-level report artifact. Content is unchanged; headings are demoted one
level to nest under this section.

### Harness and methodology

Harness: `crates/leto-ops/benches/kernels.rs` (`cargo bench -p leto-ops`).
Methodology: Criterion, sample_size 10 in the current harness, median + 95% CI;
pinned deterministic inputs (no RNG); default features; f64. Historical rows
retain their recorded sample size. Machine class: Windows 11 x86_64 dev
workstation (AVX2-class). These baselines gate optimization work (Atlas ADR
0002 leto slice): a statistically significant regression in a touched kernel
blocks merge, and no change is labeled an optimization without a recorded
comparison.

### Current state (fused multi-row AXPY, 0.19.7, 2026-06-13)

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

### Topology policy evaluation (2026-08-08)

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

### Contiguous and non-unit-stride coverage (2026-07-23)

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

### Non-unit-stride reduction audit (2026-07-23)

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

### Dense matmul parity audit (2026-07-23)

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

### Historical oracle comparison gate (ndarray / nalgebra, 0.19.7, 2026-06-13)

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

### Measured optimization history

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

### Rejected optimization candidates (do not retry without a changed model)

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

### Open measured targets

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
