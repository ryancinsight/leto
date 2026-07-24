# Leto Gap Audit: ndarray / nalgebra Replacement for Atlas

## 2026-07-23 Oracle ownership reconciliation

- **Finding:** the backlog still claimed that legacy `ndarray`/`nalgebra`
  oracle code should be removed, but current parity examples and differential
  tests actively use those crates as independent value-semantic references.
  `ndarray`, `ndarray-rand`, and `nalgebra` are declared only as dev
  dependencies in the provider manifest.
- **Evidence:** `cargo tree --locked -p leto-ops --no-default-features
  --edges normal` contains no `ndarray`, `ndarray-rand`, or `nalgebra` edge.
  The dev graph resolves `ndarray 0.16.1`, `ndarray-rand 0.15.0`, and
  `nalgebra 0.35.0`. Active source references are limited to seven files:
  `crates/leto-ops/benches/kernels.rs`, the two parity examples, and four
  differential/parity test modules.
- **Resolution:** close `LETO-EXTERNAL-ORACLE-1` as an ownership
  reconciliation. Retain the independent dev-only oracle boundary and its
  useful benchmark rows; deleting it without equivalent analytical or
  published-reference coverage would weaken verification and violate the
  evidence requirement.
- **Limit:** this proves dependency direction and current oracle ownership,
  not complete parity across every provider operation. New removal work needs
  an explicit replacement oracle per operation family.

## 2026-07-23 Dense matmul parity audit

- **Finding:** the historical 0.19.7 oracle rows marked dense matmul as slower
  than ndarray at 64×64, 128×128, and 256×256. A quiet-host rerun against the
  current default-feature release path changed that conclusion: Leto measured
  `23.597 µs` [17.659, 29.520], `123.63 µs` [117.41, 130.97], and `233.60 µs`
  [202.28, 253.87] at those sizes; ndarray measured `12.770 µs` [11.437,
  14.460], `113.07 µs` [104.34, 120.96], and `952.54 µs` [935.68, 981.42].
- **Resolution:** no production change is justified by the current evidence.
  The existing `is_parallel_beneficial` threshold selects the parallel dense
  path at 64×64; a focused 20-sample rerun measured `23.597 µs` with parallel
  execution versus `27.483 µs` [26.478, 28.271] with parallelism disabled.
  At 128×128 and 256×256, the serial path measured `223.69 µs` [222.72,
  224.43] and `1.8522 ms` [1.8208, 1.9032], respectively. The measured
  default path is therefore the better current policy across the tested sizes.
- **Profile limit:** `cargo flamegraph` could not collect samples on this
  Windows session because `dtrace` was unavailable and its `blondie` fallback
  required administrator rights. The result is controlled benchmark and source
  dispatch evidence, not a call-stack profile or a topology proof.
- **Verification:** locked Criterion oracle runs used the same deterministic
  f64 inputs and current release profile; a no-default-feature Leto run
  isolated the serial comparison. No production source changed. The claim is
  closed as an evidence-only audit; future tile or packing work requires a
  working profiler and a changed kernel hypothesis.

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

## 2026-07-22 Runnable Migration Evidence

- **Closed evidence gap:** `leto-ops` now owns runnable `ndarray_parity` and
  `nalgebra_parity` examples in addition to its focused integration tests and
  Criterion benchmarks. The examples are deterministic and input-sensitive;
  they report measured error magnitudes rather than boolean-only success.
- **Numerical contract:** elementwise equal-order operations require exact
  equality; independent reductions use `2γₙ Σ|term|`; the manufactured Poisson
  solve uses normalized backward error, the exact discrete sine eigenmode, and
  the exact infinity-norm condition number
  `κ∞(A) = 2 maxᵢ i(n + 1 - i)` for forward bounds.
- **Architecture effect:** ndarray and nalgebra remain dev-only differential
  oracles. Production dependency ownership and kernel implementations are
  unchanged. The examples contain no one-shot timing claim; performance
  evidence remains owned by the controlled Criterion suite.
- **Evidence limit:** these examples cover a representative migration workflow,
  not the full provider surfaces. `docs/completeness/parity_matrix.md` and its
  focused contract suites remain the completeness SSOT.

## 2026-07-21 Public ndarray Compatibility Boundary

- **Finding:** `leto` contradicted its production dependency policy through an
  optional `ndarray-compat` feature. That feature exposed a public third-party
  re-export and six conversion implementations, including two unsafe raw-slice
  reconstructions. No live Atlas manifest or Rust caller uses the feature or
  conversion module; Apollo commit `324f380` consumes native Leto arrays and its
  resolved graph has no Rust `ndarray` package.
- **Resolution:** remove the feature, optional dependency, module, re-export,
  conversion-only integration tests, and the conversion fixture from Leto Ops.
  `ndarray` remains a dev-dependency oracle. Consumer language/FFI boundaries
  construct native Leto arrays directly; no replacement adapter is introduced.
- **Coverage preservation:** canonical Leto suites retain constructor/storage,
  transpose, broadcast, axis mutation, signed-stride slicing, layout bounds,
  reshape, and logical-order materialization coverage. Configured Nextest passes
  266/266 after the conversion-only suite is removed. ADR 0017 records the
  public migration and ownership decision.
- **Evidence limit:** tests prove retained Leto value semantics; the production
  dependency scan and Cargo graph prove boundary removal. Neither is a runtime
  performance measurement.

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

## 2026-07-20 Typed Laplacian Ownership

- **Finding:** Hephaestus owned a WGPU Laplacian, but Leto exposed no matching
  CPU contract. CFDrs consequently kept a live CPU formula and another local
  test oracle, and its CPU/GPU solver operators selected opposite signs.
- **Resolution:** Leto owns the validated dimensional contract and Leto Ops
  owns the CPU evaluation; Hephaestus consumes the same boundary and polarity
  types. Consumer formulas are deleted in the paired migration.
- **Evidence tier:** generic `f32`/`f64` closed-form value regression plus
  compile-time type unification and focused package gates.
- **Residual:** three-dimensional and variable-coefficient CFD operators are
  distinct contracts and remain outside this two-dimensional uniform-grid
  slice.

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

## 2026-07-20 Normal RNG at Parity (Ziggurat)

- **Finding (closed):** `normal_with_seed` used Box-Muller — even after the
  sine-half recovery (570 µs/64k `f64`) it trailed ndarray's Ziggurat ~3.9×. This
  was the last profiled path on which leto-ops lost to ndarray.
- **Resolution:** replaced Box-Muller with the Ziggurat method (Marsaglia & Tsang
  2000, 128 layers). The `kn`/`wn`/`fx` tables are reconstructed from the published
  `r`/`v` constants via the equal-area recurrence (Burkardt's reference form).
  64k `f64` normals **1108 µs (0.39.0) → 210 µs**, at parity with ndarray (212 µs).
  Correctness verified against the analytic normal — first four moments, tail
  probabilities `P(|Z|>1..4)`, and a 200-bin chi-squared goodness-of-fit over 10M
  samples (`ziggurat_normal_matches_analytical_distribution`) — with per-seed
  determinism and layout independence preserved. leto-ops now matches or beats
  ndarray on every profiled path.
- **Note:** second per-seed *sequence* change for `normal_with_seed` this release
  cycle (distribution preserved and verified; documented in CHANGELOG).

## 2026-07-19 Parallel Threshold Ignores Arithmetic Intensity

- **Finding:** `leto-ops` gates parallelism on a uniform element-count constant
  (`PARALLEL_THRESHOLD` = 65536 in `map.rs`/`unary.rs`, 32768 in `reduction.rs`)
  irrespective of an operation's arithmetic intensity. Bandwidth-bound binary
  elementwise ops (`add`/`sub`/`mul`/`div`) parallelize the moment the array
  reaches the threshold, where thread-dispatch overhead exceeds any benefit,
  while compute-bound unary `exp` (which correctly wins 9.2× vs ndarray) needs
  it — one gate cannot serve both intensities.
- **Evidence (criterion, noisy host):** `parity_oracle/add_leto_64k` = 43 µs with
  default features vs **14.6 µs** under `--no-default-features --features
  std,mnemosyne-memory,topology` (parallel off) — ~3× slower parallel; serial
  even beats ndarray's 17.6 µs. `PARALLEL_THRESHOLD` (65536) exactly equals the
  benchmark `len` (`1<<16`), so the `>=` guard trips parallelism by one element.
- **Impact:** every bandwidth-bound elementwise op on arrays from ~64k up to the
  parallel-profitable size (several MB, past shared LLC) pays parallel overhead
  for a net slowdown — `add`/`sub`/`mul`/`div` and the bandwidth-bound unary maps
  (negate/abs/copy). Compute-bound transcendentals are unaffected.
- **Fix direction ([minor]/[arch], ADR):** replace the uniform element-count gate
  with an arithmetic-intensity- and cache-aware threshold — parallelize a
  bandwidth-bound op only when its working set (`operands · N · size_of::<T>()`)
  exceeds the shared last-level cache (available through `themis::CpuTopology`),
  so extra cores contribute memory bandwidth; keep the low threshold for
  compute-bound ops. Needs an empirical crossover sweep on a quiet host to
  calibrate and verify.
- **Status:** binary, unary, and scalar paths resolved; reductions measured-fine.
  All bandwidth-bound elementwise ops (`add`/`sub`/`mul`/`div` in `binary_map`;
  `neg`/`abs` via `unary_map_into`; scalar broadcast via `scalar_map_into`) now
  gate on working-set-vs-LLC through `CacheGeometry::l3_bytes()` (cache-derived,
  not a guessed constant); `UnaryOp::COMPUTE_BOUND` keeps transcendentals eager.
  Measured: 64k `f64` `add` 43 → 16 µs, `scalar_map_into` add ~73 → 9.4 µs;
  reductions do not over-parallelize (`sum` @64k serial 3.3 µs ≈ parallel 3.6 µs,
  efficient tree-reduction) so they keep their threshold. Correctness-safe (both
  paths compute identically); 305/305 tests green. **Resolved:** the
  `parallel_crossover` sweep (`add` 512k → 8M, gate vs serial, 36 MiB-L3 host)
  confirms the L3-working-set threshold is correctly calibrated — serial and
  matching the baseline below L3, parallel and 1.26–1.78× faster above it (CIs
  non-overlapping). The cache-residency default is optimal by measurement, not a
  guess; `LETO-PARALLEL-INTENSITY-1` closed.

## 2026-07-18 Raw Reduced-Precision Ownership

- **Finding:** Leto still directly depends on `half` and implements its public
  `ScalarOperand`, `Scalar`, `RealScalar`, and reduced-precision fixtures for
  raw `half::f16`/`half::bf16`, while Eunomia owns the Atlas numeric vocabulary
  and Hermes now exposes only Eunomia reduced-precision SIMD contracts.
- **Decision:** replace the raw public implementations and all in-repo call
  sites with `eunomia::F16`/`Bf16`; delete the direct dependency rather than
  retaining a compatibility implementation. This is a pre-1.0 breaking public
  contract and targets Leto 0.39.0.
- **Evidence required:** compile-time trait coverage, exact reduced-precision
  value tests, source/manifest residue scans, one locked Eunomia/Hermes identity,
  and the full local/remote verification gates.
- **Resolution:** production and test sources now use only Eunomia `F16`/`Bf16`;
  every direct `half` dependency is deleted. The lock resolves one Eunomia 0.5.0
  identity at `c196db5`, one Hermes 0.4.0 family at `c9bbdf8`, and one Moirai
  0.4.0 family at `8a51b2a`. Full all-feature workspace compilation,
  warning-denied Clippy, configured Nextest 593/593, nine doctests, rustdoc,
  no-default-feature compilation, and full formatting pass. Warning-denied
  Clippy exposed one unrelated UDU oracle indexing lint, fixed by iterating
  directly over the right-hand-side values. The peer-owned matrix-trait,
  oracle-parity, and Schur rustfmt-only delta is composed without semantic
  changes. Cumulative code review found no P0/P1 defect; its only P2 evidence
  gap was closed by an exact generic array-scalar contract instantiated for
  `F16` and `Bf16` plus exact `Scalar::from_usize` assertions for both.
- **Delivery:** PR #46 merged as `0afece5`. Leto has no GitHub Actions workflow
  or protected-branch requirement. CodeRabbit reported success but emitted no
  review because its quota was exhausted; RecurseML returned an external
  `ERROR` without a run ID or logs. The merge therefore rests on the complete
  local machine-checked gate and the recorded cumulative code review.
- **Semver evidence:** `leto` and `leto-ops` current and `origin/main` baselines
  build and classify with no required update under the explicit 0.39.0
  pre-1.0 break. `leto-python` extraction reaches a Rust 1.95 rustdoc ICE while
  resolving NumPy's `ToPyArray::to_pyarray` intra-doc link; direct workspace
  rustdoc passes, and this migration changes no Python binding API.
- **Pre-existing supply-chain residual:** Leto has no `deny.toml`, so
  `cargo deny check` rejects the default license/source policy and reports
  existing PyO3 0.23.5 advisories RUSTSEC-2025-0020 and RUSTSEC-2026-0177.
  Re-open as a dedicated Python-boundary dependency upgrade to PyO3 0.29 or
  newer with matching NumPy bindings and value-semantic Python tests.

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

## 2026-07-18 Sparse COO/CSC Conversion Defects

- **Resolved:** the stale duplicate-conversion work no longer builds a
  coordinate HashMap plus multiple temporary vectors. One stable row/column
  ordering and streaming compaction implements both sum and keep-last policies;
  zero sums do not enter CSR storage.
- **Resolved:** `CscArray::from_coo` now owns column-major normalization.
  CSR-to-CSC transpose no longer supplies row-major coordinates to a
  column-major-only constructor, and callers no longer pre-sort redundantly.
- **Evidence tier:** exact unordered-duplicate, zero-sum, CSC column/lookup,
  and transpose regressions; sparse Nextest 18/18; full Leto Nextest 267/267;
  warning-denied all-target/all-feature Clippy; doctest 1/1; warning-denied
  rustdoc; and 196/196 applicable SemVer checks.
- **Provider refresh:** the committed lock now resolves Eunomia 0.2.0
  `6f431f2d`; the former Eunomia-owned `num-traits` graph edge is absent.
- **Open toolchain conformance:** the workspace still declares edition 2021
  and resolver 2. Move to edition 2024/resolver 3 in a dedicated coordinated
  change after every published crate and consumer passes the edition lint and
  SemVer gates.

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

## 2026-07-16 Helios Oriented-Grid Provider Gap

- **Resolved in Leto**: `UnitQuaternion::try_from_rotation_columns` owns the
  finite, orthonormal, right-handed basis validation and branch-stable
  matrix-to-quaternion conversion. The generic `f32`/`f64` suite proves the
  rotation maps the local axes to the supplied columns; exact failures cover
  non-orthogonal, reflected, non-finite, and invalid-tolerance inputs.
- **Evidence tier**: value-semantic tests plus locked check, warnings-denied
  Clippy, 249/249 configured Nextest, doctest, warning-clean Leto rustdoc, and
  repository-baseline SemVer checks for `leto`/`leto-ops`.
- **Residual external limitation**: `cargo semver-checks -p leto-python
  --baseline-rev origin/main` cannot build rustdoc on Rust 1.95 because
  NumPy 0.23 triggers `collect_intra_doc_links` ICE. `leto-python` is already
  `doc = false`; no source workaround is introduced for a compiler defect.
- **Downstream sequencing**: Helios consumes this contract after RITK exposes
  the named `ImageOrientationPatient` DICOM tag from its currently occupied
  provider lane; Helios owns the DICOM-specific tolerance and oblique-series
  regression.

## 2026-07-15 provider default-branch convergence

Leto retained revision-qualified and path-patched first-party dependencies,
which created duplicate source identities for downstream Hephaestus and Apollo
graphs. The manifest now follows Mnemosyne, Moirai, Hermes, Eunomia, and Themis
default branches. `leto`/`leto-ops` focused fmt, warning-denied Clippy, locked
nextest, and rustdoc gates pass; the locked provider-duplicate scan is empty.
Evidence tier: locked dependency-resolution plus value-semantic package tests.
Residual downstream work: Hephaestus and Apollo lock convergence.

## 2026-07-04 CFDrs Sparse Extension CSR Utility Provider Gap

- **Resolved**: `CsrMatrix` now owns diagonal extraction, scalar/value
  scaling, row scaling, column scaling, Frobenius norm, strict diagonal
  dominance, and the diagonal-dominance condition-estimate heuristic.
- **Consumer driver**: CFDrs `cfd-math::sparse::SparseMatrixExt` still exposed
  these operations over `nalgebra_sparse::CsrMatrix`, but the operation logic
  now has a Leto-owned provider target instead of downstream CSR loops.
- **Evidence tier**: provider compile/clippy and empirical nextest plus
  downstream compile/clippy/nextest. In `D:/atlas/repos/leto`, `rustup run
  nightly cargo fmt -p leto-ops --check`, `cargo check -p leto-ops`, `cargo
  nextest run -p leto-ops --test ops_tests sparse --status-level fail` (18/18),
  and `cargo clippy -p leto-ops --all-targets -- -D warnings` passed.
  Downstream CFDrs `cfd-math` fmt/check, focused sparse nextest (18/18), and
  all-target clippy passed.

---

## 2026-07-04 CFDrs AMG CSR Transpose Provider Gap

- **Resolved**: `CsrMatrix::transpose()` now returns a sorted CSR
  representation of `A^T` without dense materialization. The operation counts
  nonzeros per output row, prefix-scans the output row pointers, and scatters
  source entries in source-row order so transposed row column indices remain
  strictly increasing.
- **Consumer driver**: CFDrs AMG restriction construction needs `R = P^T`.
  This provider surface lets the downstream path move off
  `nalgebra_sparse::transpose_as_csc` instead of adding a CFDrs-local CSR
  transpose helper.
- **Evidence tier**: provider compile/clippy/doc and empirical nextest plus
  downstream compile/clippy/nextest. In `D:/atlas/repos/leto`, `rustup run
  nightly cargo fmt -p leto-ops --check`, `cargo check -p leto-ops`, `cargo
  nextest run -p leto-ops --test ops_tests sparse --status-level fail` (16/16),
  `cargo clippy -p leto-ops --all-targets -- -D warnings`, and `cargo doc -p
  leto-ops --no-deps` passed. Downstream CFDrs `cfd-math` fmt/check, focused
  sparse nextest (17/17), focused AMG nextest (6/6), and all-target clippy
  passed.

---

## 2026-07-04 CFDrs AMG CSR Product Provider Gap

- **Resolved**: `leto_ops::spgemm` now computes CSR×CSR matrix products with
  sorted output rows and exact-zero cancellation removal. `CsrRow::nnz` exposes
  row sparsity without requiring consumers to inspect row slices manually.
- **Consumer driver**: CFDrs AMG setup currently relies on
  `nalgebra_sparse` multiplication for Galerkin products (`R * A * P`). This
  provider surface gives the downstream sparse/linear-solver migration a
  Leto-owned target instead of a CFDrs-local CSR multiply.
- **Evidence tier**: provider compile/clippy/doc and empirical nextest.
  `rustup run nightly cargo fmt -p leto-ops --check`, `cargo check -p
  leto-ops`, `cargo nextest run -p leto-ops --test ops_tests sparse
  --status-level fail` (14/14), `cargo clippy -p leto-ops --all-targets -- -D
  warnings`, and `cargo doc -p leto-ops --no-deps` passed.

---

## 2026-07-04 CFDrs Mesh Rotation Provider Gap

- **Resolved**: `FixedMatrix<T, 3, 3>` now multiplies
  `leto::geometry::Vector3<T>` directly. The implementation computes the
  row-major 3x3 geometry-vector product without an identity-element dependency.
- **Consumer driver**: CFDrs `cfd-core::geometry::mesh::MeshOperations::rotate`
  moved from nalgebra `Matrix3<T>` and `Vector3<T>` to Leto fixed geometry while
  the mesh/staggered geometry cone moved scalar contracts to Eunomia.
- **Evidence tier**: provider compile/clippy and empirical nextest, plus
  downstream compile/clippy/nextest and static scans. `rustup run nightly cargo
  fmt -p leto --check`, `cargo check -p leto`, `cargo nextest run -p leto
  --status-level fail` (171/171), and `cargo clippy -p leto --all-targets --
  -D warnings` passed. Downstream `cfd-core` no-default check, full no-default
  nextest (201/201), no-default all-target clippy, and mesh/staggered provider
  scans passed.

---

## 2026-07-04 CFDrs Domain Point1 Provider Gap

- **Resolved**: `leto::geometry` now includes `Point1<T>` for one-dimensional
  domain bounds. Fixed geometry values derive `Eq` conditionally where the
  scalar supports it, matching downstream generic enum derive requirements.
  Leto's `std` and `alloc` features now propagate to serde so direct provider
  checks compile Vec-backed serde surfaces.
- **Consumer driver**: CFDrs `cfd-core::geometry::shapes::Domain` moved
  one-, two-, and three-dimensional point/vector geometry from nalgebra to Leto,
  and the dependent boundary/domain contract moved scalar bounds to Eunomia.
- **Evidence tier**: provider compile/clippy and empirical nextest, plus
  downstream compile/clippy/nextest and static scans. `rustup run nightly cargo
  fmt -p leto --check`, `cargo check -p leto`, `cargo nextest run -p leto
  --status-level fail` (170/170), and `cargo clippy -p leto --all-targets --
  -D warnings` passed. Downstream `cfd-core` no-default check, full no-default
  nextest (201/201), and no-default all-target clippy passed.

---

## 2026-07-04 CFDrs Serialized Owned-Array Provider Gap

- **Resolved**: Owned `Array<T, S, N>`, `VecStorage<T>`, and const-rank
  `Layout<N>` now support serde. `Array` deserialization reconstructs through
  `Array::new`, so malformed layout/storage bounds are rejected instead of
  constructing invalid private fields.
- **Consumer driver**: CFDrs `cfd-core::abstractions::state` moved scalar
  field state from serialized nalgebra `DVector<T>` to serialized
  `leto::Array1<T>` without a downstream compatibility wrapper.
- **Evidence tier**: provider compile/clippy and focused empirical nextest,
  plus downstream compile/clippy/nextest. `cargo fmt -p leto --check`, focused
  `cargo nextest run -p leto
  owned_array_round_trips_shape_and_values_through_serde --status-level fail`,
  and `cargo clippy -p leto --all-targets -- -D warnings` passed. Downstream
  `cfd-core` no-default check, no-default all-target clippy, and focused state
  nextest passed.

---

Audit date: 2026-06-12. Evidence tier: codebase scan of `leto` (0.19.6),
`D:/atlas/repos/apollo`, `D:/atlas/repos/coeus`, current docs.rs pages for
`ndarray 0.17` and `nalgebra`, and upstream Atlas crates. Counterparts:
`ndarray 0.17`, `nalgebra` (already removed from Apollo).

## Consumer Position

- **Apollo** (spectral transforms): migrated to native Leto host arrays at
  commit `324f380`; its manifests and resolved Rust graph contain no `ndarray` or
  `ndarray-compat` dependency edge. Transform APIs expose Leto boundaries, and
  nalgebra is removed (FrFT/GFT eigendecomposition uses
  `leto_ops::symmetric_eigen_jacobi`; GFT adjacency uses `leto::Array2<f64>`).
- **Coeus** (tensor/autodiff, burn replacement): CPU array layer **fully
  consolidated onto leto** (re-verified 2026-06-15 against coeus HEAD
  `037fdd5`; pins leto `d8d34c61`, older than current leto HEAD `723a63c`).
  coeus's CPU `BackendOps` route every array primitive (elementwise, matmul +
  batched, axis reductions, argmax/argmin, cumsum/suffix, concat/pad/split/
  stack, seeded RNG, to_contiguous/reshape/permute, cross-backend transfer,
  from_fn/eye/arange/linspace) through the `coeus-leto` const-rank dispatch shim
  (ADR 0002) into leto/leto-ops, covered by `coeus-leto/tests/contract.rs` and
  `coeus-ops`/`coeus-tensor` `*_leto_diff.rs` suites (coeus workspace 255 tests
  green). Coeus retains its sealed `ComputeBackend`, autodiff, NN kernels
  (conv/pool/attention/optimizers), higher sparse formats/backends, and
  wgpu/CUDA backends. `coeus-core` keeps a dynamic-rank `Layout` and
  `Storage`/`StorageMut` traits that
  `coeus-leto` converts to leto's const-rank views at the boundary — this is the
  intended ADR 0002 seam, not residual duplication.

## Layer Boundary Decision (proposed, [arch])

Leto owns the non-differentiable array substrate: layout/strides, storage,
views, slicing, broadcasting, elementwise binary/unary math, reductions,
matmul (incl. batched), shape ops (concat/pad/split), dense linear algebra, and
narrow CPU CSR sparse-dense parity kernels. Coeus owns autodiff, NN kernels (conv,
pool, attention), optimizer fusion, higher sparse formats/backends, and device
(GPU) backends. Apollo owns transform kernels. FFT stays in Apollo; Coeus
already routes `fft_1d` there.

## A. Gaps vs ndarray 0.16 (Apollo-facing)

Present and verified: const-rank `Array/ArrayView/ArrayViewMut` (+ rank
aliases 1–3), C/F layouts, ndarray-style `SliceArg` slicing, transpose,
broadcast, axis iteration, `zeros`/`ones`/`from_elem`/`from_vec`/
`from_shape_vec`/`from_shape_fn`/`into_vec`, `map_into`/`mapv`/`map`,
`zip_mut_with`, `zip2_mut_with`, `zip3_mut_with`, `zip5_mut_with`,
`indexed_map_inplace`, `indexed_map4_inplace`, `indexed_zip4_mut_with`,
`zip_fold`, checked all-elements `min`/`max`, `indexed_fold`,
`indexed_fold_fortran`, `coordinate_map_inplace`, `CoordinateMapPlan`,
sum/mean/min/max keep-dim axis reductions,
argmin/argmax, 2D matmul, variance/std (all + axis, finite
`ddof`), quantile/median (all + axis, five interpolation strategies),
covariance/Pearson correlation
(rowvar), CoW storage, Mnemosyne storage, and
owned-array convenience methods for consumer migration (`as_slice_mut`,
memory-order mutable slices, `mapv`, `zip_map`, `fill`, `assign`, and
`[usize; N]` indexing), plus rank-1 `usize` indexing and owned-array
`PartialEq`/`Eq` value semantics for Kwavers CPML profile/factor storage.
Owned-array serde includes `Array<T, S, N>`, `VecStorage<T>`, and const-rank
`Layout<N>` with manual rank validation for serde ranks above the fixed-array
impl limit.
Fixed-size provider geometry includes
`FixedMatrix<T, 3, 3>::try_inverse(min_abs_det)` for Gaia/Kwavers FEM
tetrahedral Jacobian inversion, `Vector2<T>` plus generic vector
norm/normalization for CFDrs FVM face geometry, and Serde-backed fixed geometry
values for CFDrs serialized velocity/parameter migration. Special-function unary markers
`ErfOp`/`ErfcOp`/`LgammaOp` are present over the Eunomia real-math SSOT for the
Coeus special-functions lane.

| Gap | ndarray counterpart | Consumer driver | Class |
| --- | --- | --- | --- |
| Contiguous-slice access on views (`as_slice`, `as_slice_mut`, memory-order variant) | `as_slice_memory_order_mut`, `is_standard_layout` | Apollo FFT butterfly kernels require contiguous mutable slices (~20 call sites) | Closed |
| Multi-array zip, fold, indexed mutable map, indexed fold, sparse coordinate map, and `Zip::indexed` | `Zip::from(..).and(..).for_each/fold`, `Zip::from(..).and(..).and(..)`, `Zip::from(..).and(..).and(..).and(..)`, higher-arity `Zip`, `Zip::indexed`, `indexed_iter().fold`, sparse coordinate loops | Apollo precision-downgrade, scaling, position-aware paths; Kwavers FWI pressure second derivative, relative model change, self-adjoint imaging conditions/source injection, sponge test helper, adjoint gradient peak logging, MOFI rigid-Jacobian multi-output fills, and FWI Fortran-order voxel lists | Closed (`zip_mut_with`, `zip_fold`, `indexed_fold`, `indexed_fold_fortran`, `coordinate_map_inplace`, `CoordinateMapPlan`, `zip2_mut_with`, `zip3_mut_with`, `zip5_mut_with`, `indexed_map_inplace`, `indexed_map4_inplace`, `indexed_zip_mut_with`, `indexed_zip2_mut_with`, `indexed_zip4_mut_with`) |
| `mapv_inplace` / in-place unary mutation | `mapv_inplace` | Apollo normalization (1/N scaling) (~5 sites) | Closed |
| Reshape / `into_shape` on contiguous arrays | `into_shape_with_order` | Apollo (low frequency), Coeus `reshape` (required) | Closed |
| Scalar–array elementwise ops (array + scalar, array * scalar) | `&a + 1.0`, `mapv` shortcuts | Apollo scaling, Coeus bias/scale paths | Closed |
| Broadcast-aware binary ops into caller-owned output | broadcasted elementwise ops | Coeus passes `a_layout`, `b_layout`, `c_layout`; Apollo validation and scale paths | Closed |
| std::ops operator impls on arrays/views (`Add`, `Sub`, `Mul`, `Div`, `Neg`) | operator overloads | Ergonomics for both consumers; std-trait integration mandate | Deferred by ADR 0001; current scalar/binary map APIs cover driven cases |
| `concat`/`stack` along axis | `ndarray::concatenate`, `stack` | Coeus `cat()`; Apollo validation builders | Closed (`concat`; `stack` via `InsertAxis` rank helper) |
| Dynamic-rank escape type at I/O boundaries | `IxDyn` | Apollo generic-over-dimension helpers (~30 sites use `Array<T, D>`); Coeus layout is dynamic-rank | Closed (`ArrayD`, `LayoutDyn`, zero-copy rank bridge; ADR 0007 boundary carrier, compute still via const-rank recovery) |
| 1D dot / vector ops | `Array1::dot` | Apollo, Coeus | Closed |
| Elementwise unary math suite (`exp`, `ln`, `sin`, `cos`, `sqrt`, `abs`, `neg`, `powf`, `erf`, `erfc`, `lgamma`) as named ZST ops | `mapv` with std/special float fns | Coeus `UnaryOp` enum and exact-GELU / `torch.special` parity lane | Closed |
| `cumsum` / prefix scans along axis | (ndarray lacks native; Coeus has) | Coeus `cumsum`, `suffix_sum` | Closed (`scan_axis`, `cumsum`, fwd/rev, CumSum/CumProd) |
| Random constructors (uniform/normal, seeded) | `ndarray-rand` | Coeus init (`Xorshift64`, Box-Muller); keep deterministic, seed-based | Closed (`uniform_with_seed`, `normal_with_seed`) |
| Pad / split along axis | (manual in ndarray) | Coeus shape ops | Closed (`pad`, `split`) |
| Batched (rank-3) matmul | (via einsum/manual) | Coeus batched contraction — boundary decision places it in leto | Closed (`batched_matmul`, batch broadcast) |
| Variance / standard deviation | ndarray-stats / ndarray `var` | Array statistics parity | Closed (`var_all`, `std_all`, `var_axis`, `std_axis`; two-pass, finite `ddof`) |
| Quantile / median | ndarray-stats / numpy | Array statistics parity | Closed (`quantile_all`, `median_all`, `quantile_axis`, `median_axis`; five interpolation strategies, NaN/range rejection) |
| Correlation / covariance | ndarray-stats | Array statistics parity | Closed (`covariance`, `pearson_correlation`; rowvar, two-pass centered covariance, exact empty/ddof rejection) |

Non-goals confirmed: conv/pool/attention/optimizer kernels, higher sparse
formats/backends beyond CPU CSR SpMV/SpMM, autodiff — these stay in Coeus. GPU
buffers stay behind Coeus's `ComputeBackend`.

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

## C. Coeus backend integration ([arch]) — COMPLETE (2026-06-15)

The CPU array-kernel consolidation is done (verified against coeus HEAD
`037fdd5`). The plan-of-record integration path resolved as:
1. Leto provides the CPU array kernels (unary suite, broadcast-aware binary into
   caller-owned output, reductions incl. argmax/cumsum, matmul + batched,
   concat/pad/split/stack, seeded RNG). DONE.
2. Coeus routes its CPU `BackendOps` through the `coeus-leto` const-rank dispatch
   shim (ADR 0002) into those kernels — the duplicated array-primitive traversal
   loops in coeus are retired. DONE, with `coeus-leto/tests/contract.rs` and
   `*_leto_diff.rs` differential suites; coeus workspace 255 tests green.
3. Coeus keeps `ComputeBackend` ownership, wgpu/CUDA backends, autodiff, NN
   kernels (conv/pool/attention), higher sparse formats/backends, optimizers.
   As designed.

Framing correction: `coeus-tensor` is **not** a duplicate of leto to delete — it
is the autodiff-integrated `Tensor`/COW wrapper over `coeus-core`'s dynamic-rank
`Layout`, with CPU compute delegated to leto via `coeus-leto`. coeus-core's
dynamic-rank layout + `Storage` traits, converted at the `coeus-leto` boundary,
are the intended ADR 0002 seam. No leto-side capability gap remains for the CPU
re-base. Remaining cross-repo work is the apollo internal FFT-kernel migration
(apollo-owned) and the themis-0.9 re-pin cascade (§D), which gates clean consumer
rev-bumps to leto 0.24.0.

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

## Performance gap analysis (2026-06-15, AVX2 Win11 x86_64, criterion sample-10)

Decomposition surface, leto/nalgebra median ratio (`decomposition_compare`
benchmark group):

| Kernel | 32×32 gap | 64×64 gap | Bound / cause |
| --- | --- | --- | --- |
| SVD (default `svd_decompose`) | ~10.6× → **~3.5×** | ~18× → **~4.1×** | was one-sided Jacobi; **now** bidiagonal QR (Golub–Reinsch) — see below |
| eigenvalues | ~16× → **~5.8×** | ~16× → **~7.4×** | **was** complex single-shift QR; **now** real Schur (Francis), no-Q path — see below |
| matexp | ~5.7× → **~1.15×** | — | **RESOLVED** (commit `01a197d`). Root cause was *not* matmul (matpow, also matmul-bound, was already ~1.2×) but the **Padé-denominator inverse**: `LuDecomposition::solve_in_place` read the packed `L`/`U` via the bounds-checked logical `Array2::get([r,c])` per element, and `inv()` calls it `n` times → `O(n³)` checked gets dominating. Fixed by the contiguous slice + SIMD `dot_slice` substitution. 64² 2.14 → 0.39 ms. Speeds every LU solve/inv/det. |
| QR | ~3.6× | **~1.0×** | reach parity (row-oriented SIMD apply + blocked ≥256²) |
| LU | ~3.5× | **~0.96×** | reach parity (bulk-copy input + SIMD axpy elimination); solve/inv O(n³) `get` removed |
| Cholesky | ~3.4× | — | solve/inv O(n³) `get` removed (commit `6920655`); decompose scalar (constant-factor) |
| singular_values (values-only SVD) | — | 64² ~2.25×, **256² 1.28×, 512² 1.03×** | **REFRAMED (ADR 0011): not a structural gap.** Levers exhausted at 64²: blocked `dlabrd` (implemented, oracle-verified, reverted — regressive); allocation-free reduce (shipped, ~5%); `apply_left` small-span scalar threshold (tried, **reverted** — a clean light-session toggle A/B showed always-SIMD 121 µs vs threshold 140 µs; the apparent "28%" was CPU-contention dispatch-overhead noise, not a real win — measurement-discipline lesson). The 64² residual is irreducible small-`n` constant overhead vs nalgebra's tuned inner loops; leto is at parity at scale.

  **Root cause, grounded in nalgebra 0.35 source (`linalg/{bidiagonal,householder}.rs`), correcting the earlier guess:** nalgebra's bidiagonalization is **unblocked** (`for ite in 0..dim-1` → `clear_column_unchecked`/`clear_row_unchecked`), *not* blocked — so the earlier "GEMM-parity → blocking doesn't pay" framing was wrong (it assumed nalgebra blocks; it does not, which is *also* why leto's blocked `dlabrd` couldn't beat it). The actual edge is constant-factor tightness: nalgebra computes each reflector **in place** on the matrix column/row (`reflection_axis_mut`, no gather, no per-reflector `v` allocation) with **reused `axis_packed`/`work` scratch**, and its **column-major** layout makes the left (column) reflector contiguous. leto is **row-major**, so the column reflector needs a strided gather, and the reduce still allocates the reflector `v`. The gap is therefore a layout-and-tightness constant factor in the unblocked reduce (plus the sequential Givens sweep), fundamental to leto's row-major n-D design and immaterial at scale (512² 1.03×) where the `O(n³)` SIMD work dominates the constants. Closing it locally would mean an in-place strided-aware reflector + fused gemv/ger apply matching nalgebra's kernels — a significant rewrite of the shared `householder` primitive for microsecond small-`n` gains, against an actively peer-edited file. The blocked `dgebrd`/`dlabrd` was implemented + oracle-verified (192² matches nalgebra) then **reverted as measured-regressive** (256² 4.69→5.68 ms, 512² 32.7→38 ms — the `X`/`Y` look-ahead ~doubles flops, unrecovered by leto's GEMM ratio). Key finding: leto's **unblocked** bidiagonalization is **already at parity at scale** (512² 1.03×); the ~2.25× is a small-`n` (64²) constant-factor artifact. **Profile-confirmed (warm 64² reduce ≈ 50 µs): `apply_left` 21 µs + `apply_right` 17 µs (the SIMD applies) dominate; reflector gather + `v` alloc is only ~12 µs.** The applies run *exactly* nalgebra's flop count (gemv + rank-1 update per reflector), so the residual is **SIMD apply throughput at equal flops** (nalgebra's column-major tuned kernels vs leto's row-major `axpy`-per-row), an irreducible per-flop-throughput floor of the row-major design — *not* allocation (reuse ~5 %), *not* dispatch (scalar-threshold A/B: SIMD beats scalar), *not* blocking (`dlabrd` verified-but-regressive), *not* the sweep (same Golub–Kahan Givens as nalgebra). Every cheaper lever is eliminated by measurement (incl. `axpy_rows` for the rank-1 update: clean multi-run A/B null within noise — the apply is compute-bound, not dispatch-bound); immaterial at scale. UPDATE: per a user decision, a **column-major working-buffer** values reduce (`bidiagonal/colmajor.rs`; left reflector contiguous, no global layout change) shipped — clean A/B ~4–5% faster, narrowing 64² to **~1.64×**, returning `(d,e)` directly (row-major values path removed; factor path unchanged). The remaining ~1.64× is the diffuse small-`n` constant; the row-major→global column-major inversion that could close it fully stays prohibited. FINAL: 12 levers evaluated total — 5 shipped (allocation-free reductions, active-block deflation, values-path zeroing-skip, column-major buffer, reciprocal Givens); 7 measured null/regressive and reverted (blocking/`dlabrd`, dispatch threshold, `axpy_rows`, `validate_input` rewrite, 2×2 closed-form SVD, direct-index scalar apply, `larfg` SIMD norm) **plus a fat-LTO build (null)**. leto's `singular_values_64²` is ~112–120 µs vs nalgebra's session-variable ~53–94 µs — an invariant ~2× per-flop codegen-efficiency constant that holds across every lever incl. LTO, that I could not attribute without a sampling profiler (unavailable in this Windows env: flamegraph needs admin/ETW; no perf/dtrace/samply), and that no discipline-compliant change closes. Immaterial at scale (512² 1.03×). |
| singular_values (superseded note) | — | — | **profiled** (warm 64²): REDUCE ~82 µs + SWEEP ~48 µs vs nalgebra ~78 µs total. The SIMD `apply_right` dot trimmed the reduce ~6%; full parity needs **major algorithm work**, not constant-factor: (a) the bidiagonalization REDUCE is ~2.25× nalgebra's, and a flop analysis proves
this is **not** allocation or constant-factor: leto's 82 µs is already *below* a
naive per-reflector SIMD estimate (~131 µs), so the axpy applies are well
vectorized — removing the per-step `col`/`row`/`w` heap allocations would not
move it. nalgebra's 40 µs is **higher sustained flop/ns from blocked GEMM**
(`dlabrd`) vs leto's per-reflector axpy passes (bandwidth/latency-bound); the only
lever is blocking, the intricate look-ahead-coupled kernel (ADR 0010 Phase 2 note,
needs the `X`/`Y` accumulators). (b) the values SWEEP is **sequential Givens** with
a per-rotation `sqrt`, parity-closeable only by **divide-and-conquer SVD**
(`dbdsdc`). Both are major sequential/blocked-kernel rewrites of the same class as
the deferred eig multishift — correctly phased, not rushed (a subtle bug ships
wrong singular values, a HARD-tier defect). The verified compact-WY block reflector
(`reflector_block`) is the ready substrate for `dlabrd`; the remaining intricacy is
the two-sided look-ahead, not the block apply. |
| dense matmul | ~2.0× | ~1.5× | AXPY (rank-1) scheme: O(n³) output traffic; needs register-blocked GEMM micro-kernel (tile-accumulating SIMD primitive → upstream hermes) |
| matpow | ~1.5× | — | matmul-bound (tracks matmul) |

Priority finding: the largest gaps are **SVD** and **eigenvalues**, *not* matmul.

- **eigenvalues — RESOLVED (partial, [patch] ×2)**: (1) consolidated onto the
  real Schur (Francis double-shift) iteration, deleting the complex single-shift
  QR (`eigenvalues/{complex,qr}.rs`, `Cplx`) — one QR iteration in the crate
  (SSOT), real arithmetic. (2) **no-Q Francis path**: `francis::run` is
  const-generic over `ACCUMULATE_Q`; eigenvalues-only passes `false`, so the
  Schur-vector update is DCE'd (zero cost) and standardization is skipped; block
  extraction factored into one shared helper. (3) **active-block confinement**:
  the eigenvalues-only apply is restricted to the active window `[lo, hi]` (left
  columns `≤ hi`, right rows `≥ lo`); skipped entries are strictly
  upper-triangular and never feed back, so the spectrum is bitwise identical
  (proof: `hi` decreases, `lo` is monotone non-decreasing for fixed `hi` via
  exact-zero deflation). 64×64 `eig` 1.69 → 1.50 ms. The safe confinement is the
  perturbation-free subset of the LAPACK `dlahqr` WANTT=false window.

  **eig DISPARITY RESOLVED (within-block window, now SHIPPED — commit `676ff72`).**
  The `dlahqr` WANTT=false narrowing — left columns `[k, hi]`, right rows
  `[lo, k+len]`, explicit bulge zeroing — is ~half the apply work of the `[lo,hi]²`
  confinement and cuts the dominant scalar right-apply to the bulge neighbourhood.
  Clean A/B 64×64: confinement 2.69 ms → restriction 0.69 ms (3.9×); vs nalgebra
  0.60 ms ⇒ **1.16× (near parity, from ~4.6×)**. It was rejected three times
  *only* because it diverges from nalgebra by `√(ε‖A‖) = 1.75e-7` on the defective
  eigenvalue (below), exceeding the *old* fixed `1e-7` test bound; it is
  backward-stable and became admissible once that bound was corrected (Phase 0) to
  the analytically-derived `8·√(ε‖A‖)`. The earlier "#20 rejected" verdict was
  conditional on the brittle test, which is now fixed — the window is the genuine
  win, not a defect.

  Historical note (#20, machine-checked): the window's divergence was *not* a bug
  and *not* bad scaling — the 16×16 fixture has a **defective (high-multiplicity)
  zero eigenvalue**: `det(A) = -8.7e-30 ≈ 0`, nullity 3 (smallest singular values
    `[5.55, 1.3e-15, 0, 0]`). A defective eigenvalue perturbs as `O(√(ε‖A‖))`, and
    `√(ε‖A‖) = 1.54e-7` matches the spurious `1.75e-7` imaginary part nalgebra
    reports for the defective-0. Both algorithm variants are backward-stable; they
    legitimately differ by `√(ε‖A‖) > 1e-7`. The narrowing is correctness-equivalent
    (leto's `~1e-15` is in fact nearer the true real-0) but cannot match the oracle
    at the over-tight 1e-7 absolute tolerance, for a sub-2× apply gain over the
    confinement already landed — not pursued.
  - **Matrix balancing (`dgebal` Parlett–Reinsch scaling)** — implemented, verified
    spectrum-preserving (radix-2 exact), then **reverted**: on the well-scaled
    integer benchmark matrices it produces `D ≈ I`, so it neither conditions the
    defective eigenvalue (the narrowing still failed *with* balancing) nor speeds
    convergence, while adding O(n²·sweeps) overhead that **regressed** `eig`
    (64×64 1.50 → 1.64 ms). Net-negative for the target metric with no consumer
    need; removed per subtractive/anti-over-engineering discipline.

  (4) **SIMD apply** (now the eig test is backward-error-correct, see below):
  the shared Householder apply and the Francis **left**-apply route their
  contiguous inner sweeps through `Scalar::axpy_slice` (SSOT SIMD), row-oriented
  with a reused scratch buffer and a `SPAN_SIMD_MIN = 32` threshold (narrow spans
  stay scalar). Bitwise-identical to the column-oriented form. 64×64 `eig` 1.50 →
  **1.11 ms** (5.9× → **4.4×**); 32×32 284 → **242 µs**. The brittle `1e-7`
  oracle was first replaced by the analytically-correct backward-error bound
  `8·√(ε‖A‖)` (defective-eigenvalue derivation above, machine-checked), which
  unblocked all backward-stable reorderings.

  Residual gap (~4.4×) is the Francis **right**-apply (3-wide per row, no
  contiguous span to vectorize) plus the per-step reflector overhead. Closing it
  needs the blocked/aggregated-reflector (compact-WY / small-bulge multishift)
  rewrite so the apply becomes a GEMM — tracked as `[major/arch]` (#21), not a
  patch. **#20 CLOSED**: the within-block `[k, k+len]` narrowing and balancing
  were both investigated and rejected (above).
- **SVD — RESOLVED ([patch]+[minor])**: both singular values and the full thin
  SVD now use the implicit-shift **bidiagonal QR** (`svd/bidiagonal_qr.rs`,
  reusing `bidiagonalize`). No `AᵀA`, so accuracy is `κ(A)` not `κ(A)²` (resolves
  `diag(1,1e-6)` to 1e-15). `svd_via_bidiagonal` accumulates `U`/`V` via Givens
  (const-generic `VEC`; DCE'd for values-only); `svd_decompose` routes to it
  (Gram path deleted, SSOT). Current validation: full SVD 32×32 ~292→103.6 µs
  (10.6×→3.5×), 64×64 ~3.1 ms→758.8 µs (18×→4.1×); values-only
  `singular_values` 57.1 µs (32×32) and 275.0 µs (64×64), a 25.5%/39.5% local
  median reduction after skipping factor accumulation. One-sided Jacobi
  (`svd_rank_revealing`) retained for rank-deficient / maximal accuracy.
  **SIMD bidiagonalization** (shared Householder apply now routes through
  `Scalar::axpy_slice`, see the eigenvalues entry): 64×64 full SVD 758.8 →
  **417 µs** (4.1× → **3.4×**), `singular_values` 275.0 → **133 µs** (3.8× →
  **2.3×**). Residual gap is the scalar **Givens** bidiagonal-QR sweep (strided
  2×2 rotations, sequential bulge chase).

  **ADR 0010 Phase 2 (blocked U/V factor formation) — implemented, verified, then
  REVERTED as valueless.** The factor formation was rewritten to store the panel
  reflectors and form `U = L₀…L_{n-1}` / `V = R₀…R_{n-2}` by blocked compact-WY
  `apply_block_right` (one-sided, no `dlabrd` look-ahead); verified correct by a
  256² `A = U B Vᵀ` reconstruction + orthogonality + singular-value contract. But
  the A/B measurement is decisive: **full SVD 256² is 164 ms blocked vs 163 ms
  unblocked — no difference.** The entire 256² SVD cost (~163 ms vs nalgebra
  46.6 ms, 3.5×) is the **sequential Givens bidiagonal-QR sweep**; the U/V
  formation is < 1 ms. Blocking it is therefore cargo-cult (complexity with no
  present performance need) and was removed. **The genuine SVD lever is Phase 3
  — accelerating the Givens sweep itself** (`dbdsqr`), which is inherently
  sequential (each rotation feeds the next in the bulge chase) and is exactly why
  it is the residual. `apply_block_right` reverted with it (no value-adding
  consumer; `apply_block_left`/blocked-QR Phase 1 retained, a measured 256² win).

  **ADR 0010 Phase 3 (SVD) — DONE, disparity resolved.** Without restructuring the
  serial bulge chase, the per-Givens `U`/`V` **column** rotation (striding the
  row-major factors — a cache line per row, no SIMD) was made contiguous by
  accumulating `U`/`V` **transposed**, so each rotation mixes two contiguous rows:
  bitwise-identical factor, cache-friendly auto-vectorized loop. **256² full SVD
  164 → 34.7 ms (4.7×, now *faster* than nalgebra 46.6 ms); 64² 1.31 → 0.60 ms**
  (clean A/B), verified by the SVD reconstruction + nalgebra batteries (commit
  `9bef76e`). The trick does *not* transfer to eig (`H` is the read/written working
  matrix, not an isolated accumulator), so the eig residual still needs the
  multishift block rewrite.
- **matmul — OPEN**: register-blocked GEMM micro-kernel needs a tile-accumulating
  SIMD primitive owned upstream in hermes (multi-repo, peer-agent lane); a prior
  scalar 4×4 tile regressed (no SIMD in the tile). Coordinated effort.

- `stack` (rank `N -> N+1`): CLOSED ([minor]) — implemented via the `InsertAxis`
  rank helper (dual of `RemoveAxis`, ranks 0..=7). `concat`/`pad`/`split`/`stack`
  all closed.
- Dynamic-rank boundary: DECIDED ([major]) in
  `docs/adr/0002-coeus-rank-boundary.md` — const-generic dispatch shim at the
  Coeus boundary, shim owned by Coeus, Leto stays const-rank. Phase 6 leto-side
  capabilities authored const-rank.
- `symmetric_eigen_jacobi`: CLOSED ([minor]) — now generic over `T: RealScalar`,
  native precision, no hidden widening. Residual: no wider-accumulator variant;
  consumers needing higher working precision than storage convert first
  (explicit). f16/bf16 transcendentals use the documented f32 fallback.
- `symmetric_eigenvalues_jacobi`: CLOSED ([minor]) — sorted eigenvalues without
  eigenvector allocation, implemented by a zero-sized no-vector rotation target
  over the shared Jacobi diagonalization kernel. Evidence tier:
  value-semantic full-vs-values parity and strided-view tests.
- Contiguous-slice view access: CLOSED — `as_slice`/`as_mut_slice` are now
  offset-independent C-dense; `as_slice_memory_order`/`as_mut_slice_memory_order`
  expose F-order/offset blocks. Apollo's end-to-end native Leto migration is
  complete at commit `324f380`; its resolved Rust graph contains no `ndarray`.
- std::ops operator overloading: DEFERRED ([arch]) in
  `docs/adr/0001-elementwise-operator-overloading.md` (orphan rule). `scalar_map`
  covers array–scalar arithmetic; no consumer blocked.
- Indexed zip parity: CLOSED ([minor]) — `indexed_zip_mut_with` and
  `indexed_zip2_mut_with` provide `Zip::indexed`-style logical coordinates for
  one- and two-input mutable zip traversals.
- Stage C3 column-walk elementwise traversal: CLOSED for binary and unary
  `map_into` plus `zip_mut_with` ([patch]/[minor]) — all use shared cache-line
  `TileGeometry`.
  Evidence tier: value-semantic strided tests plus criterion differential
  timing before/after the optimization. Remaining cache-aware CPU kernel work
  is blocked matmul cache hierarchy selection and themis topology wiring.
- Stage C2 dense norm SIMD coverage: CLOSED for `norm_l1`/`norm_l2`/`norm_max`
  over dense memory-order slices. `norm_l1` and `norm_max` now route f32/f64
  through Hermes absolute-value reductions; reduced precision keeps the native
  scalar fallback. Evidence tier: value-semantic norm tests plus criterion
  in-run scalar-reference comparison. Remaining reduction work is truly
  non-dense strided reductions, which need per-lane partial accumulators.
- Coverage of new ops: value-semantic tests plus ndarray differential oracles
  now cover the unary math suite (`exp`/`sqrt`), `scalar_map`, `concat`,
  `stack`, `batched_matmul` (per-batch ndarray `dot`), and `cumsum` (reference
  accumulate), alongside the existing map/reduction/matmul differentials. RNG is
  validated against closed-form mean/variance (correct per policy, not ndarray).
  Remaining: differential coverage is leto-internal; consumer-side (Apollo/Coeus)
  migration tests are the next cross-repo step.
- `leto-python` rustdoc ICE via `numpy 0.23`: RESOLVED without changing the
  FFI dependency constraint. `leto-python` is a PyO3 extension boundary with no
  public Rust API, so its library target sets `doc = false`; full workspace
  docs no longer walk NumPy 0.23's broken intra-doc link path.
- Differential coverage: ndarray oracle covers map/reductions/matmul, unary
  suite, concat/stack, batched matmul, and cumsum. RNG uses closed-form
  references. Indexed zip currently rests on value-semantic traversal tests.
- Oracle performance parity: reverse-last-axis `sum`/`norm_l2` is at parity or
  faster than ndarray on the recorded 256x256 benchmark. Dense matmul is not
  at parity with ndarray/nalgebra and blocks replacement-performance claims.
  The 0.19.7 Hermes fused multi-row AXPY path improved Leto medians from
  21.443 µs / 127.63 µs / 2.4357 ms to 17.430 µs / 108.98 µs / 1.0631 ms for
  64x64/128x128/256x256, but ndarray/nalgebra remain faster. The 0.19.2
  zero-skip branch-removal experiment was rejected after canonical dense
  256x256 instability/regression. The 0.19.3 packed-RHS dot and
  scalar-row-update experiments also regressed 128x128. The 0.19.4 Hermes
  `tiled_gemm` f64 dense path regressed 128x128, and small-matrix serial
  scheduling was slower than row-block parallelism. The 0.19.5
  `MATMUL_ROW_BLOCK=16` and first-shared-row output initialization experiments
  did not meet the release benchmark stability/performance gate. Post-0.19.7
  Hermes column-chunk `axpy_rows` regressed 64x64/128x128/256x256 and ended
  with `STATUS_ACCESS_VIOLATION`; `MATMUL_ROW_BLOCK=64` also regressed against
  the 32-row baseline. Row-block fused-branch/alpha-buffer hoisting produced
  no statistically significant 128x128 improvement and also ended with
  `STATUS_ACCESS_VIOLATION`. Generic 4x4 registered dense tiles regressed the
  oracle shapes and are not retained. Hermes `axpy_rows_batch` improves the
  local themis-0.9 128x128 Leto oracle median from 212.64 µs to 98.853 µs when
  gated to that row regime, but broad depth-batched routing regressed other
  oracle shapes and is not retained. Next work needs allocation-controlled
  reusable packing scratch or a verified external micro-kernel provider with
  profile evidence.
- themis-0.9 migration + dependency resolution (re-diagnosed/MEASURED
  2026-06-15). Three coupled facts:
  1. **`Cargo.lock` is gitignored** in leto (`.gitignore`), so there is no
     committed lock — contrary to the prior "commit Cargo.lock" claim. leto's
     standalone build depends on a locally-generated lock.
  2. **Fresh pure-git resolution is broken.** `hermes-simd` (`efac0454`, the
     measured-good fused-AXPY matmul pin) and `mnemosyne-arena`/`moirai-iter`
     transitively require `themis ^0.8.0` with no rev, so `cargo
     generate-lockfile` floats themis to the default-branch HEAD (0.9.11) and
     fails `^0.8.0`. cargo will not unify a rev-pin with the transitive
     version-spec, and `[patch]` to the same git source is rejected — so no
     leto-local pin change resolves it. Builds worked only via a frozen
     pre-drift themis-0.8 lock (now superseded locally).
  3. **The themis-0.9 path regresses matmul.** Built leto 0.24.0 against the
     local themis-0.9 stack (path-patches): all 122 tests + 4 doctests pass, but
     the required hermes bump `efac0454`→`e6761ac` (dispatch/AXPY refactor)
     regresses dense matmul **64² 17.4→24.9 µs (+43%)**, **128² 109→176 µs
     (+61%)**, 256² ~unchanged. So themis-0.9 adoption is blocked not just by pin
     coordination but by a **measured hermes matmul regression**.
  Resolution = stack-wide re-pin cascade (themis → mnemosyne → moirai → hermes →
  leto → apollo/coeus, in order, since each pins old revs of the others) PLUS a
  hermes-side fix restoring the AXPY/dispatch perf on its themis-0.9 line; then
  leto migrates and re-measures matmul. Owned at the meta/stack level, not
  leto-local. Process gap to fix alongside: either commit a frozen `Cargo.lock`
  (un-gitignore) or pin the whole stack so fresh resolution is reproducible.
  Interim: leto builds locally via the apollo/coeus-style path-patch set
  (uncommitted, flagged in `Cargo.toml`); consumer rev-bumps to leto 0.24.0 wait
  on the cascade (coeus already verified compatible — 22/22 contract tests green
  against working-tree leto 0.24.0).
- Evidence tier of this audit: codebase scan + existing test suites +
  criterion benchmark measurements. No machine-checked proof was performed.

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
