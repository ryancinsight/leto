<a id="leto-stable-qualification"></a>
## LETO-STABLE-QUALIFICATION — leto stable-release qualification [patch] — todo
- priority: verification; milestone: M-array-core; basis: `e929b7f8cb3e`.
- Outcome: Settle ownership of the duplicate sparse matrix representations. Follow the [release scope](docs/adr/0037-release-scope-1-0.md) and [milestone exits](backlog/milestones.md). See the [proposed decision](docs/adr/0041-canonical-sparse-storage.md). Planning targets for `leto` are 0.52.1, then 0.53.0 proposed after sparse-storage review, then 1.0.0. Verification alone can advance the current version.
- Acceptance and sequence: [stable-release checklist](checklist.md#stable-release-checklist).

<a id="LETO-REGISTRY-SCALAR-PROVIDER"></a>
## LETO-REGISTRY-SCALAR-PROVIDER [major]
- status: todo; priority: correctness; milestone: M-array-core; needs: published Hermes 0.10.0.
- outcome: Leto and Leto Ops 0.51.0 resolve one published scalar identity with Aequitas.
- acceptance: [ADR 0040](docs/adr/0040-registry-scalar-provider.md) and the 0.51.0 checklist.

<a id="LETO-DIGAMMA"></a>
## LETO-DIGAMMA [minor]
- status: blocked; priority: verification; milestone: M-array-core; needs: none; integrator: pi:01a1885f-3f38-73dd-a2f0-7e3fde63b66a
- Blocker: `leto-ops` 0.47.0 publication waits on `moirai-runtime` 0.11 on
  crates.io, and the Coeus optional backend capability belongs to its owner.
- Re-open trigger: the pending publisher reports the cohort published, or the
  Coeus owner lands the backend capability.
- Outcome: `DigammaOp` is a zero-sized `UnaryOp<T: RealScalar>` marker that
  calls Eunomia's single `FloatElement::digamma`; Leto keeps its layout-aware
  traversal, so contiguous, strided, offset, transposed and empty views stay
  copy-free and no scalar formula is duplicated in Leto.
- Delivered: provider delegation, the `leto`/`leto-ops` 0.47.0 cohort, the
  design note and the acceptance regressions. The design note is the contract:
  [digamma](docs/design/digamma.md).
- Evidence: analytic half/integer anchors, transposed input and output, empty
  shapes, signed zeros, poles, special values and native doubles outside the
  f32 range, each executed against the provider's own scalar contract.
- Provider: Eunomia 0.11.0 is published and 0.12.0 follows it. Leto keeps
  `>=0.11.0, <0.12.0` because Aequitas 0.3.1 caps the same cohort; a wider
  Leto range resolves two Eunomia versions in one packaged graph and fails the
  packaging verification, so the cohort moves together.
- Hardware remains partial until Hephaestus implements and validates its own
  device kernels; CPU results cannot establish device parity.

# Leto Work Backlog — active patch **0.46.3**, [SpMV finite-range recovery](backlog/leto-spmv-finite-range-20261020.md)

<a id="LETO-SPARSE-FORMAT-CONSOLIDATION-2026-11-13"></a>

## LETO-SPARSE-FORMAT-CONSOLIDATION-2026-11-13 — Two sparse format families, `leto::*Array` and `leto_ops::*Matrix` [arch] — todo
- priority: tightening; milestone: M-array-core
- needs: none
- basis: d2206475
- scope: `crates/leto/src/infrastructure/sparse/`, `crates/leto-ops/src/application/sparse/`, the `leto` crate-root re-exports, and every consumer of the retired family
- Evidence: at d2206475 `git grep -l -E '\b(CsrArray|CscArray|CooArray)\b'` over `*.rs` finds 7 files, all in `leto`, and one consumer file (prometheus `src/stoichiometry.rs`); `\b(CsrMatrix|CscMatrix|CooMatrix)\b` finds 42, all in `leto-ops`. `leto-ops` depends on `leto`. ADR 0028 (Accepted) governs the `*Array` family and ADR 0034 puts narrow CPU CSR kernels in Leto, so the survivor is a decision, not a default.
- Outcome: one sparse format family. An ADR (new, or a revision of ADR 0028) records the survivor and why; the retired structs, their tests ported, every in-repo caller and the `*Array` consumer migrate in the same slice, with no alias or adapter.
- Acceptance: `git grep -n -E 'pub struct (Csr|Csc|Coo)(Array|Matrix)\b' -- crates` lists one family; each retired test case runs against the survivor; prometheus builds against it.
- Next step: review [proposed ADR 0041](docs/adr/0041-canonical-sparse-storage.md) against both families’ APIs, test suites and the Prometheus consumer before starting migration.

<a id="LETO-CSR-INDEX-WIDTH-2026-09-27"></a>

## LETO-CSR-INDEX-WIDTH-2026-09-27 — Narrow sparse index storage from usize to u32 [major] [arch] — todo
- priority: tightening; milestone: M-array-core
- needs: LETO-SPARSE-FORMAT-CONSOLIDATION-2026-11-13
- basis: d2206475
- scope: `crates/leto-ops/src/application/sparse/` and `crates/leto/src/infrastructure/sparse/` (public format change); every consumer of the `&[usize]` accessors
- Delivered: [ADR 0038](docs/adr/0038-csr-index-width.md) (Proposed): decision, the derived `f64` ceiling `1.308×` on `bench_spmv`, two unverified hypotheses about what binds SpMV, and the migration consequences.
- Outcome: migrate the surviving format's CSR, CSC and COO to `I: SparseIndex = u32` with checked entry, then every in-repo caller, in one slice per format.
- Acceptance: count-level `fits::<I>` boundary tests; `u32` and `u64` results bitwise equal on the sparse suites; `bench_spmv` at 1<<20 gives a ratio interval with lower bound above 1 and not wholly above `1.308`, and `bench_csc_spmv` at 1<<20 a lower bound above 1 (ADR 0038, Verification plan), else the ADR's overturning clause applies.
- Next step: accept or redirect ADR 0038, then migrate `leto-ops` CSR with its callers.

<a id="LETO-BLOCKED-LU-L3-GATE-2026-09-27"></a>

## LETO-BLOCKED-LU-L3-GATE-2026-09-27 — Cache-aware gate for blocked (BLAS-3) LU [patch] — todo
- priority: tightening; milestone: M-array-core
- needs: none
- scope: `crates/leto-ops/src/application/linalg/lu/decompose.rs`, `crates/leto-ops/benches/kernels.rs`
- Evidence: the 2026-07-20 blocked-LU experiment (`docs/benchmarks.md#blocked-lu-cache-resident-regression-2026-07-20`) measured a right-looking blocked LU as a regression at n=256/512 on a 36 MiB-L3 host, because unblocked SIMD axpy already runs at cache bandwidth below `n ≈ 1200`; it was reverted, not deleted from history.
- Outcome: a blocked LU gated on `working_set > l3_bytes` (reusing the parallel policy's cache-aware threshold) that never regresses cache-resident sizes and wins past the LLC, with trailing-update copies eliminated via matmul into strided views.
- Acceptance: `lu_scaling` shows no regression at any cache-resident `n`, and a measured win past `l3_bytes` on a quiet host.
- Next step: instrument `working_set > l3_bytes` as the routing predicate before reintroducing the blocked path.

<a id="LETO-SPARSE-DIRECT-1"></a>

## LETO-SPARSE-DIRECT-1 — Sparse LU with partial pivoting past the dense route [feature] — todo
- priority: feature; milestone: M-ops-linalg
- needs: none
- basis: b738a4e2
- scope: `crates/leto-ops/src/application/sparse/{lu_numeric/,lu_symbolic.rs,lu_sparse/}`
- Evidence: delivered at b738a4e2 are `SparseLuSolver` (CSR, generic scalar, natural or AMD ordering, dense dispatch) and `examples/poisson_sparse_lu_accuracy.rs`, which exits 0 with a backward error of 9.2e-17 against its derived 3.4e-13 bound; 37 `lu_sparse`, `lu_numeric`, `lu_symbolic` and `amd` lib tests pass, among them `factor_random_sparse_n64_diff_dense` and `solver_is_generic_over_f32`. The sparse factorization is pivot-free: a matrix needing partial pivoting takes the dense route (`SMALL_SWITCH_DEFAULT` doc, test `owned_factor_falls_back_to_dense_when_pivoting_required`), and `max_size` (default 2048) rejects larger orders.
- Consumer: CFDrs keeps a third-party solver for its direct-after-GMRES tier until Leto offers this (`repos/CFDrs/gap_audit.md`, entry 2026-07-17); the failure-mode-independence requirement stands.
- Outcome: a generic sparse LU whose pivoting path stays sparse, so a pivoting-required system above the dense limit solves without an iterative fallback or dense materialization.
- Acceptance: a pivoting-required CSR system of order above 2048 solves with a residual below its derived backward-error bound; parity with the dense route at orders both reach.
- Next step: locate the authoritative reference algorithm for sparse LU with partial pivoting and its domain of validity before implementation.

<a id="LETO-LOW-PRECISION-FACTOR-ORACLE-2026-09-25"></a>

## LETO-LOW-PRECISION-FACTOR-ORACLE-2026-09-25 — Bf16 and large-order factor bounds are vacuous across the scans [patch] — todo
- priority: verification; milestone: M-array-core
- needs: none
- basis: fba6315393561b514b59299079ff1da88637c0a2
- scope: `crates/leto-ops/tests/ops/backward_error/`, `graded_scan/coverage.rs`, `symmetric_qr/fixtures.rs`, `col_piv_qr.rs`, `scale_range/pivoted_qr.rs`; the id is cited at `schur/skew_ties.rs`
- Evidence: `gamma_tilde` is Higham and Mary, SIAM J. Sci. Comput. 41(5), 2019, Theorem 2.4, eqs. 2.1, 2.3 (MIMS EPrint 2018.33; journal pages from citing literature), `PROBABILISTIC_LAMBDA` 7. Delivered: `graded_scan/coverage.rs` tallies the SVD and Schur residual and orthogonality sites per format over the scan (F16 162 cases, Bf16 1506) and large-order bands, and requires every unasserted case to derive its bound at least 1 from its own rounding counts (`E(N_r) + 2 ln(1 + √m·expm1(E(N_v))) ≥ ln 2`, `E = ln(1 + γ̃)`). Tallies (asserted, vacuous), scan: F16 SVD (162, 0), Schur residual (148, 14), Schur orthogonality (162, 0); Bf16 SVD (0, 1506), Schur residual (0, 1506), Schur orthogonality (272, 1234); large orders: F16 (0, 12) and Bf16 (0, 6) at every site. They equal the b738a4e2 probe, which placed the 14 vacuous F16 Schur-residual cases at n = 8 and the 272 asserted Bf16 Schur-orthogonality cases at n = 2 and 3 (orders are not tabulated by the test). Not yet counted: symmetric QR (F16 6 of 7 asserted, Bf16 0 of 7) and pivoted QR (F16 2 of 2, Bf16 0 of 2), the PR 415 figures.
- Bound: at `u = 2^-8` and `λ = 7`, `γ̃_k < 1` holds exactly for `k ≤ 624` (pinned by `coverage.rs`; the first-order `λ√k·u < ln 2` gives 642 and is necessary only); one length-3 reflector costs 24 (`γ̃_24 = 0.144`). At the current per-entry counts the Bf16 SVD and Schur-residual cases all derive vacuity.
- Outcome: for each F16 and Bf16 factor site (SVD, Schur, symmetric QR, pivoted QR) in the scan band and the large-order band, the residual and orthogonality are asserted by an oracle informative there, or the site is recorded vacuous by its derived rounding count and covered by the a-posteriori certificates; the original alternative is the deterministic bound at an instrumented transformation count.
- Acceptance: the counting test of `coverage.rs` covers symmetric QR and pivoted QR as it covers SVD and Schur: per (format, site) the asserted and vacuous tallies, each vacuous case deriving its bound at least 1, the 6 of 7 F16 symmetric-QR tally included; `PROBABILISTIC_LAMBDA` stays 7 (a smaller `λ` raises the per-product failure probability `2exp(-λ²(1-u)²/2)`, a larger one widens the bound); no tolerance widening.
- Next step: tally symmetric QR (`symmetric_qr/fixtures.rs` `backward_bound` and `assert_backward_stable`) and pivoted QR (`col_piv_qr.rs`, `scale_range/pivoted_qr.rs`) through `Coverage::record`; then instrument the per-entry rounding count of the Bf16 Francis and Golub-Kahan paths and tabulate the orders at which `γ̃` falls below 1.

<a id="LETO-OUTPUT-ZEROFILL-COST"></a>

## LETO-OUTPUT-ZEROFILL-COST — Confirm the zero-fill cost of the allocating kernels on isolated cores [patch] — todo
- priority: tightening; milestone: M-array-core
- needs: none
- basis: bd349c5e
- scope: `crates/leto-ops/benches/output_allocation.rs`, `embedding_gather`, `reduce_axis`, `topk_axis`, `triangular`, `ray_line_integrals`
- Evidence: [ADR 0039](docs/adr/0039-zero-initialized-output-storage.md) replaced `VecStorage::uninit` with zero-filled output storage. Its A/B wall-clock run under 29-100% peer load is discarded as invalid, since the spread inside one binary exceeds any effect, so the cost is neither established nor excluded.
- Outcome: per operation, the B/A median ratio with a confidence interval from isolated cores or instruction and cache counters, recorded in ADR 0039. A ratio above the noise bound on a hot path opens the typed `MaybeUninit` route (`refs/pull/360/head`) as sliced items.
- Acceptance: ADR 0039 carries the measured ratio and interval for each of the five operations at a recorded host load.
- Next step: run `cargo bench -p leto-ops --bench output_allocation` on a quiet host against `bd349c5e`.

<a id="LETO-MIRI-GATE-2026-09-10"></a>

## LETO-MIRI-GATE-2026-09-10 — Widen the Miri gate past the storage substrate and the reduction output [patch] [safety] — todo
- priority: verification; milestone: M-array-core
- needs: none
- basis: b738a4e2
- scope: `.github/workflows/ci.yml` (`miri` job), `.config/nextest.toml` (`[profile.miri]`), `crates/leto-ops/tests/ops/`
- Evidence: the job runs `leto --lib` and `leto-ops ops_tests -E test(/reduction/)` under both Stacked and Tree Borrows (the `Miri — reduction output` steps of `ci.yml`), the second excluding `parallel_axis_reduction_matches_the_serial_result` (the 40000x8 input exceeds the 600 s profile bound, so Miri covers the serial output path only). The serial route is safe code over zero-filled output storage, so Miri does not reach the `unsafe` raw-pointer writes of the parallel route; `VecStorage::uninit` is removed because it broke the `Vec::set_len` initialization precondition, which no Miri run can observe ([ADR 0039](docs/adr/0039-zero-initialized-output-storage.md)). Measured 2026-09-10 (Windows host, Miri 0.1.0, nightly 2026-08-26): `leto --lib` 130 tests 206 s Stacked, 363 s Tree; `leto-ops --lib` (216 tests), `leto-ops --test ops_tests` (352) and `leto layout_property_tests` each exceed 10 min, so they stay out as unaffordable, not as believed sound.
- Open observations, unconfirmed on Linux: `complex_transpose_allocations` aborts under Miri inside the test's own `CountingAllocator` (Windows `HeapFree`, `GetNumaHighestNodeNumber`); two `leapfrog` tests fail an exact `assert_eq!` under Miri with the native/interpreted float divergence unexplained.
- Outcome: each excluded surface, the parallel route of the reduction output included, joins the job with a measured bound or is closed with the reason it stays out; both borrow models stay (an MN-437 fix passed Stacked Borrows while Tree Borrows still rejected it).
- Acceptance: the `miri` job is green on main under both models over the widened scope; the two observations are confirmed or refuted on `ubuntu-latest`.
- Next step: run `leto-ops --lib` and `ops_tests` under Miri on `ubuntu-latest` with a bounded workload and record per-test time.

<a id="leto-windows-source-identity"></a>

## LETO-WINDOWS-SOURCE-IDENTITY — Regress the older-timestamp source swap in the atlas identity checker [patch] — todo
- priority: verification; milestone: M-array-core
- needs: none
- basis: b738a4e2
- scope: owner is the atlas stack scripts (`scripts/tests/test_atlas_build_identity.py`, `scripts/atlas_build_stale_packages.py`), not a leto path; leto's `.githooks/pre-push` is a stack-synced shim and stays unedited
- Evidence: the 2026-09-07 failure (release reused local `leto-2960c0092a779492` metadata without `transpose_copy`, E0432 in `application/layout/complex/batch.rs:8`, after a primary/lane source swap under one shared target) predates the shim. The shim runs `atlas-build-identity.py`, whose records bind path packages to file digests, so another tree's artifact cannot stand in for this one (`_path_packages` docstring), and one real-cargo test alternates source states under one shared target (`test_a_peer_rerunning_a_build_script_is_not_trusted`, in the `shutil.which("cargo")`-gated `MultiPackageRunTestCase`); `test_source_transition_cleans_once_and_rebuilds_the_affected_package` is a scripted-build unit test (a Python script writes the source token into a fake rlib under a patched toolchain identity), not cargo. Read at atlas 2b000c51 on 2026-11-13.
- Missing: no `scripts/tests/test_atlas_build_*.py` case gives the incoming source mtimes older than the recorded artifact (the only `os.utime` on a source file, `test_atlas_build_package_source.py:77`, moves its mtime forward), and no check covers an ad hoc `cargo build` outside the hook, the probable path of the 2026-09-07 failure (hypothesis: a local release-profile build; leto has no committed release test gate).
- Outcome: the checker's alternation of two source states of one path package under one shared target is regressed with the incoming state older, and the ad hoc and release-profile build cases are decided in or out of scope; if out, the release-profile reproduction is filed as `correctness`.
- Acceptance: state A builds, then state B whose files carry mtimes older than A's artifact builds; each build exposes exactly its own API and computes its expected values; no cache fork, flag suppression, retries or source workaround.
- Next step: file the regression with the atlas stack scripts board.

## LETO-CRATES-METADATA-1 — Satisfy registry metadata [patch, blocked] — milestone: M-array-core
**Owner:** Codex; implementation metadata reconciled 2026-08-17.
**Outcome:** give both publishable packages precise crates.io descriptions so
the registry can validate the current 0.42.0 archives.
**Acceptance:** locked metadata and package dry runs pass; `leto` and
`leto-ops` 0.42.0 are verified for publication; trusted-publishing-only
enforcement and exact GitHub Releases are configured by the release authority;
and the dependent publication queue resumes.
**Status:** both package manifests carry bounded-context descriptions, complete
registry metadata, and the current workspace version `0.42.0`; locked
no-dependency metadata confirms those fields. The locked package gate is
blocked under the Atlas development overlay because Cargo requests a lockfile
rewrite. The offline package attempt succeeds only after generating overlay
lock churn (removing git source identities and adding `[patch.unused]` entries),
which is not release evidence and was discarded. Re-open when a standalone
locked package job runs without the umbrella overlay. Hosted verification,
merge, indexing, trusted-publisher registration, and publication remain
external release work; this task does not authorize registry mutation.

## LETO-PYTHON-RELEASE-1 — Python release wheels [patch, in progress] — milestone: M-array-core
**Owner:** Codex `/root`
**Scope:** the `leto-python` release workflow, protected GitHub environment,
distribution documentation, and PyPI trusted publisher. Python binding behavior
is a non-goal.
**Acceptance:** a GitHub Release tagged `leto-python-v<version>` builds locked
Linux, Windows, and universal macOS wheels for CPython 3.9–3.13, installs and
imports each wheel as `leto_python`, validates Cargo-owned distribution identity,
attests and attaches the exact artifacts, then publishes the same wheels to the
`leto-python` PyPI project through OIDC.
**Current evidence:** the release workflow and synchronized distribution
contract are implemented, and GitHub environment `pypi` accepts only
`leto-python-v*` tags. A historical locked CPython 3.13 wheel built as
`leto-python` 0.39.0, installed into an isolated target, and imported as
`leto_python`; the current workspace version is 0.42.0, so that artifact is not
current-release proof. Hosted CI and pending-publisher registration remain
open.

<a id="LETO-INPLACE-INTENSITY-GATE-2026-09-01"></a>

## LETO-INPLACE-INTENSITY-GATE-2026-09-01 — `map_inplace` has no cache-residency gate [minor] — blocked
- priority: tightening; milestone: M-array-core
- needs: none
- basis: b738a4e2
- scope: `crates/leto-ops/src/application/unary/traversal.rs`; `map_inplace`'s eager raw-closure behavior and the `_into` and binary gates stay unchanged; no new threshold constant
- Outcome: a typed `unary_map_inplace` that routes through the existing `map_into_gated` policy, so a `UnaryOp` with `COMPUTE_BOUND = false` stays sequential below the cache gate.
- Evidence: `unary_map_into` honors `UnaryOp::COMPUTE_BOUND`; `map_inplace` (`crates/leto-ops/src/application/unary/traversal.rs`) gates on the flat `PARALLEL_THRESHOLD` element count for every op. Measured against a sequential loop over the same slice and closure (36 MiB-L3 AVX2 host, best of 9): ratio 1.00 at 16384-49152 elements, parallel loses at 65536-131072 (f64 4.85-6.44, f32 7.82-9.80) and wins from about 1M elements (0.29 f64, 0.35 f32 at 1048576); a compute-bound `sin` control reads 0.15 at 65536. The LLC-byte model holds on the binary path (naive-parallel break-even 1.57M f64, 2.1M-3.1M f32); the in-place figures do not transfer to it.
- Blocker: no in-stack bandwidth-bound caller of an in-place `UnaryOp` (`AbsOp` and `NegOp` declare `COMPUTE_BOUND = false`); an API without a caller is speculative generality, so the hazard is documented at the call site.
- Re-open trigger: the first bandwidth-bound in-place caller in the stack, or an external report against published `map_inplace`. Re-checked 2026-11-13: no file outside leto imports `leto_ops::map_inplace`; the CFDrs hits are `Field2D::map_inplace` and the kwavers hits `indexed_map_inplace`, a sequential row walk.
- Acceptance: value-identical to `map_inplace` for the same op; a `COMPUTE_BOUND = false` op stays sequential below the cache gate and matches the sequential control within noise across 64 KiB-2 MiB; a `COMPUTE_BOUND` op parallelizes as today; zero allocations after caller storage.

<a id="LETO-BOARD-LEGACY-FOLD-2026-11-13"></a>
## LETO-BOARD-LEGACY-FOLD-2026-11-13 — Convert the remaining legacy board blocks and `checklist.md` to the DoR schema [patch] — todo
- priority: verification; milestone: M-array-core
- needs: none
- basis: 9d322c397b8096875f8f6a8826a1e218c2d9b33d
- scope: `backlog.md` from `## Atlas in-house replacement roadmap` to its end, the `LETO-CRATES-METADATA-1` and `LETO-PYTHON-RELEASE-1` blocks, and `checklist.md`
- Evidence: at 9d322c39 the tail holds 46 `[x]` and 13 unchecked `[ ]` lines (Stage A2, Phases 2-9) under headings with no id, status or priority; the two named blocks carry status `milestone`, no anchor and no priority, and still say 0.42.0 while crates.io lists `leto` 0.47.0 and `leto-ops` 0.46.4 and PyPI `leto-python` 0.46.1 (looked up 2026-11-13); `checklist.md` (280 lines) is a checklist file (`board_checklist_files` 1) that holds the stable-release acceptance cited by `LETO-STABLE-QUALIFICATION` and `LETO-REGISTRY-SCALAR-PROVIDER`.
- Outcome: each remaining `[x]` line is checked against its landing commit and deleted; each `[ ]` line becomes a DoR item or is deleted with a reason (ADR 0001 operator overloading and ADR 0012 dqds already have records); the two blocks are verified against the registries and the trusted-publisher settings, then deleted or narrowed; `checklist.md` folds into items, with its stable-release acceptance moving into those items, and is deleted.
- Acceptance: `board_checklist_files` 0; no board heading without an id, status and priority; `atlas-board-lint.py` passes.
- Next step: verify Stage A2 and Phases 2-3, then the two registry blocks, against the tree and registries, in one slice.

<a id="LETO-HERMES-DISPATCH-UNARY-SCAN-2026-11-24"></a>
## LETO-HERMES-DISPATCH-UNARY-SCAN-2026-11-24 — Unary math and prefix scans bypass Hermes [patch] — todo
- priority: tightening; milestone: M-array-core
- needs: none
- basis: 9d322c397b8096875f8f6a8826a1e218c2d9b33d
- scope: `crates/leto-ops/src/application/unary/`, `crates/leto-ops/src/application/scan.rs`, `crates/leto-ops/src/domain/scalar/contract.rs`
- Evidence: the 0.14.3 hot-kernel audit (legacy Stage C2, retired in leto#439) listed the non-dense strided norm fallback, scans and unary math as not dispatched through Hermes. At 9d322c39 `git grep -n hermes_simd` over `application/unary` and `application/scan.rs` finds nothing, and the `Scalar` slice hooks in `contract.rs` cover elementwise arithmetic, scale, sum, dot, axpy, gemv, tiled gemm and min/max, with none named for unary functions or scans; the strided norm fallback is not re-checked here. The same audit named a Hermes fused square-accumulate kernel, to be pursued only if profiling shows the dot self-alias overhead of `norm_l2` (`Σx²` through Hermes `dot`) is material; no such profile is recorded.
- Outcome: for each of unary elementwise math, prefix scans and the strided norm fallback, the surface routes through a Hermes kernel with a Criterion comparison against its recorded baseline (`docs/benchmarks.md`), or the missing Hermes operation is filed on the Hermes board and cited here with the reason Leto keeps its loop; the profile-triggered fused square-accumulate kernel for `norm_l2` is carried in `docs/benchmarks.md` (Open measured targets) and outlives this item.
- Acceptance: one recorded outcome per surface, each a merged routing change with median and confidence interval or a Hermes board item; the existing value tests pass unchanged.
- Next step: list the `hermes_simd` public operations against `unary/operators/` and `scan.rs` and record which have a provider.

<a id="LETO-ZIP-PROVIDER-SURFACE-CONSUMERS-2026-11-24"></a>
## LETO-ZIP-PROVIDER-SURFACE-CONSUMERS-2026-11-24 — Seven Kwavers-driven zip and coordinate names have no in-stack consumer [major] — todo
- priority: tightening; milestone: M-array-core
- needs: none
- basis: 9d322c397b8096875f8f6a8826a1e218c2d9b33d
- scope: `crates/leto-ops/src/application/zip/{coordinate,fold,map}.rs`, the re-exports in `application/mod.rs`, `application/zip/mod.rs` and `lib.rs`, `crates/leto-ops/tests/ops/structure_ops.rs` and `structure_ops/`
- Evidence: `indexed_map4_inplace`, `indexed_fold`, `indexed_fold_fortran`, `coordinate_map_inplace`, `coordinate_map_plan`, `coordinate_map_plan_inplace` and `CoordinateMapPlan` were added for Kwavers (legacy Stage A0, retired in leto#439). At 9d322c39 `git grep -w` for each name matches only leto-ops sources and tests. On 2026-11-24, on the fetched default branch of all 27 other registered members (each equal to its remote HEAD), the same search finds none of the seven. None has a doctest or example. The retired entry recorded the coordinate plan as unconsumed until Kwavers' focused-test runtime is profiled below 30 s.
- Outcome: each name has a stack consumer, a doctest or example, or is deleted with its tests; deleting a published item is a break batched into the next major.
- Acceptance: `git grep -w` over every member's fetched default branch finds a consumer for each kept name; each deleted name is absent from `crates/`, and the next major's migration notes list it.
- Next step: ask the Kwavers board whether it plans a call site for any of the seven names.

<a id="LETO-FIXED-MATRIX-VECTOR3-MUL-2026-11-24"></a>
## LETO-FIXED-MATRIX-VECTOR3-MUL-2026-11-24 — `FixedMatrix<T, 3, 3> * geometry::Vector3<T>` is claimed but absent [minor] — todo
- priority: feature; milestone: M-array-core
- needs: none
- basis: 9d322c397b8096875f8f6a8826a1e218c2d9b33d
- scope: `crates/leto/src/geometry/operators.rs`, `crates/leto/tests/core/geometry.rs`, the `CHANGELOG.md` line that claims the operator
- Evidence: a retired Stage A0 `[x]` line (leto#439) and `CHANGELOG.md:1050` state the operator exists. At 9d322c39 `git grep -n -E 'Mul<.*Vector' -- crates/leto/src` finds six impls; none has `FixedMatrix` on the left and `geometry::Vector<T, N>` on the right (`Mul<FixedVector<T, C>> for FixedMatrix`, `fixed/matrix/ops.rs:91`, takes the other vector type). No commit reachable from origin/main adds it (`git log -G'Mul<[^>]*Vector'`). CFDrs works around it at `crates/cfd-core/src/geometry/mesh/operations.rs:44` (cbaf26ac).
- Outcome: implement `Mul<geometry::Vector3<T>> for FixedMatrix<T, 3, 3>`, which a consumer needs, rather than retract the claim; the `CHANGELOG.md` line becomes true by delivery. Priority `feature`: nothing computes a wrong value, so not `correctness`, and the operator is new public surface.
- Acceptance: a value-semantic test in `tests/core/geometry.rs`, over `f32` and `f64`, asserts `M * v` for a rotation and a non-symmetric matrix; it fails at the basis because the impl does not exist.
- Next step: add the impl beside `Mul<Vector3<T>> for Isometry3` in `geometry/operators.rs` with the test.

<a id="LETO-GROWTH-GAMMA-TILDE-SSOT-2026-11-16"></a>
## LETO-GROWTH-GAMMA-TILDE-SSOT-2026-11-16 — Write the `γ̃` exponent once in the graded scan [patch] — todo
- priority: tightening; milestone: M-array-core
- needs: none
- basis: 274da18e1cff2d123c5dca9e05b848374bc3eb6e
- scope: `crates/leto-ops/tests/ops/graded_scan/coverage.rs`, `crates/leto-ops/tests/ops/backward_error/mod.rs`
- Evidence: `growth` (`coverage.rs:74-77`) repeats the exponent `λ√N·u + N·u²/(1−u)` that `gamma_tilde` exponentiates (`backward_error/mod.rs:119-122`); `growth_matches_gamma_tilde` (`:148`) pins
  one copy to the other within `ROUND_TRIP_ULPS = 8.0` (`:145`). Its doc names five operations, one rounding each and doubled for the libm functions; the judge sums them to 7 (F3), and the sum is 8 only
  if `sqrt` counts as libm, which the doc does not say. `git grep -c 'sqrt() \* u' -- crates/leto-ops/tests` finds the expression at two sites. Once the exponent is shared, a round-trip test checks
  only libm's `ln_1p(exp_m1(x))`, which no leto defect can fail, so the test is deleted, not retained. The crossing test at `:172` calls `growth`. Source: judge on leto#425, F2 and F3; leto#426, F1.
- Outcome: the exponent is written once, and no tolerance constant lacks its derivation.
- Acceptance: `git grep -c 'sqrt() \* u' -- crates/leto-ops/tests` totals 1; `gamma_tilde` and `derives_vacuous` call that one function; `growth`, `growth_matches_gamma_tilde`, `ROUND_TRIP_ULPS` and the
  `growth` crossing test (`:172`) are deleted; the `vacuity_threshold` tests of `coverage.rs` and the `assert_rows` tallies pass unchanged.
- Next step: make `gamma_tilde` exponentiate the shared exponent and read `derives_vacuous` from it, then delete the round-trip and `growth` crossing tests.

<a id="LETO-COVERAGE-RECORD-SIGNATURE-2026-11-16"></a>
## LETO-COVERAGE-RECORD-SIGNATURE-2026-11-16 — Type the arguments of `Coverage::record` [patch] — todo
- priority: tightening; milestone: M-array-core
- needs: none
- basis: 274da18e1cff2d123c5dca9e05b848374bc3eb6e
- scope: `crates/leto-ops/tests/ops/graded_scan/coverage.rs`, `crates/leto-ops/tests/ops/graded_scan/case.rs`
- Evidence: `Coverage::record(precision: i32, site, asserted: bool, counts, eps: f64)` (`coverage.rs:100-107`) takes a bare `bool` and both `precision` and `eps`, which are functions of the one
  format (`T::PRECISION`, `format::epsilon::<T>()` at `format.rs:51`). Callers: `case.rs:106`, `:123`, `:181`, `:198`. Source: judge on leto#425, F5.
- Outcome: a call names the format once and states whether the assertion ran by a two-variant type, so a precision and an epsilon of different formats cannot be passed together.
- Acceptance: `git grep -nE 'enum Assertion' -- crates/leto-ops/tests` finds one definition with two variants; `Coverage::record` has no `bool` parameter and every one of the four callers passes an
  `Assertion`, so a renamed `bool` fails; `record` takes the format as one argument from which precision and epsilon derive; the `assert_rows` tallies and
  `cargo nextest run -p leto-ops -E 'test(graded_scan)'` pass unchanged.
- Next step: replace the `bool` with `enum Assertion { Ran, Skipped }` and derive `precision` and `eps` from a `Format` type parameter.

## Atlas in-house replacement roadmap — leto slice [arch]
Cross-repo program to eliminate ndarray, nalgebra, rayon, tokio, std::simd, and
burn from the Atlas stack using monomorphized zero-cost in-house crates. SSOT
map: ndarray→leto, nalgebra→leto-ops linalg, rayon/tokio→moirai, std::simd→hermes,
burn→coeus, alloc→mnemosyne, capabilities→melinoe, GPU=wgpu+cuda-oxide behind
coeus `ComputeBackend`. leto owns the CPU array substrate and stays CPU-only; GPU
backends live in coeus/apollo and index leto-style host-side layout metadata.
### Stage A2 — ndarray consolidation (support coeus/apollo) — milestone: M-array-core
- [x] [patch] Extend stack fixed primitives for RITK spatial metadata:
  `FixedVector::iter`, `FixedMatrix::iter`, and 3-D row-major/column-major
  constructors/extractors. Verification: focused fixed primitive tests and
  RITK consumer spatial gates.
- [x] [minor] Add ndarray-stats variance/std parity for all-elements and axis
  reductions (`var_all`, `std_all`, `var_axis`, `std_axis`) with finite `ddof`
  validation and two-pass accumulation. Verification: closed-form population
  and sample cases, ndarray `var`/`std`/`var_axis` differential, invalid empty
  and non-positive/non-finite degrees-of-freedom rejection.
- [x] [minor] Add quantile/median parity for all-elements and axis reductions
  (`quantile_all`, `median_all`, `quantile_axis`, `median_axis`) with an
  `Interpolation` enum covering Linear/Lower/Higher/Nearest/Midpoint. Shared
  SSOT kernel sorts a caller-owned scratch slice; axis path reuses one scratch
  buffer across lanes. Verification: closed-form interpolation oracles,
  per-lane equivalence, unsorted input, empty/range/NaN rejection.
- [x] [minor] Add covariance/Pearson correlation parity for rowvar observation
  matrices (`covariance`, `pearson_correlation`). Shared degrees-of-freedom
  validation comes from the variance contract; covariance uses two-pass centered
  cross-products and correlation delegates to covariance. Verification:
  closed-form sample/population covariance, diagonal == `var_axis`, symmetry,
  perfect +/-1 correlation, normalized covariance identity, empty/ddof
  rejection.
- [ ] [minor] Provide any CPU kernel `coeus-leto` needs to retire coeus's
  duplicate traversal (reductions incl. argmax/cumsum already present; add gaps
  as coeus integration surfaces them).
- [ ] [patch] Keep ndarray strictly a dev-dependency differential oracle; core
  crates never depend on it in production.

## Phase 2: ndarray API Parity Required by Apollo [minor] — milestone: M-array-core
- [x] Add rank-specific aliases for `Array1`, `Array2`, `Array3` and corresponding view types. Verification: value test constructs `Array1` and `Array2` aliases and reads through views.
- [x] Add a stable `RankMarker` / `RemoveAxis` helper for rank-dropping shape and stride calculations over ranks 1 through 4. Verification: value tests cover rank-3 axis removal and out-of-bounds rejection.
- [x] Add `zeros`, `from_elem`, `from_vec`, `from_shape_fn`, `from_shape_vec`, and `into_vec` equivalents. Verification: value tests cover filled/generated/vector constructors, length mismatch rejection, and zero-copy contiguous `into_vec`.
- [x] Add axis iteration APIs that cover row/column traversal without forcing copies. Verification: value test iterates matrix rows as read-only subviews; mutable iterator rejects zero-stride aliasing layouts at construction.
- [x] Add named row and column convenience wrappers after axis iterator ergonomics are settled. Verification: value tests cover `rows`, `columns`, `rows_mut`, and `columns_mut` as zero-copy wrappers over the axis iterator implementation.
- [x] Add `mapv`/typed conversion APIs for scalar storage used by Apollo verification and Python outputs. Verification: value and ndarray differential tests cover caller-owned `map_into`, allocating `mapv`, explicit f64-to-f32 conversion, contiguous traversal, and strided transposed inputs.
- [x] Add mutable zip-map traversal for Apollo migration call sites. Verification: value tests cover contiguous shape-matched mutation, shape mismatch rejection, and strided transposed views.
- [x] Add representative Apollo complex-storage map fixtures for `Array1<Complex64>` to `Array1<Complex32>` and half-pair storage conversion. Verification: `migration_fixtures` covers generated complex arrays, caller-owned output storage, and `mapv` precision conversion without hidden widening.
- [ ] Add caller-owned output variants for all constructors and operations used in Apollo to preserve zero-copy and allocation control.
- [ ] Add differential tests against `ndarray` for every Apollo-facing API before replacing a downstream crate dependency. Current coverage includes map-style traversal, keep-dim axis reductions, and 2D matmul; remaining coverage must include all transform-specific Apollo migration fixtures.

## Phase 3: Coeus Tensor Substrate Requirements [minor] — milestone: M-array-core
- [ ] Add shape/stride/layout contracts suitable for tensor batches, channels, and rank-generic model activations.
  The iterator portion is now provider-complete: `&Array` and `&ArrayViewMut`
  implement logical, stride-aware `IntoIterator`; plain mutable yielding is
  available through fallible `try_iter_mut` and the indexed form, so aliased
  layouts are rejected before any `&mut T` escapes. Remaining work is the
  broader tensor-batch/channel contract, not basic iteration ergonomics.
- [x] Add representative broadcast semantics compatible with tensor elementwise operations, including keep-dim `[N, 1] -> [N, C]` read-only broadcast into elementwise add/mul. Verification: `migration_fixtures` covers Coeus normalization-like row reductions and broadcasted arithmetic.
- [x] Add reductions over axes with keep-dim output modes required by Coeus: `sum_axis_into`, `mean_axis_into`, `min_axis_into`, and `max_axis_into`. Verification: value and ndarray differential tests cover row/column reductions, strided transposed inputs, shape mismatch rejection, and empty-axis behavior.
- [x] Add allocating convenience wrappers for axis reductions only after storage constructors are complete. Verification: value tests cover contiguous row/column reductions, strided transposed input, C-contiguous output, and empty-axis sum/mean semantics.
- [x] Add 2D matmul coverage for contiguous inputs, transposed/strided inputs, caller-owned output, and differential parity against `ndarray`.
- [x] Resolve batched matmul ownership: the `gap_audit.md` §C boundary decision places rank-3 batch contraction in Leto; implementation tracked in Phase 6.
- [ ] Keep Leto non-differentiable. Coeus owns autodiff graph, gradient storage, and optimizer state; Leto owns layout/storage/views only.

## Phase 4: Operations, Performance, and Architecture [minor] — milestone: M-array-core
- [x] Replace duplicated elementwise functions with one generic binary traversal kernel selected by ZST operation markers. Verification: direct `binary_map::<AddOp>`/`binary_map::<MulOp>` tests and transposed strided-view elementwise test.
- [x] Extract shared logical flat-index conversion helpers for core constructors and leto-ops traversals. Verification: all constructor, map, elementwise, and reduction tests pass after the split.
- [x] Split matrix multiplication into its own module and documented each raw-pointer block with storage-span safety invariants. Verification: `leto-ops` focused tests and clippy pass.
- [x] [patch] Route dense-but-offset matmul views (batched `b>0`, sliced sub-array outputs) through the in-place fast kernels via offset-independent `is_c_dense`/`is_f_dense`, removing the per-batch scratch allocation + operand copy + copy-back; replace `batched_matmul`'s per-batch parallel `Mutex` poll with a relaxed `AtomicBool` early-out. Verification: 405 workspace tests incl. new offset-dense in-place test + batched/differential/parity oracles.
- [x] [patch] Fixed `batched_matmul`'s parallel closure full-buffer `&mut` aliasing: each task now borrows only its batch's physical span (`min_max_offsets`) with a rebased offset, so no two concurrent `&mut` overlap. A disjointness guard (`batch_stride ≥ per-matrix span`, plus non-empty) routes interleaved-batch outputs to the sound sequential loop. Verified by new interleaved-output (vs C-contiguous reference) and empty-output tests + the batched differential/parity oracles (407 workspace tests). (Per-row `parallel_dot/cc/outer` kernels were already disjoint.)
- [x] [patch] `LETO-KERNEL-BENCHMARKS-1`: Add contiguous fast paths and
  strided fallback benchmarks for elementwise ops, reductions, and matmul.
  **Owner:** Codex `/root`. **Claimed files:**
  `crates/leto-ops/benches/kernels.rs`, `checklist.md`, and the matching
  performance-audit evidence in `gap_audit.md`. The new rows use prepared
  C-dense and step-2 256×256 f64 views; the elementwise and sum rows are
  stable coverage evidence, while matmul remains noisy and is explicitly not
  an optimization claim.
- [ ] Verify Moirai scheduling uses bounded work partitioning without raw-pointer aliasing hazards.
- [ ] Integrate Hermes SIMD through sealed scalar/vector traits, not ad hoc per-operation dispatch.
- [ ] Keep Mnemosyne allocation optional and feature-gated; no downstream Apollo/Coeus crate should need allocator-specific types in public domain structs.

## Phase 5: Python and Interop [minor] — milestone: M-array-core
- [ ] Keep Python as a thin PyO3/NumPy boundary over Rust operations.
- [x] [patch] Resolve the reopened `numpy-0.23.0` rustdoc ICE in the Python FFI
  documentation path. `leto-python` is a PyO3 extension boundary, not a Rust
  library API surface, so Cargo no longer invokes rustdoc for that target
  (`doc = false`). Full workspace docs now complete without excluding
  `leto-python`. Verification: package docs, full workspace docs, package
  clippy, and package nextest all pass.
- [x] Replace current Python result construction that clones through `Vec` after computation. Verification: `leto-python` now transfers owned `VecStorage` with `Array::into_vec()` and `PyArray1::from_vec`, then reshapes without the former `as_mut_slice().to_vec()` clone path.
- [x] Add Python boundary tests for shape validation, C-contiguous input, rejected non-contiguous inputs, and value parity with NumPy-visible outputs. Verification: `leto-python` unit tests cover `add`, `sum`, `matmul`, shape mismatch rejection, and a real NumPy transposed non-contiguous input.

## Phase 6: Coeus Backend Consolidation [arch] — milestone: M-array-core
Source: `gap_audit.md` §C. Coeus delegates non-differentiable CPU array
operations to Leto while retaining its autodiff-integrated tensor/COW wrapper.
Coeus owns backend selection, autodiff, NN orchestration, optimizers, and higher
sparse formats; Leto owns CPU attention and narrow CPU sparse parity kernels,
while Hephaestus owns accelerator attention.
- [x] [major] Decide the const-rank vs dynamic-rank boundary: resolved in `docs/adr/0002-coeus-rank-boundary.md` — const-generic dispatch shim at the Coeus boundary; Leto stays const-rank; the shim lives in Coeus (consumer-owned). Phase 6 leto-side capabilities are authored const-rank.
- [x] [minor] Add a named unary math-op suite as ZST ops through the existing traversal kernel: `ExpOp`, `LnOp`, `SinOp`, `CosOp`, `SqrtOp`, `AbsOp`, `NegOp`, `RecipOp`, `PowfOp` via the `UnaryOp` trait and `unary_map`/`unary_map_into`, on the segregated `RealScalar` trait. Coeus's 17 activation/gradient `UnaryOp` variants compose from these in Coeus, not in Leto.
- [x] [minor] Add broadcast-aware binary ops that write through caller-owned output layouts. `binary_map`/`add`/`sub`/`mul`/`div` now broadcast each input layout to the caller-owned output shape when compatible, preserve the contiguous equal-shape fast path, reject aliased mutable output layouts, and cover Coeus `[N,1]`/`[1,C]` elementwise paths. Verification: value tests for dense and strided broadcast inputs plus ndarray differential broadcast add.
- [x] [minor] Add `reshape`/`into_shape` for contiguous arrays, `permute` (named alias over transpose semantics), and `to_contiguous` materialization. `Layout`, owned arrays, borrowed views, and mutable views now support dense row-major reshape; arrays/views can materialize strided, transposed, or broadcasted logical row-major data into canonical C-order storage. Verification: value tests for reshape/into_shape/reshape_mut/permute/to_contiguous plus ndarray contract coverage for reshape and strided materialization value order.
- [x] [minor] Add shape ops along an axis: `concat`, `pad`, `split` (leto core `application/structure/`). `concat`/`pad` allocate C-contiguous output reading logical row-major order; `split` returns zero-copy subviews. Verification: value tests incl. transposed-input concat and bad-size rejection.
- [x] [minor] Add `stack` (rank `N -> N+1`) via an `InsertAxis` rank helper mirroring `RemoveAxis` (ranks 0..=7, shared `RankMarker` ZST). `stack::<T, N, M>` inserts a new axis at `0..=N` and writes C-contiguous output in logical order. Verification: leading/trailing-axis, rank-2→3, transposed-input, and shape-mismatch tests.
- [x] [minor] Add batched rank-3 matmul (`batched_matmul`), dispatching each batch to the rank-2 `matmul` kernel; batch dim broadcasts when 1. Verification: explicit-batch and broadcast value tests, shape-mismatch rejection.
- [x] [minor] Add `cumsum`/prefix-scan along an axis: `scan_axis`/`scan_axis_into` with `CumSumOp`/`CumProdOp` and `ScanDirection` (Forward/Reverse), plus `cumsum`/`cumsum_into`. Verification: forward axis-0/axis-1 and reverse cumprod value tests.
- [x] [minor] Add deterministic seeded random constructors (`uniform_with_seed`, `normal_with_seed` via Box-Muller) over the `Xorshift64` PRNG domain type. Verification: determinism, range, and closed-form mean/variance for uniform and normal.
- [ ] [arch] Re-base Coeus's CPU storage/layout layer onto Leto types (or thin adapters) and delete the duplicate, as a coordinated cross-repo unit per the co-evolution protocol; file the consumer-side item in the Coeus backlog naming Leto as provider.

## Phase 7: ndarray Parity Completion (Apollo hot kernels) [minor] — milestone: M-ops-linalg
`LETO-FFT-LAYOUT-THROUGHPUT` [minor] is in progress. It provides the measured,
caller-owned layout-movement primitive required by Apollo's non-contiguous 2-D
and 3-D FFT axis passes. The implementation profiles the existing cache-tiled
gather/scatter loops first, adds one rank- and scalar-generic Leto operation
only when the profile confirms the provider boundary, replaces Apollo's
duplicated loops, and preserves zero steady-state allocation. FFT arithmetic,
Apollo pass scheduling, and GPU dispatch are non-goals. Acceptance covers
analytical and ndarray-differential permutation tests (rectangular, empty,
singleton, invalid-axis, aliasing, and failure-atomic cases), a same-address
Criterion observations against Apollo's current loop, reported without a
formal non-inferiority claim; Apollo 2-D/3-D round trips and oracle parity; and
the allocation census. Risk: public additive API and
throughput-sensitive memory ordering. Integrator: Codex
`01a0253c-6013-7552-99cc-36bbbcf77f6d`. Dependencies: merged Hermes provider
revision `bbc7bdb` and Apollo PR #125. Lease: `crates/leto-ops/src/application`
Rustdoc links and this item entry through the workspace-doc repair commit.
Provider commit `5410f47` adds the validated assignment kernel and its
adversarial layout tests. Baseline: checked `assign` is 15.3–29.0× slower
than Apollo's same-address tiled loop across four FFT shapes, with disjoint 95%
confidence intervals in both directions; see `gap_audit.md`. Last update:
2026-08-26.
Source: `gap_audit.md` §A. Apollo already exposes `forward_leto`/`inverse_leto` boundaries; these items unblock replacing ndarray inside the kernels.
- [x] [minor] Add contiguous-slice access on views: `as_slice`/`as_mut_slice` (now offset-independent C-dense) plus `as_slice_memory_order`/`as_mut_slice_memory_order` and `is_c_contiguous`/`is_f_contiguous`/`is_contiguous` queries (named Apollo FFT butterfly blocker). Value tests cover offset-contiguous subviews, F-order blocks, strided-gap rejection, and mutable offset-block writes.
- [x] [patch] Add `map_inplace` in-place unary mutation (Apollo 1/N normalization sites); memory-order fast path, zero-stride aliasing rejected.
- [x] [patch] Add 1D `dot` (contiguous fast path + strided fallback, native-precision accumulation).
- [x] [minor] Add scalar–array elementwise ops: `scalar_map`/`scalar_map_into` reusing `BinaryOp` markers.
- [ ] [arch] std::ops operator overloading on arrays/views: DEFERRED, see `docs/adr/0001-elementwise-operator-overloading.md` (orphan rule; revisit when a consumer driver exists; `scalar_map` covers the scalar case meanwhile).
- [x] [minor] Add 3+-operand zip traversal: `zip2_mut_with` (one mutable output + two read inputs), the `Zip::from(out).and(a).and(b)` analogue. Verification: fused multiply-add and strided-input value tests.
- [x] [minor] Add indexed mutable zip traversal: `indexed_zip_mut_with` and `indexed_zip2_mut_with`, the `Zip::indexed` analogue for one- and two-input mutable zip paths. Verification: dense logical-index and strided-transposed value tests.

## Phase 9: Blocked-reflector vectorization (eig/SVD disparity) [major] — milestone: M-array-core
Source: `docs/adr/0010-blocked-reflector-vectorization.md`; `gap_audit.md` eig/SVD residuals.
Residual disparity is structural: the Francis right-apply and the Givens bidiagonal-QR
apply *single* width-2/3 reflectors (bandwidth-bound, no contiguous SIMD span). The fix
is the compact-WY block reflector (Schreiber–Van Loan), applying `nb` aggregated
reflectors as `tiled_gemm` (BLAS-3). Phased, each verified against the unblocked oracle.
- [x] [patch] Phase 0: vectorize contiguous single-reflector sweeps via `axpy_slice`
  (Householder apply + Francis left-apply). eig 5.9×→4.4×, svd 4.1×→3.4×,
  singular_values 3.8×→2.3×. Backward-error eigenvalue tolerance correction
  (`8·√(ε‖A‖)`, machine-checked defective-eigenvalue derivation) unblocks blocked
  reorderings. Done (commits `5104a60`, `8df636f`).
- [x] [major] Phase 1: `linalg/reflector_block/{mod,accumulate}` — compact-WY
  block reflector (Schreiber–Van Loan `build_t` + `tiled_gemm` block apply),
  differential-tested vs `r` sequential applies + orthogonality. First consumer:
  panel-blocked `qr_decompose` (`dgeqrf`), gated on `BLOCK_MIN_ROWS = 256` (A/B
  crossover ≈ 200): 256² QR 1.51 → 1.29 ms, ≤128² byte-for-byte unchanged.
  Verified by a 256² known-`x` solve. Done (commit `c78b843`). Blocked Hessenberg
  folded into Phase 2.
- [x] [major] Phase 2: blocked U/V factor formation — implemented + verified (256²
  `A=UBVᵀ` reconstruction/orthogonality), then **reverted as measured-valueless**:
  256² full SVD 164 ms blocked vs 163 ms unblocked (the sequential Givens sweep is
  the whole cost; formation < 1 ms). `apply_block_right` reverted with it. Lesson:
  the SVD/eig residual is the iteration (Phase 3), not the reductions/factors.
- [x] [major] Phase 3 (SVD) DONE: the dominant `U`/`V` Givens accumulation made
  contiguous via transposed factors (each rotation mixes two contiguous rows,
  bitwise-identical). 256² full SVD 164 → 34.7 ms (4.7×, now faster than nalgebra);
  64² 1.31 → 0.60 ms. Commit `9bef76e`. SVD disparity resolved at scale.
- [x] [major] Phase 3 (eig) DONE — disparity resolved to near parity (1.16×) by the
  `dlahqr` WANTT=false within-block apply window (commit `676ff72`), unblocked by the
  Phase-0 backward-error tolerance fix. 64² confinement 2.69 → restriction 0.69 ms
  (3.9×). The full multishift (`dlaqr5`) GEMM rewrite is no longer needed to close
  the disparity; it remains an optional future lever for very large `n` only.
- [ ] [major] SVD values-only **dqds** fast path (ADR 0012). Replace the implicit-shift
  Givens sweep in `singular_values` with a full `dlasq`-class dqds (block splitting +
  Fernando–Parlett shift cases + ping-pong). NOTE: the 64² ~1.9× nalgebra gap is NOT
  algorithmic — nalgebra uses the same Givens sweep (verified); the gap is a per-step
  implementation constant. dqds (0√+1÷ vs 2√+2÷) is an *absolute* speedup lever that
  would beat both, not the explanation for nalgebra's lead. DoR: prototype reverted
  (no-split regresses; naïve splitting breaks rank-deficient). DoD: differential parity
  across the battery + adversarial clustered/tiny/zero/wide-range inputs, AND a measured
  64²/256² win before merge (asymptotic-only is insufficient at n=64).
