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
- Next step: draft the ADR comparing dependency direction, API and test parity, and the one consumer.

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
- basis: b738a4e2
- scope: `crates/leto-ops/tests/ops/backward_error/`, `graded_scan.rs`, `graded_scan/case.rs`, `symmetric_qr/fixtures.rs`; the id is cited at `schur/skew_ties.rs`
- Evidence: delivered are `gamma_tilde` (Higham and Mary, SIAM J. Sci. Comput. 41(5), 2019, Theorem 2.4, eqs. 2.1, 2.3, checked against the MIMS EPrint 2018.33 preprint, journal pages from citing literature; `PROBABILISTIC_LAMBDA` 7) over instrumented sweep, chase and step counts, with every factor assertion behind `informative(...)`. A temporary logging `informative` in an export (not committed) over the 46 `ops_tests` tests of the graded scan, Schur, symmetric QR and pivoted QR at b738a4e2 counted the cases whose bound is below 1, as scan cases and large-order cases (the 18 cases of `large_orders_factor_in_the_narrow_formats`, F16 orders 16-48 and Bf16 32 and 64, are the tail of the per-site totals 174 and 1512). F16: SVD residual and orthogonality 162 of 162 scan, 0 of 12 large; Schur residual 148 of 162 and 0 of 12 (the 14 vacuous scan cases are all at n = 8); Schur orthogonality 162 of 162 and 0 of 12; symmetric-QR residual and orthogonality 6 of 7; pivoted-QR residual 2 of 2. Bf16: SVD residual and orthogonality 0 of 1506 and 0 of 6; Schur residual 0 of 1506 and 0 of 6; Schur orthogonality 272 of 1506 (all 215 cases at n = 2, 57 of 215 at n = 3) and 0 of 6; symmetric-QR 0 of 7; pivoted-QR 0 of 2. The QR rows are the PR 415 figures, not re-probed.
- Bound: at `u = 2^-8` and `λ = 7`, `γ̃_k < 1` holds exactly for `k ≤ 624` (the first-order `λ√k·u < ln 2` gives 642 and is necessary only), and one length-3 reflector costs 24 (`γ̃_24 = 0.144`). The probe finds Bf16 sites vacuous except Schur orthogonality at n = 2 and 3; the rounding counts that exceed 624 at the other sites are not tabulated, so the vacuity is observed, not derived.
- Outcome: for each F16 and Bf16 factor site (SVD, Schur, symmetric QR, pivoted QR) in the scan band and the large-order band, the residual and orthogonality are asserted by an oracle informative there, or the site is recorded vacuous by its derived rounding count and covered by the a-posteriori certificates; the original alternative is the deterministic bound at an instrumented transformation count.
- Acceptance: a committed counting test reports, per (format, site, band), cases below the bound; each row is nonzero or carries its derived zero, with the F16 Schur residual at n = 8, symmetric QR and `large_orders_factor_in_the_narrow_formats` named rows; `PROBABILISTIC_LAMBDA` stays 7 (a smaller `λ` raises the per-product failure probability `2exp(-λ²(1-u)²/2)`, a larger one widens the bound); no tolerance widening.
- Next step: commit the counting test, then instrument the per-entry rounding count of the Bf16 Francis and Golub-Kahan paths and tabulate the orders at which `γ̃` falls below 1.

<a id="LETO-VECSTORAGE-UNINIT-2026-11-13"></a>

## LETO-VECSTORAGE-UNINIT-2026-11-13 — Safe `VecStorage::uninit` exposes uninitialized memory [major] — todo
- priority: correctness; milestone: M-array-core
- needs: none
- basis: b738a4e2
- scope: `crates/leto/src/infrastructure/storage/vec.rs`, the `leto` crate-root re-export (`lib.rs:31`), and its 7 call sites: `leto-ops` `embedding.rs:115`, `reduction/axis.rs:217`, `reduction/topk.rs:154-155`, `triangular.rs:189`, `volume.rs:216`, and `leto` `structure/pad.rs:98`
- Evidence: `VecStorage::uninit` is a safe `pub fn` that `set_len`s a `Vec::with_capacity` (`vec.rs:48-58`), contrary to the `Vec::set_len` precondition, with a `# Safety` doc on a safe fn. Miri (nightly bff8e12ff 2026-08-26, `-Zmiri-disable-isolation`) on b738a4e2 with a test holding no `unsafe` block, `VecStorage::<f64>::uninit(4)` then `as_slice()[0]`, reports UB: reading uninitialized memory. This shows the public API is unsound from safe code, not that `reduce_axis` executes UB: its write-before-read coverage proof for the three routes is in `git show b738a4e2:backlog.md` (`ATLAS-LETO-REDUCE-SINGLE-WRITE`), and the `miri` job runs the serial reduction. Whether a `&mut [T]` over uninitialized memory is itself UB is open in the Rust reference (hypothesis). Introduced by d16e3b2 (PR 157), which removed the zero-fill of `reduce_axis`; that removal and `ATLAS-LETO-OPERATOR-OWNED-LHS` (62a0434) landed.
- Outcome: no safe code path reaches uninitialized `T`: `VecStorage::uninit` becomes `unsafe fn` whose callers discharge a `// SAFETY:` write-before-read obligation, or each call site writes through a `MaybeUninit<T>` output slot generic over one traversal (the deleted block's direction, with no duplicated body).
- Acceptance: a `compile_fail` doctest shows `VecStorage::<f64>::uninit(4)` outside `unsafe` is rejected, or the symbol is gone; `git grep -n 'VecStorage::uninit'` lists only sites with a `SAFETY` proof, or none; the Miri `reduction` steps stay green under both models; `cargo-semver-checks` records the break under [major].
- Next step: choose between `unsafe fn` and the output slot by the cost of the zero-fill it removed (`alloc_zeroed` for zero primitives, per the deleted block), then migrate the 7 sites.

<a id="LETO-MIRI-GATE-2026-09-10"></a>

## LETO-MIRI-GATE-2026-09-10 — Widen the Miri gate past the storage substrate and the uninitialized reduction output [patch] [safety] — todo
- priority: verification; milestone: M-array-core
- needs: none
- basis: b738a4e2
- scope: `.github/workflows/ci.yml` (`miri` job), `.config/nextest.toml` (`[profile.miri]`), `crates/leto-ops/tests/ops/`
- Evidence: the job runs `leto --lib` and `leto-ops ops_tests -E test(/reduction/)` under both Stacked and Tree Borrows (`ci.yml:263-282`), the second excluding `parallel_axis_reduction_matches_the_serial_result` (`ci.yml:259-262`: the 40000x8 input exceeds the 600 s profile bound, so Miri covers the serial output path only). The serial route exercises the write-before-read ordering of `reduce_axis` over `VecStorage::uninit` (d16e3b2); it does not make `VecStorage::uninit` sound, which is the open `correctness` item LETO-VECSTORAGE-UNINIT-2026-11-13. Measured 2026-09-10 (Windows host, Miri 0.1.0, nightly 2026-08-26): `leto --lib` 130 tests 206 s Stacked, 363 s Tree; `leto-ops --lib` (216 tests), `leto-ops --test ops_tests` (352) and `leto layout_property_tests` each exceed 10 min, so they stay out as unaffordable, not as believed sound.
- Open observations, unconfirmed on Linux: `complex_transpose_allocations` aborts under Miri inside the test's own `CountingAllocator` (Windows `HeapFree`, `GetNumaHighestNodeNumber`); two `leapfrog` tests fail an exact `assert_eq!` under Miri with the native/interpreted float divergence unexplained.
- Outcome: each excluded surface, the parallel route of the uninitialized reduction output included, joins the job with a measured bound or is closed with the reason it stays out; both borrow models stay (an MN-437 fix passed Stacked Borrows while Tree Borrows still rejected it).
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
- basis: b738a4e2
- scope: `backlog.md` from `## Atlas in-house replacement roadmap` to its end, the `LETO-CRATES-METADATA-1` and `LETO-PYTHON-RELEASE-1` blocks, and `checklist.md`
- Evidence: at b738a4e2 the tail holds 95 `[x]` and 13 unchecked `[ ]` lines under headings with no id, status or priority; the two named blocks carry status `milestone`, no anchor and no priority, and still say 0.42.0 while crates.io lists `leto` 0.47.0 and `leto-ops` 0.46.4 and PyPI `leto-python` 0.46.1 (looked up 2026-11-13); `checklist.md` (258 lines, copies of both blocks) is a checklist file (`board_checklist_files` 1).
- Outcome: each `[x]` line is checked against its landing commit and deleted; each `[ ]` line becomes a DoR item or is deleted with a reason (ADR 0001 operator overloading and ADR 0012 dqds already have records); the two blocks are verified against the registries and the trusted-publisher settings, then deleted or narrowed; `checklist.md` folds into items and is deleted.
- Acceptance: `board_checklist_files` 0; no board heading without an id, status and priority; `atlas-board-lint.py` passes.
- Next step: delete the all-`[x]` stages S0, A0 and A1, citing each landing commit from `git log --grep`.

## Atlas in-house replacement roadmap — leto slice [arch]
Cross-repo program to eliminate ndarray, nalgebra, rayon, tokio, std::simd, and
burn from the Atlas stack using monomorphized zero-cost in-house crates. SSOT
map: ndarray→leto, nalgebra→leto-ops linalg, rayon/tokio→moirai, std::simd→hermes,
burn→coeus, alloc→mnemosyne, capabilities→melinoe, GPU=wgpu+cuda-oxide behind
coeus `ComputeBackend`. leto owns the CPU array substrate and stays CPU-only; GPU
backends live in coeus/apollo and index leto-style host-side layout metadata.
### Stage S0 — scalar SSOT audit
- [x] [patch] Add the Atlas special-functions provider lane:
  `leto_ops::{ErfOp,ErfcOp,LgammaOp}` over Eunomia `FloatElement`, with f64
  value-semantic coverage against known `erf`/`erfc`/`lgamma` values. Driver:
  Coeus exact GELU and `torch.special` parity surfaces.
- [x] [major] Rebase `leto-ops::Scalar` and `RealScalar` onto Eunomia
  supertraits instead of re-owning numeric constants, arithmetic/bit contracts,
  finite predicates, and real transcendental methods. Leto keeps only
  operation-local slice/SIMD hooks and `Scalar::from_usize`. Downstream fallout
  stays in Apollo/Coeus: use `eunomia::NumericElement` /
  `eunomia::FloatElement` directly for removed UFCS constants/constructors. No
  Leto compatibility shims.
- [x] [minor] Extend Eunomia's primitive numeric SSOT to `isize`/`usize`, then
  re-enable the corresponding `leto-ops::Scalar` impls through the Eunomia
  supertrait contract. Platform-sized scalar support remains upstream-owned; no
  Leto compatibility shims.
### Stage A0 — consumer-driven geometry and array surface
- [x] [minor] Add the Helios-driven checked rotation-column constructor:
  `UnitQuaternion::try_from_rotation_columns` validates finite, right-handed,
  orthonormal world-space axes without silently projecting an affine basis to a
  rotation. Helios consumes it for `ImageOrientationPatient` grid poses.
  Verification: generic `f32`/`f64` rotation tests plus invalid-basis tests,
  package fmt/check/clippy/nextest/doc, and repository-baseline SemVer checks.
  The downstream oblique DICOM-grid test remains Helios-owned and is sequenced
  behind RITK's named `ImageOrientationPatient` attribute contract.
- [x] [patch] Add the CFDrs sparse-extension CSR utility provider surface:
  `CsrMatrix::diagonal`, `scale_values`, `scale_rows`, `scale_columns`,
  `frobenius_norm`, `is_strictly_diagonally_dominant`, and
  `condition_estimate`. Driver: CFDrs `SparseMatrixExt` can move remaining
  CSR utility loops out of `cfd-math` and into Leto-owned CSR storage while
  the downstream public sparse storage boundary is migrated separately.
  Verification: provider fmt/check/clippy and focused sparse nextest (18/18),
  plus downstream cfd-math fmt/check/focused sparse nextest/all-target clippy.
- [x] [patch] Add the CFDrs AMG-driven CSR transpose provider surface:
  `CsrMatrix::transpose()` constructs `A^T` with sorted CSR rows and no dense
  materialization. Driver: CFDrs AMG restriction construction can move off
  `nalgebra_sparse::transpose_as_csc` while preserving Leto-owned CSR
  products. Verification: provider fmt/check/clippy/doc, focused sparse
  nextest (16/16), and downstream cfd-math fmt/check/focused sparse+AMG
  nextest/all-target clippy.
- [x] [patch] Add the CFDrs AMG-driven CSR×CSR sparse product provider
  surface: `leto_ops::spgemm` multiplies two CSR matrices through Leto-owned
  row accumulation, exports the operation at crate root, and adds `CsrRow::nnz`
  for sparse-pattern consumers. Driver: CFDrs AMG Galerkin products can move
  off `nalgebra_sparse` instead of preserving a downstream sparse multiply.
  Verification: provider fmt/check/clippy/doc and focused sparse nextest.
- [x] [patch] Add CFDrs mesh-rotation provider support:
  `FixedMatrix<T, 3, 3> * leto::geometry::Vector3<T>`. CFDrs uses this to move
  `cfd-core::geometry::mesh` transforms from nalgebra `Matrix3`/`Vector3` to
  Leto fixed geometry without a downstream helper. Verification: provider
  fmt/check/clippy/full nextest (171/171), downstream cfd-core no-default
  check/clippy/full nextest (201/201), and clean downstream mesh/staggered
  provider scans.
- [x] [patch] Add the CFDrs Domain-driven `Point1<T>` fixed geometry primitive,
  conditional `Eq` derives for fixed geometry values, and serde feature
  propagation for `std`/`alloc`. CFDrs uses this provider contract to migrate
  `cfd-core::geometry::shapes::Domain` and boundary/domain geometry from
  nalgebra point/vector/scalar contracts to Leto/Eunomia without a downstream
  wrapper. Verification: provider fmt/check/clippy, full provider nextest
  (170/170), downstream cfd-core no-default check/clippy, full downstream
  cfd-core no-default nextest (201/201), and clean migrated-cone scans.
- [x] [patch] Add the CFDrs state-driven owned-array serde provider surface.
  `Array<T, S, N>`, `VecStorage<T>`, and `Layout<N>` now serialize and
  deserialize without a downstream wrapper; array deserialization validates the
  decoded layout against storage through `Array::new`. `Layout<N>` serde now
  serializes shape/stride slices and validates decoded rank manually, so ranks
  above serde's fixed-array impl limit compile without a downstream wrapper.
  Verification: provider fmt, focused value-semantic serde nextest, provider
  clippy, downstream `cfd-core` no-default check/clippy/state nextest, and the
  Kwavers-driven rank-33 layout serde regression.
- [x] [patch] Add the CFDrs FVM-driven `Vector2<T>` fixed geometry alias plus
  generic fixed-vector norm and normalization methods so FVM face
  centers/normals and velocity fields can use Leto geometry instead of
  nalgebra `Vector2`. Verification: provider compile and focused
  value-semantic nextest, downstream `cfd-2d` compile, and downstream focused
  FVM nextest.
- [x] [patch] Add CFDrs-driven Serde derives to fixed geometry value types
  (`Point2`, `Point3`, `Vector3`, `UnitVector3`, and `Isometry3`) so serialized
  consumer domain values can use Leto geometry directly instead of retaining
  nalgebra or adding downstream wrapper types. Verification: provider
  touched-file rustfmt, `cargo nextest run -p leto geometry`, and downstream
  `cargo check -p cfd-core`.
- [x] [patch] Add the Kwavers FWI-driven four-read-view mutable zip provider
  surface (`leto_ops::zip3_mut_with`) so consumers can replace
  `ndarray::Zip::from(out).and(a).and(b).and(c)` at the provider boundary.
  Verification: dense fused second-difference and strided logical-order value
  tests, no-default consumer-feature `cargo check`/clippy, and downstream
  Kwavers FWI time-domain nextest.
- [x] [patch] Add the Kwavers FWI-driven two-read-view reduction provider
  surface (`leto_ops::zip_fold`) so consumers can replace two-array
  `ndarray::Zip` reductions at the provider boundary instead of adding local
  compatibility helpers. Verification: contiguous, strided logical-order, and
  shape-mismatch value tests, no-default consumer-feature `cargo check`/clippy,
  and downstream Kwavers FWI time-domain nextest.
- [x] [patch] Add the Kwavers self-adjoint FWI-driven multi-read provider
  surfaces (`leto_ops::zip5_mut_with` and `indexed_zip4_mut_with`) so consumers
  can replace reconstructed/stored-history imaging-condition `ndarray::Zip`
  paths at the provider boundary. Verification: contiguous, strided
  logical-order, and logical-index value tests, no-default consumer-feature
  `cargo check`/clippy, and downstream Kwavers FWI time-domain nextest.
- [x] [patch] Add the Kwavers FWI-driven one-view indexed mutable provider
  surface (`leto_ops::indexed_map_inplace`) so consumers can replace indexed
  mutable test-helper traversals at the provider boundary. Verification:
  logical-index value test, no-default consumer-feature `cargo check`/clippy,
  and downstream Kwavers FWI time-domain nextest plus source audit.
- [x] [patch] Add the Kwavers FWI-driven all-elements signed extrema provider
  surface (`leto_ops::{min,max}`) so consumers can replace model-range
  reductions at the provider boundary instead of adding downstream ndarray
  helpers. Verification: contiguous, sliced logical-view, and empty-input error
  value tests, no-default consumer-feature `cargo check`/clippy, and downstream
  Kwavers FWI time-domain nextest.
- [x] [patch] Add the Kwavers FWI-driven one-view indexed reduction provider
  surface (`leto_ops::indexed_fold`) so consumers can replace
  `indexed_iter().fold` reductions at the provider boundary instead of adding
  downstream ndarray helpers. Verification: logical-index and strided
  logical-order value tests, package `cargo check`, and downstream Kwavers FWI
  time-domain nextest.
- [x] [patch] Add the Kwavers MOFI-driven four-output indexed mutable provider
  surface (`leto_ops::indexed_map4_inplace`) so consumers can fill related
  model/Jacobian buffers in one provider-owned coordinate traversal instead of
  adding downstream loops or helpers. Verification: logical-index multi-output
  value test, package clippy, and downstream Kwavers MOFI nextest.
- [x] [patch] Add the Kwavers FWI-driven Fortran-order indexed reduction
  surface (`leto_ops::indexed_fold_fortran`) so consumers can preserve
  recorder/source column-major row-order contracts at the provider boundary.
  Verification: column-major logical-order value test, package clippy, and
  downstream Kwavers FWI time-domain nextest.
- [x] [patch] Add the Kwavers self-adjoint FWI-driven sparse coordinate mutable
  provider surface (`leto_ops::coordinate_map_inplace`) plus the prevalidated
  `CoordinateMapPlan` companion so consumers can inject source and residual
  terms through provider-owned logical-coordinate traversal instead of
  downstream coordinate loops. Verification: repeated-coordinate order,
  out-of-bounds, and plan-layout-mismatch value tests, package check/clippy,
  and downstream Kwavers self-adjoint/FWI time-domain nextest. Kwavers currently
  consumes the direct sparse map path; planned consumption stays provider-ready
  but unconsumed until its focused-test runtime is profiled below 30 s.
- [x] [minor] Add the Gaia/Kwavers-driven fixed-vector, fixed-matrix, and small
  geometry primitives (`Point3`, `Vector3`, `UnitVector3`, `Isometry3`) plus
  the owned-array convenience methods required for Atlas consumers to replace
  ndarray/nalgebra at the provider boundary. `FixedMatrix<T, 3, 3>::try_inverse`
  now covers the tetrahedral-Jacobian inverse needed by Kwavers FEM geometry.
  Verification: focused value tests for fixed/geometry primitives and array
  indexing/fill/map/zip semantics; current inverse evidence is package
  check/clippy plus `cargo nextest run -p leto fixed_matrix_inverse`.
- [x] [patch] Add the Kwavers CPML-driven rank-1 `Array1` `usize` indexing and
  owned-array `PartialEq`/`Eq` semantics so consumers can replace ndarray
  `Array1` profile/factor storage at the provider boundary. Verification:
  focused value tests for mutation, shape-sensitive equality, downstream
  CPML/PSTD/PML nextest, and downstream boundary/GPU compile/lint gates.
### Stage A1 — nalgebra linalg completion (leto-ops `application/linalg/`)
Each routine generic over `T: RealScalar`, native-precision accumulation (wider
accumulator only via a trait-encoded associated type with numerical justification),
admitted only with a named consumer driver (coeus/apollo) and a differential
oracle (nalgebra / ndarray-linalg as dev-dependency). SRP leaf modules.
- [x] [patch] Vector/matrix norms over `RealScalar`: `NormKind` ZST markers (`NormL1`/`NormL2`/`NormMax`) through one generic `norm` traversal in `application/linalg/norms.rs`; `norm_l2` covers Euclidean (rank-1) and Frobenius (rank-2+) in one entry point. Eigensolver consolidated into `linalg/` (re-export paths stable). Verification: nalgebra differential oracle, strided layout-independence, empty-view, and exact f16 tests.
- [x] [minor] Eigenvalues-only symmetric Jacobi API (`symmetric_eigenvalues_jacobi` and tolerance variant): shares the same diagonalization kernel as the full decomposition via a monomorphized `RotationTarget`; the eigenvalues path uses a zero-sized no-vector target and avoids `n*n` eigenvector storage. Verification: value tests for full-vs-values parity, strided input, closed-form eigenvalues, and invalid input rejection.
- [x] [minor] LU with partial pivoting (`linalg/lu.rs`): `lu_decompose`/`LuDecomposition<T>` (packed factors, pivots, parity) with `solve`, `det`, `inv` — generic over `RealScalar`, native precision. Driver: CFDrs `cfd-math`. Verification: nalgebra oracle, pivot parity, `inv·A=I`, `det(Aᵀ)=det(A)` via strided view, singular/non-finite rejection, f32 genericity.
- [x] [minor] QR (Householder) + least-squares solve (`linalg/qr.rs`): compact packed reflectors, Q never materialized, least-squares via reflector application + back-substitution. Oracle: nalgebra SVD (independent path) + LU cross-check + residual-orthogonality property.
- [x] [minor] Cholesky (SPD) factorization + solve/det/inv (`linalg/cholesky.rs`): lower-triangle-only reads, constructive positive-definiteness verification, determinant from `Π diag(L)^2`, inverse through identity-column solves over the same triangular substitution helper. Oracle: nalgebra cholesky().l()/determinant + LU cross-check + `A·A⁻¹=I` + strided symmetry invariance.
- [x] [patch] Wide full-row-rank thin SVD support: `svd_decompose` uses `A Aᵀ` for wide matrices and derives `V = Aᵀ U Σ⁻¹`; tall/square inputs keep `Aᵀ A`. Verification: value-semantic wide reconstruction, singular-value ordering, and right singular-vector orthonormality tests.
- [x] [patch] Rank-deficient singular-values-only support: `singular_values` diagonalizes the smaller Gram matrix and maps near-zero eigenvalues to zero singular values without constructing missing null-space vectors. `svd_decompose` still rejects rank-deficient matrices until a rank-revealing vector contract exists. Verification: tall and wide rank-deficient value tests.
- [x] [major] Rank-revealing SVD via one-sided Jacobi plus rank-deficient pseudoinverse; ADR 0005 records the selected algorithm and verification plan. Verification: reconstruction, right-vector orthonormality, nalgebra singular-value parity, nalgebra `pseudo_inverse` parity, and Moore-Penrose identities.
- [x] [minor] Non-symmetric eigenvalues (real + complex) through Hessenberg + shifted complex QR; ADR 0006 records the staged eigensolver track. Verification: nalgebra `complex_eigenvalues` battery and exact spectra. Schur vectors remain a separate open surface.
- [x] [minor] Unpivoted symmetric indefinite `U D Uᵀ` factorization (`udu_decompose`, `MatrixDecompose::udu`) with solve/inverse/determinant helpers. Verification: reconstruction, determinant/solve/inverse parity with nalgebra, invalid-input and zero-pivot rejection. Pivoted Bunch-Kaufman remains open for matrices requiring symmetric pivoting.
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
### Stage C2 — hermes SIMD coverage audit
- [x] [patch] Audit leto-ops hot kernels (matmul inner loop, reductions, scans,
  unary math) to ensure they dispatch through hermes `SimdOps` rather than
  ad-hoc scalar loops; file hermes coverage requests for any missing op/dtype.
  Dense f32/f64 `norm_l2` now routes `Σx²` through Hermes dot via
  `Scalar::dot_slice` (28.07 µs → 5.508 µs for 64k elements). Remaining
  coverage gaps: non-dense strided norm fallback, scans, unary math, matmul
  inner loops, and a future Hermes fused square-accumulate kernel if profiling
  shows dot self-alias overhead is material. Audit result (0.14.3): current
  Hermes public surface covers dense pairwise elementwise ops and dense
  sum/dot/min/max; no zero-allocation scalar-AXPY/fused row-update API is
  available for matmul. Rejected measured Leto-local candidates: const-generic
  dense blocking regressed `64x64` to ~48.5 µs and `256x256` to ~3.37 ms;
  a generic `mul_add` hook regressed `64x64` to ~245.6 µs and `256x256` to
  ~12.5 ms. Do not retry these paths without a changed kernel model.
- [x] [minor] (0.16.0) Consume the Hermes scalar-AXPY / fused row-update
  provider (`hermes_simd::axpy`, delivered hermes 51131a6): `Scalar::axpy_slice`
  routes the matmul unit-stride row update through Hermes fmadd lanes with no
  temporary allocation. Measured: `matmul/dense_256x256` 2.210 ms → 1.529 ms
  (−31%); `dense_64x64` unchanged within noise. The sum reduction also gained a
  dense memory-order fast path: `sum_transposed_256x256` 44.9 µs → 4.48 µs
  (−90%), matching the norm path.
- [x] [minor] (0.17.0) Consume Hermes absolute-value reductions for dense
  `norm_l1`/`norm_max`: defaulted `RealScalar::{abs_sum_slice, abs_max_slice}`
  hooks keep reduced-precision scalar fallback while f32/f64 route through
  `SimdOperations` to `hermes_simd::{abs_sum, abs_max}` with no temporary
  allocation. Measured in-run against scalar-fold references:
  `norm_l1_64k` 34.174 µs → 4.069 µs (−88.1%, 8.4×);
  `norm_max_64k` 39.961 µs → 5.293 µs (−86.8%, 7.5×).
- [x] [patch] (0.19.7) Consume Hermes fused multi-row AXPY
  (`hermes_simd::axpy_rows`, delivered hermes `efac045`) in Leto dense
  row-blocked matmul. The fused path updates a positive-stride output row
  block through one runtime-dispatched SIMD kernel per RHS row and keeps
  strided/transposed layouts on the existing value-correct path. Criterion
  oracle medians improved: 64x64 21.443 µs → 17.430 µs, 128x128 127.63 µs →
  108.98 µs, and 256x256 2.4357 ms → 1.0631 ms. Dense matmul remains slower
  than ndarray/nalgebra, so replacement-performance parity is still open.
  **Superseded 2026-08-28** by `LETO-MATMUL-PARITY-VERDICT-2026-08-28`: the
  gap closed and the parity thread is no longer open.
- [x] [patch] Consume Hermes batched row-panel AXPY
  (`hermes_simd::axpy_rows_batch`, delivered hermes `d4a01bd`) for the
  measured 128-row dense matmul regime. The path keeps caller-owned output,
  borrows contiguous RHS panels directly, and packs only the fixed-size alpha
  panel on the stack. Local themis-0.9 stack criterion medians improved
  `oracle_compare/matmul_leto_128x128` from 212.64 µs to 98.853 µs. Dense
  matmul still trails nalgebra's recorded 128x128 median, so parity remains
  open. Broad depth-batched routing across 64x64/256x256 was rejected after
  regression.
### Stage C3 — cache-aware CPU kernels (atlas ADR 0002 leto slice)
Criterion baselines recorded in `gap_audit.md` (2026-06-11); every
item below must show a statistically significant improvement against them —
no unmeasured "optimization" per performance_engineering.
- [x] [patch] Row-walk strided traversal (`RowMajorTraversal` in
  `application/index.rs`, shared by binary/unary serial + parallel strided
  paths): one offset computation per innermost row, stride-increment walk.
  Measured: transposed add 1.206 ms → 49–51 µs (−95.9%, 23.7×, p < 0.05),
  contiguous unchanged; negative-stride differential tests added. Remaining
  gap vs contiguous is ~3.6× (cache-line behavior of column walks).
- [x] [patch] Row-walk whole-array strided reduction and norm traversal:
  `sum` and generic `norm` now share the same innermost-row base-offset policy.
  Criterion baselines added for transposed and reverse-last-axis reductions:
  transposed `sum` 40.73 µs, transposed `norm_l2` 28.67 µs,
  reverse-last-axis `sum` 30.55 µs, reverse-last-axis `norm_l2` 30.21 µs.
- [x] [patch] (0.13.1) Row-walk policy completed across zip (all four
  variants, indexed forms via incremental last coordinate), `map_inplace`,
  and scan lane walks — every strided fallback now routes through
  `RowMajorTraversal`. Measured: transposed zip 553.4 µs → 55.9 µs (−89.9%,
  9.9×, p < 0.05). Axis-reduction output-index decomposition is amortized
  over the axis length (cost 1/axis_len per element) and deliberately left.
- [x] [patch] (0.14.4) Cache-line micro-tiling for column-walk strided
  elementwise (binary serial + parallel): tile side = 64-byte line /
  `size_of::<T>()` (analytic, not tuned); applied only when some operand's
  |last-axis stride| ≥ elements-per-line. Measured: transposed add
  50.65 µs → 28.4 µs (−43.5%, p < 0.05), contiguous unchanged; gap vs
  contiguous now ~1.8× (cumulative 42× from the 1.206 ms origin). Line
  tiling needs only the line size, so the themis `CacheLevel` wiring item
  now applies to the L1/L2-sized matmul blocking below. Residual ~1.8× is
  TLB/prefetch behavior of large-stride walks; revisit only with profile
  evidence.
- [x] [minor] (0.15.0) Extend line micro-tiling to the unary strided
  fallbacks (`map_into` serial + parallel) through the same `TileGeometry`
  SSOT. Mixed input/output scalar maps use the smaller
  `line_elements::<T/U>()` value. Measured: transposed unary
  `map_into` 57.631 µs (56.477–58.379 µs CI) → 35.303 µs
  (34.221–36.468 µs CI), −38.7% median with non-overlapping confidence
  intervals. Contiguous `map_into` remains within observed run-to-run noise;
  no contiguous speedup is claimed.
- [x] [patch] (0.18.1) Row-block dense matmul on top of the Hermes AXPY row
  kernel: one authoritative const-generic row-block kernel reuses each RHS row
  across 32 output rows, writes caller-owned output in place, and allocates no
  temporaries. Criterion all-features current medians:
  `dense_64x64` 22.536 µs (~−19.8% vs recorded 28.1 µs table baseline);
  `dense_256x256` 1.4016 ms (~−8.3% vs recorded 1.529 ms table baseline).
- [x] [minor] Close topology-adaptive matmul tile sizing from `CacheGeometry`.
  **Implementation delivered:** `MatmulTilePolicy` uses one quarter of detected
  L2, caps at the existing 32-row specialization, rounds to a safe power-of-two
  const-generic shape, and preserves the measured common-shape 32-row route.
  **Route-coverage evidence (2026-08-08):** dense C×C inputs now use the same
  policy-aware row-block/tiled-GEMM route as the generic layout path; the legacy
  `serial_cc_matmul`/`parallel_cc_matmul` bypass was removed. A 64×64 explicit
  fixed-1 versus fixed-32 differential test is value-equivalent, and the dense
  64×64/256×256 benchmarks execute through the production route (`6.0371 µs`
  and `116.35 µs` medians in the recorded run). The alternating-order,
  checksum-consuming 64×64×4096 strided policy comparison still selected 16
  rows automatically versus fixed 32 with overlapping intervals, so it provides
  no adaptive speedup or regression claim. Production convenience APIs retain
  fixed 32; the explicit adaptive seam remains available for hardware-specific
  experiments.
- [x] [minor] `LETO-MATMUL-PERF-1`: Close dense matmul oracle performance
  parity before any replacement claim. **Owner:** Codex `/root` (complete).
  **Claimed files:** `crates/leto-ops/src/application/matrix.rs`,
  `crates/leto-ops/benches/kernels.rs`,
  `gap_audit.md`, and `checklist.md`.
  The 0.19.7 fused multi-row AXPY improves Leto but still trails
  ndarray/nalgebra at 64x64, 128x128, and 256x256. Current medians:
  Leto 17.430 µs / 108.98 µs / 1.0631 ms; ndarray 8.4923 µs / 66.527 µs /
  495.95 µs; nalgebra 8.7752 µs / 62.935 µs / 505.35 µs. Investigate
  row/block/column micro-kernel geometry, cache-topology-selected tile shapes,
  and allocation-controlled reusable packing scratch. Do not retry the
  rejected 0.14.3 const-generic blocking, generic `mul_add` hook, 0.19.2
  zero-skip branch removal, 0.19.3 packed RHS dot path, 0.19.3 scalar
  row-update path, 0.19.4 Hermes `tiled_gemm` path, reduced small-matrix
  parallel scheduling, 0.19.5 `MATMUL_ROW_BLOCK=16`, or 0.19.5 first-shared-row
  output initialization, post-0.19.7 Hermes column-chunk `axpy_rows`, or
  post-0.19.7 `MATMUL_ROW_BLOCK=64`, or post-0.19.7 row-block
  fused-branch/alpha-buffer hoisting, or post-0.19.7 generic 4x4 registered
  dense tiles, or broad depth-batched row-panel AXPY routing without a changed
  kernel model and profile evidence. **Closed 2026-07-23 as an evidence-only
  audit:** current default-feature medians are Leto 23.597/123.63/233.60 µs
  versus ndarray 12.770/113.07/952.54 µs at 64/128/256; serial Leto is
  27.483/223.69/1.8522 ms. The current parallel threshold remains the measured
  better policy. Flamegraph collection is blocked by missing Windows dtrace
  and administrator-only blondie; no speculative production rewrite landed.
- [x] [patch] `LETO-STRIDED-REDUCE-1`: Reduce the overhead of the genuinely
  non-unit-stride whole-array reduction fallback without copying the view.
  **Owner:** Codex `/root` (complete). **Claimed files:**
  `crates/leto-ops/src/application/reduction.rs`,
  `crates/leto-ops/tests/ops/reduction.rs`,
  `crates/leto-ops/benches/kernels.rs`,
  `gap_audit.md`, and `checklist.md`. Acceptance: one canonical generic
  fallback preserves logical values and the existing reduction contract for
  positive and negative strides, performs no allocation or materialization,
  and shows a value-preserving benchmark improvement for the existing
  `sum_strided_step2_256x256` case; if the measured result is not positive,
  close with evidence and retain the current implementation. Do not add a
  second scalar-type or operation-specific kernel. **Closed 2026-07-23 as
  evidence-only:** the order-preserving four-way loop candidate measured
  `27.793 µs` versus the quiet baseline `28.849 µs` with `p = 0.06`, while the
  contiguous control regressed in the candidate run; the production helper
  was removed. The zero-copy fallback remains unchanged and the new focused
  regression test passes.
- [x] [minor] (0.19.0) Route reverse-last-axis whole-array reductions through
  borrowed unit-stride physical row slices. `sum` uses `Scalar::sum_slice`;
  `norm` uses `NormKind::accumulate_slice` plus the new defaulted
  `NormKind::combine` hook so row partials combine in accumulator space.
  Criterion: `sum_reverse_last_axis_256x256` 5.1575-5.2534 µs (−21.56%
  median, p < 0.05) and `norm_l2_reverse_last_axis_256x256` 9.1467-9.9752 µs
  (−18.00% median, p < 0.05).
- [x] [minor] (0.18.0) Wire themis as an optional leto-ops dependency for
  `CacheLevel` queries through `leto_ops::CacheGeometry` and the `topology`
  feature. The public API is additive; when the feature is disabled, callers
  get the documented fallback L1/L2/line constants. The `themis` cache-level
  reader walks the borrowed slice directly and does not allocate copies.

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
