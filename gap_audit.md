# Leto Gap Audit: ndarray / nalgebra Replacement for Atlas

## Unit directions and checked input invariant [patch] — core0.46.2 under review

The smallest positive basis has exact direction[1,0,0] but reciprocal length
is infinite; equal maximum finite components have finite direction[1/sqrt(3);3]
but scalar length overflows. Preserve the ordinary native path and normalize
exceptional finite vectors through v/max(abs(v)); the common scale cancels.
Reuse Vector::norm rather than widening or cloning algorithms by scalar/dimension.
Direct zero/nonfinite IEEE results retain their established classifications.

Unit::try_new is documented as the zero-rejecting boundary. Keep its existing
norm>min_norm IEEE comparison, including valid vectors with negative thresholds;
NaN/+infinity thresholds reject. Reject zero/empty/nonfinite vectors regardless
of threshold, preserving the checked unit invariant. A finite norm proves finite
components; infinite norm requires distinguishing finite overflow from infinite
input. Share the already computed norm with canonical vector normalization.
No new public bound, threshold-validation domain, source pin or range-error API.

Preserve the original audit checkout,20 failures/four controls and all prior
#371/owner evidence. Original24 tests plus the new contract test pass.39124 vectors
per debug/opt2/opt3 mode span all binary exponents, random IEEE components,
dimensions0/1/2/3/4/8/17 and native/wrapped f32/f64.37778 ordinary outputs/mode keep
baseline bits; wrapper parity holds. Decimal100-digit reference from exact input
components observes≤2epsilon component error and≤4epsilon squared-unit residual.
Bounds for tested N allow(N+8)epsilon component and(2N+16)epsilon squared-unit
rounding error; no universal correctly-rounded promise. Unsupported F16/Bf16 are
not RealField implementations; do not weaken that boundary to expand a test.

Strict minimal/lint/docs,1247 native tests/zero skips and25 isolated focused Miri
tests per borrow model pass. Seven alternating pairs across opt2/opt3 direct/opaque/indirect/stored callers,
four scalars and dimensions3/17 retain every raw timing/assembly sample. The
selected generic helper inline hint improves some initial cases; borrowed,
eager and in-place alternatives remain preserved. No always-inline/CPU/threshold
or lint override. Direct opt3 mixed3D f32 normalize2.42→2.57ns, f64 normalize
2.08→2.78ns and checked unit2.45→2.85ns; stored3D f64 normalize2.60→2.69ns.
Material costs remain: maximum ordinary+12.05ns/+91.31% for17D stored checked
unit13.19→25.24ns at opt2. Corrected cold paths4.41-54.03ns; baseline exceptional
outputs are incorrect.1024 warmed memory cases/cohort request zero allocations,
peak or retained bytes. These are local ARM64 primitive/caller observations,
not universal throughput or formal heap/stack/cache bounds. Independent
performance acceptance remains a hold; do not claim all costs are preserved.
Final-head package/source/VCS, API, normal hooks, hosted checks and independent
review remain pending. Only core advances0461→0462; provider records, ops/Python,
limits, workflows, settings and owner branches retain ancestor bytes.

This separate draft is stacked on frozen #371 and cannot release through the
unresolved Moirai cohort. Verified existing local task Re-derive sparse kernel
integration owns #362, Apollo #674 and Coeus #533. It is separate from the active
geometry/review tasks. Coordinator action: assign that owner the single reserved
0.47 integration and coordinated Moirai0.10 bindings, or explicitly transfer that
ownership. No message or owner edit is performed here. Preserve Coeus584/585,
Apollo LaneBuffer/service and other release holds. Rebase/reclassify and renew
exact-head validation after coherent integration; do not duplicate reserved0.47.

## Quaternion inverse intermediate range [patch] — core 0.46.1 held for review

For finite basis q=a, q inverse is 1/a; imaginary bases negate the component.
f32 powers 2^±80 and f64 powers 2^±600 have exact finite inverses and identity
products, but baseline squared norm over/underflows first. Preserve the native
normal-norm path; recover other finite inputs with binary exponent scaling,
scaled squared norm in [1,16), native product/sum/quotient residual compensation.
Preserve legacy zero/nonfinite inputs and nonfinite corrected-output fallback.
No widening, checked range policy, new tolerance or public bound is selected.

Independent review P2 found that joining whole Option values introduces costly
aggregate stores/reloads in ordinary callers. Select the minimal payload-first
join: ordinary conjugate/n2 or recover_inverse(self,n2)?, then Some(inverse).
The predicate, arithmetic and private cold helper stay identical. This removes
that aggregate join without precision-specific branches or forced inlining.
Preserve the original checkpoint, unsuccessful alternatives and reviewer patch.

Expanded comparison includes 34936 author and 54544 independent exact cases in
each debug/opt2/opt3 mode (89480/mode). All returned norm/inverse bits and native/
wrapped F32/F64 parity match the reviewed head. Finite-input Hamilton product
bits also match. At opt2/opt3, 128 nonfinite-input downstream product NaN payload/
sign rows differ; debug has zero. These are not changed inverse return bits or
finite product errors. The untouched exact Fraction oracle passes; observed
fallback error is one ulp, ordinary four and product residual four epsilon.
These samples do not promise universal correct rounding. Twenty regressions
pass in all modes versus baseline twelve failures/eight controls. Unrepresentable
inverse outputs retain their separate legacy range policy.

Seven alternating baseline/reviewed-head/repair triples cover opt2/opt3 direct,
opaque, indirect, stored-input streams and unwrapped-output stores, using Rust
1.97 and repository thin-LTO/codegen1. No CPU, threshold, inline or lint override.
Direct mixed f64 opt3 medians: baseline1.24ns, old11.97ns, repair1.22ns. f64 stream
0.94/1.89/0.94ns and unwrapped store0.96/2.04/0.95ns respectively. Tradeoffs remain:
maximum ordinary cost versus old+0.53ns (versus baseline+0.53ns/51.06%); opaque
f32 mixed1.03→1.54ns; indirect f64 cold small quartet27.57→31.46ns. Cold repair
range6.10-31.48ns. All sampled paths request zero heap after setup. Raw samples,
assembly and source/flags are preserved. This is warmed local ARM64 evidence,
not universal latency, whole-mesh throughput or formal heap/stack/cache bounds.
Independent acceptance of these caller tradeoffs remains a release hold.

Unchanged minimal, strict Clippy, 1222 native tests/zero skips, 33 doctests/one
ignore and strict docs pass. Final-head locked API, published-core patch API,
package source/VCS identity, dry-run, fresh typed consumers, normal hosted gates
and exact-head independent review remain required. Existing hosted locked-source
API acceptance is not fresh workspace/cohort or publication acceptance.

Only core leto0.46.1 is selected; ops0.46.0/Python0.46.1 retain source/versions.
No workflow, credentials, limit, service deployment or protected hold changes.
Both independent performance acceptance and the separate workspace coherence
prerequisite must be resolved before merge/publication; a sole-core plan alone
is insufficient. Reassess version/scope if coordinated provider work lands.

## Current Moirai public cohort prerequisite [minor] — owner coordination

Unchanged main requires Moirai^0.9.3 at canonical Git URL; current default is
0.10.0 (68466225913301d30281b134d42b17c604baf78e). Fresh baseline and reviewed-head
workspace/raw ops API resolution fails identically. Current core-only consumers
resolve because they exclude Moirai. Public for_each_task_partition_mut_with and
for_each_task_partition_mut expose Moirai runtime, policy and ExecutorResult;
changing their provider's nominal identity is a pre-1.0 breaking cohort boundary.

Active #362 (24a162a1703f14e0a2adc7d647a082d36af09394) owns borrowed CSR/caller
dispatch and reserves workspace0.47.0, including root requirements and package
lock versions. #360 (0762c219cbac099f4a158e7e33b5e964a5818b63) also touches shared
manifests. Prepare a prerequisite, not a competing draft or silent #371 upgrade:
agree owner and landing/version sequence; preserve canonical URL/features, admit
0.10 and refresh coherent default providers without older pins; document nominal
type migration and reconcile Python's already explicit0.46.1 with #362's older
base. Do not invent another version or automatically republish Python.

A separate copied-main diagnostic changes only the Moirai requirement and drops
its copied lock for fresh resolution. It is not a versioned release candidate.
Fresh all-feature workspace metadata and source compilation pass with one
current Moirai0.10 identity; no Rust edit is needed for that source check. This
does not establish public typed-consumer/runtime behavior, API or release
acceptance; preserve the fresh lock/provider evidence and remaining plan.
Before delivery require fresh whole-workspace checks and positive current-cohort/
negative old-cohort typed callers; API classification/migration; unchanged full
native/minimal/Miri/wheel/package gates; dependency identity and provider-first
release graph; exact-head independent review. Owner coordination is unresolved,
so production requirements and locks remain untouched in #371.

## 2026-10-16 Native/extension linkage and macOS wheel loader [patch] — closed

- findings: unconditional PyO3 extension linkage breaks native CPython 3.13 linking; implicit Rust/LLVM stripping misaligns the optimized macOS wheel; unprivileged PR verification previously attempted privileged attestation.
- delivery: #365 merges as `747d0bbb` with its reviewed tree unchanged; Atlas #542's immutable recipe guard is selected. [Python 0.46.1](https://pypi.org/project/leto-python/0.46.1/) publishes from that exact source in run `38835589358`; Rust packages stay 0.46.0 and all 223 external lock records are unchanged.
- evidence: default native 1,202 passed, zero skips; release 30 rows/270 cases, zero skips; all 31 registry files match release bytes and verified provenance. Actual registry wheel passes all 38 published-source tests; all ten Mac slices align; 534 critical source payloads equal rebuilt source. Automatic Pages run `38834105768` builds the same commit and serves HTTP 200.
- residual: the unchanged graded-scan correctness case remains slow at 38.885s under the existing 30/60s policy; no performance work or limit change occurred. The additional non-required managed scan cannot start its configured model and is explicitly accepted as nonblocking; no security setting changes.
- re-open: a supported native linker, wheel loader, value/error contract, release identity or source/registry hash check fails. Existing unrelated numerical findings below remain open.

## 2026-10-15 Provider-default resolution [minor]

- risk: published 0.45.2 belongs to the older provider line. Main #364 now admits the new Git cohort; 0.46 must expose that published boundary without changing Rust sources or numerical assertions.
- boundary: Mnemosyne 0.10, Hermes 0.9 and Moirai 0.9.3 are integrated/published. The refreshed lock follows those defaults; fresh Git-consumer and exact-head gates remain distinct from publishing Leto 0.46. Main integration also triggers legacy Pages and the existing pending publisher.
- acceptance: exact-source API/native/hosted checks, one provider identity, provider-first integration and a clean consumer after the final refresh. No path override, manifest revision pin or conversion adapter masks an old edge.

## 2026-09-27 Francis skew ties beyond one ulp (Bf16)

- risk: a Bf16 skew tridiagonal whose iterate couples two `[[0, a], [−a, 0]]` blocks by `ulp·2a < δ ≲ 2·ulp·2a` cycles to the iteration cap and returns `ConvergenceError` (3 of 90,000 scan matrices, orders 3–8); `dhseqr`'s recovery for a failed `dlahqr` is `dlaqr0` (AED), which leto lacks.
- evidence: `tests/ops/schur.rs::unresolvable_bf16_skew_tie_reports_its_residual`; the Wilkinson shift sits between the clusters `±i(a ± δ/2)` and Bf16 rounding restores the tie after each exceptional shift.
- re-open trigger: a consumer needs Bf16 `schur`/`eigenvalues` on skew-symmetric input, or an AED (`dlaqr3`) port is planned.

## 2026-09-01 Apollo complex matrix-batch transpose

- risk: none (closed) — see ADR 0027.
- evidence: ADR 0027 records the provider median reductions (86.7-89.8% at
  1,024x4x4; 26.1-53.3% at 256x16x16) and the zero-allocation warmed census.
- re-open trigger: n/a.

## 2026-08-26 Apollo FFT layout-copy baseline

- risk: `assign`'s checked N-D indexing measured 15.3-29.0x slower than
  Apollo's tiled gather/scatter loop; a tiled/bounds-elided candidate closed
  most of the gap (4/8 shape/direction rows now statistically
  indistinguishable, 4/8 still favor the candidate by 4.7-11.9%).
- evidence: full before/after tables in
  `docs/benchmarks.md#layout-copy--assign-kernel-apollo-fft-gatherscatter-2026-08-26`.
- re-open trigger: closes with backlog `LETO-FFT-LAYOUT-THROUGHPUT`
  (full FFT round-trip and steady-state allocation verification, owned by
  Apollo integration).

## 2026-08-13 Convolution provider closure

- risk: none (closed) — ADR 0019; provider PRs #78-80 merged at `e525d8d`,
  Coeus consumer cutover at `aabdec6`.
- evidence: hosted runs `31663241086` (provider) and `31672329963`
  (consumer) passed full gates at those exact heads; re-verified 2026-09-27
  that `RUSTDOCFLAGS=-D warnings cargo doc -p leto -p leto-ops` is currently
  clean (the 33 broken-link warnings this entry recorded no longer exist).
- re-open trigger: n/a.

## 2026-08-08 Fallible plain mutable iteration

- risk: none (closed) — `ElementIterMut`, `Array::try_iter_mut`, and
  `ArrayViewMut::try_iter_mut` are landed and current
  (`crates/leto/src/application/iter/element/`).
- evidence: layout-rejecting preflight before any `&mut T` escapes; focused
  Nextest coverage at merge time.
- re-open trigger: n/a.

## 2026-08-08 Borrowed iterator ergonomics

- risk: none (closed) — `&Array`/`&ArrayViewMut` `IntoIterator` via
  `ElementIter` is landed; mutable iteration stays the fallible
  `indexed_iter_mut` contract by design (no infallible mutable
  `IntoIterator`).
- evidence: core iteration suite covered owned/transposed/mutable-view/
  indexed-mutable/double-ended/empty/alias-rejection cases, 14/14 at merge.
- re-open trigger: n/a.

## 2026-08-04 Provider-owned CPU cross-entropy gap

- risk: none (closed) — ADR 0023; PR #94 merged as `c743a60`. Confirmed
  current: `crates/leto-ops/src/application/loss/` exists.
- evidence: exact-head Rust verification passed at merge.
- re-open trigger: n/a. Accelerator execution and consumer dispatch are
  Hephaestus/Coeus, not Leto.

## 2026-07-29 Coeus convolution provider gap

- risk: none (closed) — ADR 0019; regular/transposed N-D forward/backward
  convolution landed generic over `T`, borrowed inputs, caller-owned
  outputs.
- evidence: 21 exact value-semantic/failure-atomicity tests across
  f32/f64/F16/Bf16 at merge; 196/196 applicable minor-release SemVer checks.
- re-open trigger: n/a. Coeus's own CPU-dispatch cutover is tracked in the
  Coeus backlog, not here.

## 2026-08-08 Topology-adaptive matmul policy

- risk: none (closed) — `MatmulTilePolicy` (`crates/leto-ops/src/application/matrix/`)
  is landed and threaded through serial/parallel row-block paths; the legacy
  unpolicy-aware fast path was removed.
- evidence: adaptive-vs-fixed-32 ranking stayed inconclusive (overlapping
  95% CIs) — full numbers in
  `docs/benchmarks.md#topology-policy-evaluation-2026-08-08`. Fixed 32 rows
  remains production; the adaptive selector remains available.
- re-open trigger: fresh hardware-specific evidence for adaptive ranking.

## 2026-07-23 Contiguous and non-unit-stride benchmark coverage

- risk: none (closed) — coverage gap only, no kernel defect; new
  `crates/leto-ops/benches/kernels.rs` rows for elementwise/sum/matmul
  C-dense vs step-2 views landed.
- evidence: numbers in
  `docs/benchmarks.md#contiguous-and-non-unit-stride-coverage-2026-07-23`.
- re-open trigger: n/a.

## 2026-07-23 Non-unit-stride reduction audit

- risk: none (closed) — an order-preserving four-way generic loop candidate
  was measured (not a significant improvement, p=0.06) and removed rather
  than retained as speculative optimization; existing zero-copy row-walk
  retained.
- evidence: numbers in
  `docs/benchmarks.md#non-unit-stride-reduction-audit-2026-07-23`.
- re-open trigger: a working profiler or an independent measured kernel
  model, before any further strided-reduction change.

## 2026-07-22 Sparse LU native-view boundary

- risk: none (closed) — `SparseLuSolver::solve_view` over `ArrayView1`
  landed (`crates/leto-ops/src/application/sparse/lu_sparse/solver.rs`);
  legacy `&[T]`/`Vec<T>` method routes through it; CFDrs migrated off the
  copy-in/copy-out boundary.
- evidence: provider/consumer value-semantic regression tests at merge.
- re-open trigger: n/a.

## 2026-07-20 Decomposition SIMD-dispatch gap

- risk: none (closed for the shipped kernels) — Cholesky/SVD/udu inner
  reductions converted to `dot_slice`; QR panel reflector axpy investigated
  and correctly kept scalar (short/shrinking slices regress under SIMD).
- evidence: per-kernel deltas, the long-vs-short-axpy meta-pattern, and the
  still-open full_piv_lu/bunch_kaufman/col_piv_qr follow-ups in
  `docs/benchmarks.md#decomposition-kernel-simd-dispatch-2026-07-20`.
- re-open trigger: n/a here — open follow-ups filed as backlog
  `LETO-DECOMP-AXPY-FOLLOWUPS-2026-09-27`. The hermes CSR-SpMV lead noted in
  this entry is hermes's own concern, not filed on this board.

## 2026-07-20 SpMV bounds-check elision (Krylov kernel)

- risk: none (closed for the shipped elision) — CSR/CSC per-nonzero bounds
  checks collapsed to per-row via `row_ptr.windows(2)`; bitwise-identical
  output (pure refactor).
- evidence: −14%/−19%/−27% (CSR, n=4096/65536/1<<20) and −24%/−16% (CSC,
  n=4096/65536); full numbers and the rejected `get_unchecked` gather
  candidate in
  `docs/benchmarks.md#spmv-bounds-check-elision-krylov-kernel-2026-07-20`.
- re-open trigger: n/a here — the blocked `usize`->`u32` CSR index-width
  lever is filed as backlog `LETO-CSR-INDEX-WIDTH-2026-09-27`.

## 2026-07-20 Blocked LU cache-resident regression

- risk: none (closed — reverted, never shipped a regression) — a
  right-looking blocked (BLAS-3) LU regressed at n=256/512 on this host's
  36 MiB L3 (unblocked SIMD axpy already runs at cache bandwidth below
  n≈1200).
- evidence: numbers and the required re-entry conditions in
  `docs/benchmarks.md#blocked-lu-cache-resident-regression-2026-07-20`.
- re-open trigger: filed as backlog `LETO-BLOCKED-LU-L3-GATE-2026-09-27`
  (cache-aware gate before any retry).

## 2026-07-18 Eunomia 0.4 provider refresh

- risk: none (closed, superseded) — eunomia is now 0.8.0
  (`Cargo.toml`), well past the 0.4.0 this entry recorded.
- evidence: current manifest.
- re-open trigger: n/a.

## 2026-07-18 Eunomia complex oracle ownership

- risk: none (closed) — `leto-ops` test oracles bind to
  `eunomia::{Complex, Complex32, Complex64}`; no direct `num-complex`
  manifest dependency exists today (re-verified 2026-09-27).
- evidence: current manifest search.
- re-open trigger: n/a. The residual "external nalgebra/ndarray test
  oracle" concern this entry raised (`LETO-EXTERNAL-ORACLE-1`, never filed)
  is superseded by ADR 0017's deliberate decision to keep `ndarray`/
  `nalgebra` as dev-dependency oracles — not an open gap.

## 2026-07-17 CFDrs sparse direct factorization gap

- risk: filed as backlog `LETO-SPARSE-DIRECT-1` — Leto owns CSR but exposes
  no sparse direct factorization; CFDrs retains a third-party solver for
  its post-GMRES-stagnation tier.
- evidence: see the backlog item.
- re-open trigger: closes when `LETO-SPARSE-DIRECT-1` merges.

## 2026-07-15 Provider default-branch convergence

- risk: none (closed) — manifest follows Mnemosyne/Moirai/Hermes/Eunomia/
  Themis default branches; the locked provider-duplicate scan was empty at
  merge.
- evidence: focused fmt/Clippy/nextest/rustdoc gates passed at merge.
- re-open trigger: n/a. Downstream Hephaestus/Apollo lock convergence is
  their own board's concern.

## Layer boundary decision

- risk: none — promoted to ADR 0034 (Accepted, retroactive/as-built).
- evidence: `docs/adr/0034-layer-boundary-decision.md`.
- re-open trigger: n/a; revise ADR 0034 in place if the boundary changes.

## B. Gaps vs nalgebra (linear algebra)

- risk: none — this table is superseded by
  `docs/completeness/parity_matrix.md` and `docs/completeness/PLAN.md`,
  which explicitly overrides this entry's consumer-driven policy and
  re-opens several routines this entry recorded as permanently excluded.
- evidence: `docs/completeness/PLAN.md` §1 states the supersession.
- re-open trigger: n/a; consult the parity matrix for current linalg gap
  status, not this file.

## D. Residual Risk Register

- risk: none (closed) — this section recorded five point-in-time audits
  (2026-06-15 v0.24.0 status, 2026-06-23 matmul offset-routing/Tree-Borrows
  soundness sweep, 2026-07-02 scalar SSOT audit, 2026-07-04 layout serde
  rank gap, 2026-07-05 CR-4 scalar SSOT rebind), all resolved at the time
  and re-verified current 2026-09-27: `Scalar: NumericElement` /
  `RealScalar: Scalar + FloatElement + ...` still hold
  (`crates/leto-ops/src/domain/{scalar/contract,real}.rs`), and no stray
  `target_ag`/duplicate target tree exists under `D:/atlas`.
- evidence: full narrative in git history at this file's pre-2026-09-27
  revision; each audit's PR/commit is cited inline there (`git log -p --
  gap_audit.md` on a commit predating this compaction).
- re-open trigger: n/a. The 2026-06-23 matmul parallel-matmul Tree-Borrows
  soundness argument is standing evidence, not an open risk — re-derive
  only if `batched_matmul`'s disjointness guard changes.

## Leto rank-deficient singular-values parity

- risk: none (closed) — `singular_values` split from `svd_decompose`;
  diagonalizes the Gram matrix and maps near-zero eigenvalues to zero for
  finite rank-deficient inputs.
- evidence: value-semantic tall/wide rank-deficient tests at merge; no
  machine-checked proof performed.
- re-open trigger: n/a.

## Leto wide thin SVD parity

- risk: none (closed) — `svd_decompose`/`singular_values` generalized to
  all full-rank thin SVD shapes (wide via `A·Aᵀ`, `V = Aᵀ·U·Σ⁻¹`).
- evidence: value-semantic reconstruction/orthonormality tests at merge; no
  machine-checked proof performed.
- re-open trigger: rank-deficient wide inputs remain explicit errors until
  a rank-revealing SVD contract covers them (tracked in the parity matrix,
  not here).

## Leto 64² singular-values disparity

- risk: none as an open item — fully superseded by ADR 0012, which contains
  this entry's phase attribution, the dqds prototype-and-revert, and the
  later batched-reflector win (bidiag 1.72x -> ~1.40x nalgebra) in far more
  detail than this entry ever held.
- evidence: `docs/adr/0012-dqds-values-only-singular-values.md`.
- re-open trigger: n/a here; ADR 0012 states its own re-open gates for a
  scoped dqds/`dlasq4` [major] item.

## 2026-08-13 leto-ops criterion baselines

- risk: none (reference data) — relocated to `docs/benchmarks.md` 2026-09-27.
- evidence: criterion medians + 95% CI, harness `crates/leto-ops/benches/kernels.rs`.
- re-open trigger: n/a; see `docs/benchmarks.md` for the live gate baselines
  and the do-not-retry rejected-optimization list.

## 2026-09-26 Board narrative drift (backlog.md / gap_audit.md)

- risk: none (closed 2026-09-27) — all ~24 legacy entries this drift
  produced are now compacted to the risk/evidence/re-open-trigger/owner
  schema (`LETO-GAPAUDIT-ENTRY-COMPACTION`); `gap_audit.md` line count
  dropped from 664 to under 250.
- evidence: this file's git history for the pre-compaction text; per-entry
  numerical evidence relocated to `docs/benchmarks.md` and ADR 0012/0027/
  0034 rather than dropped.
- re-open trigger: `wc -l gap_audit.md` past 900, or entries again
  exceeding ~5 lines without a recurring compaction pass.

## 2026-09-26 Stack-owned source-identity pre-push check fails on lockfile pushes

- risk: `atlas-build-identity.py` (stack-owned) runs `cargo --locked` from
  `repo_root`, whose ancestor config discovery applies the `[patch]`
  overlay, which wants to rewrite the lock `--locked` forbids — fails on
  every push touching `Cargo.lock` from a member's main tree.
- evidence: reproduced twice (PR #255); confirmed fixed for lane pushes
  (`pre-push: lane of an overlaid member; gating outside the stack
  overlay`, verified on PR #256).
- re-open trigger: closes when atlas#314 lands and a main-tree
  `Cargo.lock` push verifies without `SKIP_LOCAL_GATE`. Owner: atlas (meta).
