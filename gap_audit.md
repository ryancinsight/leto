# Leto Gap Audit: ndarray / nalgebra Replacement for Atlas

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
