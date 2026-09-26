# Leto Work Backlog

<a id="LETO-BIDIAGONAL-CHASE-DIRECTION-2026-09-25"></a>

## LETO-BIDIAGONAL-CHASE-DIRECTION-2026-09-25 — Bidiagonal QR lacks `dbdsqr`'s chase-direction choice [patch] — todo
- priority: correctness
- needs: none
- scope: `crates/leto-ops/src/application/linalg/svd/bidiagonal_qr.rs`, `crates/leto-ops/src/application/linalg/svd/zero_shift.rs`, `crates/leto-ops/tests/ops/svd/`
- Evidence: the sweeps (shifted and zero-shift) always chase top to bottom with the forward convergence tests; `dbdsqr` chases bottom to top when `|d_q| > |d_p|`, with the matching backward tests. On bottom-heavy f32 bidiagonals (smallest singular value `≈ 6·10⁻¹⁵` of the largest) the smallest is returned as `0` in 28 of 600 (PR #237 review of 6e397c8; base failed all 28); the result is within the normwise bound only.
- Outcome: implement `dbdsqr`'s direction choice (`IDIR`), both sweep directions and the backward convergence tests, so tiny singular values of graded bidiagonals keep high relative accuracy.
- Acceptance: on those 600 bidiagonals every singular value is within `c·k·ε` relative of the `f64` reference, `c` derived from Demmel & Kahan's relative error bound for the zero-shift sweep.
- Next step: port `dbdsqr`'s `IDIR` selection and loops 130–160 (bottom-to-top chase) against the LAPACK source.

<a id="LETO-BF16-SKEW-FRANCIS-STALL-2026-09-25"></a>

## LETO-BF16-SKEW-FRANCIS-STALL-2026-09-25 — Francis stalls on one graded Bf16 skew-symmetric tridiagonal [patch] — todo
- priority: correctness
- needs: none
- scope: `crates/leto-ops/src/application/linalg/schur/francis.rs`, `crates/leto-ops/tests/ops/schur.rs`
- Evidence: a seeded sweep of 46,440 Bf16 skew-symmetric tridiagonals (orders 3–8, magnitudes log-uniform over `[10⁻⁴, 1]`, every exponent) found one (and a finer sweep, every exponent, 2 of 30,960) where `schur`/`eigenvalues` exhaust `MAX_ITER`: order 8, superdiagonal `(2.05e-21, −1.50e-22, 3.18e-23, −5.56e-24, 5.48e-24, 2.40e-24, −3.18e-23)`, failing at every scale the gate maps to one landing (present at 3dbea45 as well). A subdiagonal `≈ 1.6·10⁻⁶·‖H‖` sits at the rounding-noise level of the zero diagonal (`tst ≈ 10⁻⁵·‖H‖`), so neither the `ulp·tst` pre-check nor Ahues–Tisseur deflates it and the bulge dies there.
- Outcome: the order-8 case converges, and the scan's skew families add Bf16 orders with that dynamic range.
- Acceptance: the case returns `Ok` with the a-posteriori certificate holding; no regression in the skew probes.
- Next step: trace the iterate against `dlahqr`'s arithmetic emulated in Bf16 before choosing a deflation change.

<a id="LETO-LOW-PRECISION-FACTOR-ORACLE-2026-09-25"></a>

## LETO-LOW-PRECISION-FACTOR-ORACLE-2026-09-25 — No derived oracle bounds F16/Bf16 factor accuracy [patch] — todo
- priority: verification
- needs: none
- scope: `crates/leto-ops/tests/ops/backward_error.rs`, `crates/leto-ops/tests/ops/a_posteriori.rs`, `crates/leto-ops/tests/ops/graded_scan.rs`, `crates/leto-ops/tests/ops/scale_range.rs`
- Evidence: PR #237's deterministic γ bounds are vacuous for the iterative routines in F16/Bf16 (one length-3 reflector costs γ₅₃ ≈ 0.027/0.26); the a-posteriori certificates bind values to factors but assert nothing about the factors' residual or orthogonality in those formats (ADR 0033, Tests).
- Outcome: assert F16/Bf16 factor accuracy against a derived probabilistic bound: Higham & Mary, "A new approach to probabilistic rounding error analysis", SIAM J. Sci. Comput. 41(5), A2815–A2835 (2019) — `γ̃_n(λ) = exp(λ√n·u + n·u²/(1 − u)) − 1` with probability ≥ `1 − 2exp(−λ²(1 − u)²/2)` per product (Theorem 2.4, eqs. 2.1, 2.3). The theorem and equation numbers are verified against the preprint (MIMS EPrint 2018.33) only; the journal volume and pages are from citing literature. Its Model 2.1 (independent mean-zero rounding errors) is an assumption about round-to-nearest, to be stated with the failure probability chosen. Alternative oracle: the deterministic bound at an instrumented transformation count.
- Acceptance: tests assert F16 and Bf16 residual and orthogonality below that bound across the scan families.
- Next step: derive the per-entry rounding count of the Francis and Golub–Kahan paths so `γ̃` replaces `γ` term by term.

<a id="LETO-SYMMETRIC-EIGEN-ROUTE-2026-09-25"></a>

## LETO-SYMMETRIC-EIGEN-ROUTE-2026-09-25 — `MatrixDecompose::symmetric_eigen` still runs classical Jacobi [minor] — todo
- priority: performance

- Evidence: `symmetric_eigen_qr` landed in #233 (CHANGELOG [Unreleased]) for ritk's 60×60 MP-PCA Gram matrices, which cost 6.2 ms each under Jacobi's pivot scans, but `MatrixDecompose::symmetric_eigen` in `leto-ops/src/application/linalg/matrix.rs` still calls `symmetric_eigen_jacobi`.
- Outcome: route `MatrixDecompose::symmetric_eigen` through the QL solver, keeping Jacobi as the high-relative-accuracy option.
- Acceptance: existing `symmetric_eigen` tests pass unchanged within their tolerances; a 60×60 benchmark shows the QL cost.

<a id="LETO-MIRI-GATE-2026-09-10"></a>

## LETO-MIRI-GATE-2026-09-10 — The crate that depends on an uninitialized-write invariant had no Miri gate [patch] [safety]

- Status: in-progress; priority: safety; integrator: root; updated: 2026-09-10.
- **Why it matters.** `reduce_axis` writes its output through
  `VecStorage::uninit` and raw pointers (PR #157, `d16e3b2`). That is sound only
  because every output element is written before it is read — an obligation
  stated in prose on `VecStorage::uninit`, not enforced by any type. Nothing
  gated it, which is the defect generator the mnemosyne board already recorded
  for a crate sitting outside its Miri gate (MN-459).
- **Landed in this item:** `[profile.miri]` in `.config/nextest.toml`, and a
  `miri` job in `ci.yml` running both borrow models. Both are needed: an MN-437
  fix in this stack passed Stacked Borrows while Tree Borrows still rejected
  it, so a single-model gate can certify a still-broken fix.
- **Measured 2026-09-10 (local Windows host, Miri `0.1.0` / nightly
  `2026-08-26`).** Miri interprets every instruction, so cost tracks suite size,
  not the code under test:

  | Target | Tests | Stacked | Tree | Gated |
  |---|---:|---:|---:|---|
  | `leto --lib` | 130 | 206s | 363s | yes |
  | `leto-ops --test ops_tests -E test(/reduction/)` | 17 | 6s | 8s | yes |
  | `leto-ops --lib` | 216 | >10m | — | no |
  | `leto-ops --test ops_tests` (full binary) | 352 | >10m | — | no |
  | `leto layout_property_tests` | proptest | >10m | — | no |

- **Scope, stated so it is not mistaken for a clean bill.** The gated pair is
  the storage substrate and the uninitialized-output caller, which are the
  contracts the job exists for. The excluded surfaces are excluded because they
  are not yet affordable or Miri-clean — **not because they are believed
  sound** — and this item owns widening the scope:
  1. `layout_property_tests` samples thousands of layouts per run; under Miri
     each sample is interpreted. Its property is still checked natively.
  2. `leto-ops` unit and `ops_tests` suites exceed ten minutes. Narrowing them
     by an arbitrary name filter would tailor the gate to what passes, so they
     stay out until their workload is bounded.
- **Two failures observed locally and deliberately not filed as defects.** Both
  are environment or test-fragility, and neither is yet confirmed on the Linux
  runner:
  - `complex_transpose_allocations` aborts under Miri with a Stacked Borrows
    tag error whose entire backtrace is the **test's own** `CountingAllocator`
    forwarding to the Windows `HeapFree` path
    (`tests/complex_transpose_allocations.rs:31`), plus
    `unsupported operation: can't call foreign function
    GetNumaHighestNodeNumber`. Both are Windows-Miri limitations, not product
    UB; confirm on `ubuntu-latest` before treating either as a finding.
  - `leto-ops --lib` fails two `leapfrog` tests on an exact `assert_eq!`
    (`leapfrog/tests.rs:152`) under Miri. The test asserts bit-exact equality
    of a computed kernel against a plain difference; the divergence between
    native and interpreted float evaluation is not yet explained, so the test
    is neither loosened nor called a defect here.
- **Acceptance:** the `miri` job is green on `main` under both borrow models,
  and each excluded surface above is either brought into scope with a measured
  bound or recorded as closed with the reason it stays out.

<a id="LETO-WASM-32BIT-TOLERANCE-2026-09-10"></a>

## LETO-WASM-32BIT-TOLERANCE-2026-09-10 — Keep generic linalg thresholds portable on wasm32 [patch]

- Status: review; priority: correctness; integrator: root; branch: `codex/leto-wasm-tolerance`; updated: 2026-09-10.
- Outcome: `rank_pivot_ratio` converts the shared `1e-12` denominator through `FloatElement::from_f64`, so ColPivQR, Jacobi eigen, FullPivLU, SVD pseudoinverse, and UDU compile on 32-bit targets without changing the native threshold. The duplicate literal is deleted.
- Acceptance evidence: Atlas-overlay `cargo check --offline -p leto-ops --target wasm32-unknown-unknown`, standalone strict Clippy, 585/585 `leto-ops` nextest, and the new f32/f64 threshold contract test pass. PR pending.

<a id="leto-ctc-loss"></a>

## LETO-CTC-LOSS — Evaluate temporal label alignment loss [minor] [arch]
- Status: done; [PR 177](https://github.com/ryancinsight/leto/pull/177), merge ba8a879; native scalar loss/gradients and [ADR 0030](docs/adr/0030-temporal-label-alignment.md); full local gates pass, hosted checks pending.
- Primary integration preserves Apollo source and uses main's lock: Hermes `9d68a9e`, Eunomia `8e18d6d`, Moirai `0.6.0` at `00fb0ae`. Lock guard, format, minimal features, strict Clippy, CTC debug/release (10 each; `3b80fee3`/`7de39bb0`), 30 doctests and strict Rustdoc pass; prior Apollo performance evidence does not cover this graph.

<a id="leto-windows-source-identity"></a>

## LETO-WINDOWS-SOURCE-IDENTITY — Preserve source identity across mapped checkouts [patch]
- Status: todo; priority: correctness; updated: 2026-09-07.
- Outcome: standalone Windows builds cannot reuse another checkout's crate metadata when drive aliases are recycled; keep the single shared target directory.
- Scope: centralize the standalone Cargo mapping/freshness mechanism and adopt it in Leto's committed verification path; no mathematical or workload changes.
- Evidence: merged primary exports `transpose_copy`, but release reused local `leto-2960c0092a779492` metadata without that export; its relative-path dep-info and 05:07 UTC artifact were newer than primary source. Debug compilation passed; release failed E0432 in `application/layout/complex/batch.rs:8`.
- Current mechanism: verification borrows `../coeus/scripts/lockfile.py::unused_windows_drive`; alternating primary/lane sources can retain one relative-path Cargo identity. Changing Z: to Y: still produced artifact `2960c0092a779492`, so a different drive letter alone is insufficient. Leto's committed lock guard does not own build freshness.
- Recovery: invalidate only that crate's release fingerprint and artifacts; unchanged source then rebuilds metadata containing `transpose_copy` and passes release CTC 10/10 (`7de39bb0`). No full-cache deletion or source workaround.
- Acceptance: alternate the actual primary and lane source states under one shared cache; each build exposes exactly its own API and computes its expected values, including when the incoming source has older timestamps.
- Verification: source-identity regression plus the existing locked debug/release CTC gate; no cache fork, Cargo flag suppression, test retries, or source workaround.
- Dependencies/authority: Atlas-owned standalone runner integration; Change through merge. Prior Apollo performance evidence remains bound to its original dependency graph.

<a id="leto-square-transpose"></a>

## LETO-SQUARE-TRANSPOSE — Own checked complex matrix movement [major] [arch]
- Status: in-progress; integrator: Codex; branch: `codex/square-transpose`; updated: 2026-09-08; [PR 175](https://github.com/ryancinsight/leto/pull/175); provider-first delivery authorized, consumer acceptance remains open.
- Outcome: checked dense transpose and allocation-free, bit-preserving complex movement for [Apollo FourStep](../apollo/backlog.md#apollo-four-step-square-movement).
- Scope: core assignment, complex layout kernels and generic scalar/allocation tests; preserve FFT arithmetic, workload, manifest versions and locks.
- Acceptance: failure-atomic extents, all four scalar payload/coordinate/offset/tail oracles, no supported consumer regression or executable growth, unchanged allocation bounds.
- Design/migration: [ADR 0027](docs/adr/0027-hermes-complex-batch-transpose.md) owns contracts, rejected experiments and revision-specific acceptance.
- Consumer evidence: Apollo `3f1c0db7` with Leto `633acb7` is accepted: one efficiency-core full-real/262,144 gain, no supported regression, -512 executable bytes, 20 exact footprint windows. Cold peak maximum is +24 bytes; no general RustFFT lead. [Independent audit](../../output/apollo-square-transpose/integration/provider-graph/census/independent-audit.json).
- Provider evidence: exact `68745ef` passes format/minimal/Clippy, 940 debug and 940 release tests, 30 doctests (one existing ignored), strict Rustdoc and 24 smokes; 331 inputs and lock unchanged. [Results](../../output/apollo-square-transpose/integration/provider-delivery/final-checks.json).
- API: exact-head CI confirms intended free-function/public-module removals; [major] migration is documented, with no release or version bump.
- Dependency closure: `68745ef` includes landed CTC and requires Moirai 0.6. Apollo's old Hephaestus `242520e` requires Moirai ^0.5; adoption must advance Leto together with landed Hephaestus `1481a37`. Fresh consumer graph/size/allocation/census gates remain required; provider merge does not establish their result.

## LETO-STAGGERED-ARBITRARY-ORDER-2026-09-04 — Arbitrary-even-order staggered gradient/divergence pair [minor] — in-progress <a id="leto-staggered-arbitrary-order-2026-09-04"></a>

- **Integrator:** Claude on `feat/leto-arbitrary-order-staggered`; **lease:**
  `crates/leto-ops/src/application/diff/three_dimensional/`,
  `crates/leto-ops/src/lib.rs`, `backlog.md` — 2026-09-04.
- **Outcome:** Leto owns the derived-coefficient staggered first-derivative
  family at any even order `2N`, `N = 1..=8`, so the Yee gradient/divergence
  pair has one implementation in the stack. Closes the provider gap that keeps
  `kwavers-math` carrying `StaggeredLeapfrogOperator`,
  `StaggeredGridOperator`, and the Fornberg coefficient derivation.
- **Scope:** coefficient derivation (Fornberg 1988) over `leto_ops` LU;
  generic `T: FloatElement`; gradient (cell-centred to face-centred) and
  divergence (face-centred back to cell-centred) forming a negative-adjoint
  pair. Non-goals: device kernels (Hephaestus), the Coeus backend-generic
  seam, and the kwavers deletion — each its own increment.
- **Acceptance:** derived coefficients match the published rationals for
  orders 2-8 to `1e-13` relative; measured order of accuracy matches the
  nominal order; `D = -G^T` holds to the derived tolerance on non-degenerate
  fields; the `N = 1` gradient is bitwise identical to the existing
  `StaggeredForward` kernel wherever both are defined; check, Clippy, nextest,
  doctests, and rustdoc pass.
- **Acceptance correction (2026-09-04):** the claim first read "`N = 1` is
  bitwise identical to `StaggeredForward` / `StaggeredBackward`", which is
  false at the walls and was never true. The fixed kernels write one cell fewer
  on the differentiated axis and impose no wall closure; the leapfrog pair is
  grid-shaped and reflects taps about the wall to give the rigid
  `∂p/∂n = 0` the conservative pair needs. Equality holds on the overlap and is
  tested there; the far face is separately tested to be an unforced zero.
- **Consumer driver:** kwavers FDTD `SolverState::leapfrog_operator` runs
  `config.spatial_order` up to 8 (Fullwave 2.5 parity); atlas
  `docs/audit/math-ssot-ledger.md` rows 137-140.
- **Evidence (2026-09-04):** delivered on `feat/leto-arbitrary-order-staggered`.
  New `leto_ops::{staggered_first_derivative_coefficients,
  central_first_derivative_coefficients, TapCoefficients, MAX_HALF_ORDER}` and
  `leto_ops::{Axis, StaggeredLeapfrog3D}`. Coefficients derive through the
  crate's own LU rather than a second dense solver; taps live in inline
  `[T; 8]` storage so the operator stays `Copy` and the sweep allocation-free.
  Gate at this revision: `cargo fmt --check`, `cargo check --locked -p leto
  -p leto-ops --no-default-features`, `cargo clippy --locked -p leto -p leto-ops
  --all-targets -- -D warnings`, `cargo nextest run --locked -p leto -p leto-ops`
  917/917, `cargo test --doc` 26 passed, `cargo doc --no-deps` — all clean.
  15 new tests: published rationals for staggered orders 2-8 and collocated
  2-6, measured order of accuracy at 2/4/6/8, the negative-adjoint identity on
  all three axes at every order, wall closure, traversal agreement across the
  three axis kernels, the Courant derivation, and an f32/f64 instantiation
  pair.
- **Next:** `KW-*` deletes `kwavers-math`'s `staggered_leapfrog`,
  `staggered_grid`, and `central_difference_2/4/6` against this provider
  surface; the Coeus `FiniteDifference3DOps` seam and the Hephaestus device
  kernels follow as their own increments.
- **Last-update:** 2026-09-04.

## LETO-MNEMOSYNE-DEFAULT-2026-09-04 — Follow reviewed Mnemosyne main [patch] [arch] — review <a id="leto-mnemosyne-default-2026-09-04"></a>

- **Integrator:** Codex on `build/leto-provider-default`; **lease:** none.
- **Outcome:** remove the obsolete Mnemosyne PR #123 pin after confirming its
  Eunomia source correction already exists on reviewed `main`.
- **Acceptance:** standalone metadata resolves one Eunomia identity from
  provider defaults; configured Leto gates pass; no stale Mnemosyne allocator
  changes enter the graph. **Last-update:** 2026-09-04.
- **Evidence:** one Eunomia/eunomia-derive source at `02397fa`; Mnemosyne
  default resolves at `8d2f466`; all-target check and Clippy, format, 923/923
  nextest, 23 executable doctests with one ignored, and rustdoc pass. Independent
  upstream review rejected PR #123 while confirming its source correction is
  already valid on `main`.

## LETO-HERMES-IDENTITY-2026-09-03 — Align leto-ops with the Hermes provider source identity [patch] [arch] — in-progress <a id="leto-hermes-identity-2026-09-03"></a>

- **Integrator:** Codex on `build/leto-moirai-source-identity`; **lease:**
  `Cargo.toml`, `Cargo.lock`, `backlog.md`.
- **Outcome:** Advance Leto's workspace Hermes edge to PR #155 so consumers
  resolve one first-party SIMD and memory-provider source graph.
- **Acceptance:** Standalone lock resolves Hermes `5a399ee`, Mnemosyne
  `mnemosyne#123` (proposed, not merged), and Eunomia `fdbf122`; workspace
  check, Clippy, nextest, doctests, rustdoc, and diff checks pass; no adapter
  or compatibility layer.
- **Follow-up source edge:** Moirai PR #256 merged at `70d201a`; this increment
  removes its temporary revision pin and regenerates `Cargo.lock`. Hermes,
  Mnemosyne, and Eunomia remain pinned until their provider increments merge.
- **Dependency:** Hermes PR #155 (`5a399ee`); **Last-update:** 2026-09-04.

## LETO-LAYOUTDYN-CONTRACT-2026-09-03 — Complete runtime-rank layout contracts [minor] [arch] — in-progress

- **Outcome:** extend `LayoutDyn` with checked physical-span bounds, exact
  injectivity validation, and zero-copy-compatible broadcasting so Hephaestus
  fusion can validate runtime-rank views through Leto's canonical layout seam.
- **Scope/non-goals:** `crates/leto` shared layout kernels, `LayoutDyn` API,
  dynamic-rank value tests, ADR 0007 and crate documentation. No dynamic-rank
  compute substrate, element storage changes, or downstream adapter.
- **Acceptance:** const-rank and dynamic-rank layouts delegate the same
  rank-agnostic arithmetic; valid, aliased, empty, negative-stride, overflow,
  incompatible-broadcast, and value-preserving broadcast cases are tested;
  the public API is warning-clean and Hephaestus's runtime-rank caller
  compiles against the locked source.
- **Integrator:** Codex atlas-session; **lease:** `crates/leto/src/domain/layout/`,
  `crates/leto/src/domain/dynamic/`, `crates/leto/tests/core/dynamic.rs`,
  `docs/adr/0007-dynamic-rank-boundary.md`, `crates/leto/README.md`,
  `Cargo.toml`, `Cargo.lock`;
  **last-update:** 2026-09-03.

## ATLAS-LETO-OP-PERF-2026-08-28 — Operator buffer reuse and single-write reductions [patch] — in-progress

- **Integrator:** claude-fable session 03d80d33 subagent.
- **Lease:** `crates/leto/src/application/arithmetic.rs`,
  `crates/leto-ops/benches/kernels.rs`, `CHANGELOG.md`, `backlog.md`.
- **Last-update:** 2026-08-28.
- **Members:** `ATLAS-LETO-OPERATOR-OWNED-LHS` (done, `62a0434`),
  `ATLAS-LETO-REDUCE-SINGLE-WRITE` (stays filed — see its entry).

### ATLAS-LETO-REDUCE-SINGLE-WRITE — `reduce_axis` zero-fills a fully-overwritten output [patch] — todo

- Owner: unclaimed (examined 2026-08-28 under `ATLAS-LETO-OP-PERF-2026-08-28`;
  deliberately not implemented). Original evidence: audit 2026-08-27 —
  `reduce_axis` allocates `VecStorage::fill(size, T::ZERO)` then
  `reduce_axis_into` writes every element; for small `axis_len` the memset is
  up to ~50% extra write traffic.
- **Coverage proof — established**, so a later attempt need not redo it. All
  three routes of `reduce_axis_into` write each output offset exactly once:
  (1) the `N == 2 && axis == 0` fast path writes `col in 0..cols` and output
  shape is `[1, cols]`, in both the `rows == 0` and general branches;
  (2) the serial route iterates `flat_idx in 0..out_size` through the
  bijection `offset_of(index_from_flat(..))`, which for the fresh
  offset-0 C-contiguous output is the identity onto `0..out_size`;
  (3) the parallel route's `parallel_for_chunks` covers `0..len` disjointly
  (`start = idx * chunk`, `end = min(start + chunk, len)`, `idx` over
  `len.div_ceil(chunk)`) and each `flat_idx` writes once.
- **Cost premise is weaker than filed.** `VecStorage::fill` is
  `vec![value; len]`, and for a zero-valued primitive that hits std's
  `SpecFromElem` specialization: measured `alloc_zeroed` +1 / plain alloc +0
  for `f64` zero, versus plain +2 for a nonzero element. So the fill is a
  `calloc`, not malloc-plus-memset — for a large fresh output the zero pages
  come from the OS with no write traffic, and the first-touch page faults are
  paid by the single write either way. Redundant write traffic is real only
  for small or allocator-recycled blocks, well below the filed ~50%.
- **Blocker (not the proof — the API shape).** `reduce_axis_into` is public and
  writes through `&mut ArrayViewMut<'_, T, N>`, whose `data_mut()` yields
  `&mut [T]`; the parallel route derives its `*mut T` from that same slice.
  Constructing a `&mut [T]` over uninitialized memory is UB regardless of
  coverage, so single-write construction requires either changing that public
  signature to carry a `MaybeUninit<T>` output or duplicating the ~140-line
  three-route body over an uninit output — the second being the duplication
  the consolidation rules forbid. Both exceed [patch] scope.
- **Direction if resumed:** reclassify as [minor] and make the output type a
  parameter of one generic body (an output-slot abstraction implemented for
  both `&mut [T]` and `&mut [MaybeUninit<T>]`) so there is still one
  reduction traversal; then panic-safety for a partially-initialized buffer
  and miri coverage become tractable. Weigh it against the corrected cost
  model above — the win may not justify the unsafe surface.

## LETO-CRATES-METADATA-1 — Satisfy registry metadata [patch, blocked]

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

## LETO-PYTHON-RELEASE-1 — Python release wheels [patch, in progress]

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

## LETO-INPLACE-INTENSITY-GATE-2026-09-01 — `map_inplace` has no intensity gate [minor] — blocked (re-open: first bandwidth-bound in-place caller)

- **Outcome:** give bandwidth-bound ops an in-place path that reaches the
  cache-residency gate, as `unary_map_into` already does for the into-output
  form. `UnaryOp::COMPUTE_BOUND` has no in-place consumer: `unary_map_into`
  honors it, and nothing else does.
- **Driver.** `LETO-PARALLEL-INTENSITY-1` (below) records acceptance as *"every
  bandwidth-bound elementwise path gates on cache residency"*. That is not
  true of `map_inplace`, which was outside the converted set — its scope named
  `unary.rs`, but only the `_into` entries moved. `map_inplace` still gates on
  the flat `PARALLEL_THRESHOLD` element count for every op and every scalar.
  That item's "Remaining: none for the threshold policy" is therefore
  overstated; this item carries the remainder rather than reopening it.
- **Evidence** (36 MiB-L3 AVX2 host; `map_inplace` against a hand-written
  sequential loop over the identical slice and the identical closure, both
  monomorphized from the same generic `F: Fn(T) -> T`, best-of-9 blocks). The
  control reads ratio 1.00 at every sub-threshold length, which is what
  establishes that the comparison isolates dispatch:

  | elems | KiB (f64) | f64 ratio | f32 ratio | dispatch |
  |---|---|---|---|---|
  | 16384-49152 | 128-384 | 1.00 | 1.00 | sequential |
  | 65536 | 512 | **5.85** | **8.45** | parallel |
  | 98304 | 768 | **4.85** | **7.82** | parallel |
  | 131072 | 1024 | **6.44** | **9.80** | parallel |
  | 262144 | 2048 | 2.14 | 5.37 | parallel |
  | 524288 | 4096 | 1.09 | 3.71 | parallel |
  | 1048576 | 8192 | 0.29 | 0.35 | parallel |
  | 4194304 | 32768 | 0.16 | 0.31 | parallel |

  Ratios above 1.0 are parallel losing. The bandwidth-bound crossover sits near
  **1M elements for both scalars** here. **Correction (2026-09-01):** this
  originally read as evidence against the LLC-byte model the `_into` and binary
  gates use. Measured directly on the binary path with a same-binary three-arm
  probe (gate / sequential / naive scoped threads), the byte model holds: naive
  parallel breaks even at 1.57M for f64 and between 2.1M and 3.1M for f32 —
  exactly the bytes-over-L3 prediction, and nowhere near the ~3x-early
  crossover an element count would imply. The in-place figure does not transfer
  to a three-stream operation and is not a finding against that gate. A
  compute-bound control (`sin`, `f64`) inverts cleanly: 0.15 at the same 65536
  gate, confirming the flat constant is right for the case it now serves.
- **Scope/non-goals:** a typed in-place entry (`unary_map_inplace`) routing
  through the existing `map_into_gated` policy. No change to `map_inplace`'s
  raw-closure behavior — its eager default is deliberate and documented, and
  changing it would regress compute-bound closures by the inverse factor. No
  new threshold constant; no change to the `_into` or binary gates.
- **Acceptance:** value-identical to `map_inplace` for the same op; a
  `COMPUTE_BOUND = false` op stays sequential below the cache gate and matches
  the sequential control within noise across 64 KiB-2 MiB; a `COMPUTE_BOUND`
  op parallelizes exactly as today; zero allocations after caller storage.
- **Deferred, with reason — no caller.** `AbsOp`/`NegOp` declare
  `COMPUTE_BOUND = false` but no in-stack consumer maps them in place: a stack
  sweep of `map_inplace(` finds no caller in leto, apollo, hephaestus, or
  kwavers. Building the API now would be speculative generality. The hazard is
  documented at the call site instead, so a consumer meets it in Rustdoc rather
  than in a profile.
- **Re-open trigger:** the first bandwidth-bound in-place caller in the stack,
  or an external report against the published `map_inplace`.
- **Trigger re-checked 2026-09-02:** still unfired. The stack's only in-place
  mappers are kwavers-boundary's `indexed_map_inplace` calls (CPML, adaptive
  coupling), and that entry is a sequential row walk with no parallel gate, so
  it neither pays nor needs the threshold this item concerns.
- **Integrator:** Claude session 03d80d33. **Last-update:** 2026-09-01.

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

### Stage A2 — ndarray consolidation (support coeus/apollo)
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

## Phase 2: ndarray API Parity Required by Apollo [minor]
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

## Phase 3: Coeus Tensor Substrate Requirements [minor]
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

## Phase 4: Operations, Performance, and Architecture [minor]
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

## Phase 5: Python and Interop [minor]
- [ ] Keep Python as a thin PyO3/NumPy boundary over Rust operations.
- [x] [patch] Resolve the reopened `numpy-0.23.0` rustdoc ICE in the Python FFI
  documentation path. `leto-python` is a PyO3 extension boundary, not a Rust
  library API surface, so Cargo no longer invokes rustdoc for that target
  (`doc = false`). Full workspace docs now complete without excluding
  `leto-python`. Verification: package docs, full workspace docs, package
  clippy, and package nextest all pass.
- [x] Replace current Python result construction that clones through `Vec` after computation. Verification: `leto-python` now transfers owned `VecStorage` with `Array::into_vec()` and `PyArray1::from_vec`, then reshapes without the former `as_mut_slice().to_vec()` clone path.
- [x] Add Python boundary tests for shape validation, C-contiguous input, rejected non-contiguous inputs, and value parity with NumPy-visible outputs. Verification: `leto-python` unit tests cover `add`, `sum`, `matmul`, shape mismatch rejection, and a real NumPy transposed non-contiguous input.

## Phase 6: Coeus Backend Consolidation [arch]
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

## Phase 7: ndarray Parity Completion (Apollo hot kernels) [minor]

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

## Phase 9: Blocked-reflector vectorization (eig/SVD disparity) [major]
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
