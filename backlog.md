# Leto Work Backlog

<a id="leto-square-transpose"></a>
## LETO-SQUARE-TRANSPOSE — Own checked complex matrix movement [minor] [arch]

- Status: in-progress; integrator: Codex; branch: `codex/square-transpose`; updated: 2026-09-06.
- Outcome: checked borrowed dense transpose and allocation-free, bit-preserving complex square movement for [Apollo FourStep](../apollo/backlog.md#apollo-four-step-square-movement).
- Scope: core assignment/materialization and `transpose_copy`, complex layout kernels, generic Clone/scalar and allocation tests; preserve batch dispatch and FFT arithmetic.
- Acceptance: exact extent validation before mutation, all four scalar payload/coordinate/offset/tail oracles, no supported consumer regression or executable growth, unchanged allocation bounds.
- Design: [ADR 0027](docs/adr/0027-hermes-complex-batch-transpose.md); Leto owns movement, Hermes registers, Apollo scheduling and scratch.
- Dependencies: locked Hermes `07c5e5f` and Eunomia `02397fa`; no new dependency or version bump. Review pins merge upstream before final consumer delivery.
- Source reconciliation: ADR 0029 supersedes the prior pin-only branch; unmerged Mnemosyne work remains on `perf/mnemosyne-scratch-release`.
- Verification: committed CI commands and Nextest 30/60-second budgets; final dense-copy revision passes 930 native tests, 366 release tests, 28 doctests (one existing ignored), Clippy, minimal features, rustdoc and 24 smoke cases.
- API/review: both public packages pass 196 SemVer checks against `a2006ad` (58 inapplicable checks each); independent source review finds no production defect.
- Evidence: `output/apollo-square-transpose/dense-copy/leto-gates/final-checks.json` records the exact 14-file hash set and unchanged lock. No timing or size acceptance follows from provider gates.
- Experiment state: earlier builds fail size or regression acceptance; tile-span iteration additionally introduces AVX-512 division and payload spills and is removed. ADR 0027 owns those findings.
- Next: integrate the verified forward restoration; the square-movement campaign remains in-progress with no accepted consumer candidate.
- Preflight verification: unchanged 30 focused tests pass in debug and release; all-target Clippy, format and independent source review pass. Only two inline annotations change production; consumer acceptance remains pending.
- Tile-diagnostic outcome: rejected despite clean source gates; executable +8,192 bytes and supported efficiency-core complex/1,024 regression of 0.59–4.22%. [Codegen](../../output/apollo-square-transpose/tile-diagnostics/codegen.md) and [census](../../output/apollo-square-transpose/tile-diagnostics/audit-summary.json) trigger restoration; prior candidate still fails size acceptance.
- Restoration verification: tile source exactly matches `9a47d6b`; format, Clippy and unchanged 30 focused debug/release tests pass. Evidence: `output/apollo-square-transpose/tile-diagnostics/restoration/final-checks.json`; no timing rerun or lock change.

## LETO-FD-MUTABLE-VIEW-DST-2026-09-04 — Take a mutable view as the FD destination [major] — review <a id="leto-fd-mutable-view-dst-2026-09-04"></a>

- **Integrator:** Claude on `feat/leto-fd-mutable-view-dst`; **lease:**
  `crates/leto-ops/src/application/diff/three_dimensional/`, `backlog.md` —
  2026-09-04.

- **Outcome:** `FiniteDifference3D::apply_{x,y,z}_into` and
  `StaggeredLeapfrog3D::{gradient_into, divergence_into}` accept a mutable
  3-D view rather than `&mut Array3<T>`, so a caller whose destination is not
  a Leto-owned array can write into it without a copy. `laplacian_2d_into`
  already takes `&mut ArrayViewMut1`; this brings the 3-D surface to the same
  contract.
- **Driver (found 2026-09-04, drafting the Coeus seam):** the Coeus backend
  seam hands a CPU kernel `&mut [T]` out of a `DeviceBuffer` through
  `CpuAddressableStorageMut::as_mut_slice`. A `&mut Array3<T>` parameter cannot
  receive that slice — `Array3` owns its storage — so a `coeus-leto`
  implementation of a finite-difference op would have to allocate an `Array3`
  per sweep and copy the result back. That is an allocation inside an FDTD
  timestep: the exact defect class the kwavers consolidation
  (`KW-LETO-FD-SSOT`, ADR 128) just removed. The seam is blocked on this and
  should not be built around a copy.
- **Scope:** the destination parameter only. Interior kernels already work
  through `slice_mut`/`as_slice_mut`/`fill`/index, all of which a mutable view
  provides. Non-goals: the input side (already a view), the fixed 2-D
  Laplacian (already correct), and the Coeus/Hephaestus seams themselves.
- **Acceptance:** every 3-D FD entry point takes a mutable view; a test writes
  through a view over a plain `&mut [T]` and gets the same values as the
  owned-array path; existing behaviour is unchanged elsewhere; check, Clippy,
  nextest, doctests, and rustdoc pass.
- **Change class:** `[major]`. Call sites pass `&mut dst.view_mut()` instead of
  `&mut dst`. In-repo call sites and the kwavers consumer migrate in the same
  co-evolution cycle — no borrowed/owned sibling methods, which would fork the
  surface this row exists to unify.
- **Blocks:** the Coeus `FiniteDifference3DOps<T>: ComputeBackend` seam (trait
  draft written and held), and behind it the Hephaestus 3-D device kernels and
  the deletion of `kwavers-gpu`'s FDTD shader copy (ADR 128, "What this does
  not yet do").
- **Delivered 2026-09-04.** `FiniteDifference3D::apply_{x,y,z}_into` and
  `StaggeredLeapfrog3D::{gradient_into, divergence_into}` take
  `&mut ArrayViewMut3<'_, T>`. The kernels needed only the two adaptations the
  feasibility check predicted. Two new tests write through a view over a plain
  `Vec<f64>` this crate does not own and compare bitwise against the owned
  path — the fixed-scheme case and, on all three axes for both operators, the
  leapfrog pair, whose contiguous fast path through `as_mut_slice` is the one
  that would have degraded silently. Gate: `cargo fmt --check`,
  `cargo check --locked --no-default-features`, `cargo clippy --locked
  --all-targets -- -D warnings`, `cargo nextest run --locked` **919/919**,
  `cargo test --doc` 26, `cargo doc --no-deps` — all clean.
- **Feasibility, checked against `application/view.rs` (2026-09-04):**
  `ArrayViewMut` already carries everything the kernels use — `shape`, `fill`,
  `slice_mut`, `as_mut_slice`, `IndexMut`, and `reborrow`. Two mechanical
  adaptations: the owned path's `as_slice_mut` is `as_mut_slice` on the view,
  and `slice_mut` consumes `self`, so each of the repeated interior/boundary
  slices needs a `reborrow()` first. No missing capability; this is a signature
  and call-site change, not a new kernel.
- **Last-update:** 2026-09-04.

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

## LETO-EUNOMIA-IDENTITY-2026-09-04 — Follow merged Eunomia source [patch] [arch] — done 2026-09-04 <a id="leto-eunomia-identity-2026-09-04"></a>

- Merged-provider pins removed; the temporary Mnemosyne pin was closed by
  `LETO-MNEMOSYNE-DEFAULT-2026-09-04`. One Eunomia source; check, Clippy,
  923/923 nextest, 23 doctests, and rustdoc pass.

## LETO-HERMES-IDENTITY-2026-09-03 — Align leto-ops with the Hermes provider source identity [patch] [arch] — in-progress <a id="leto-hermes-identity-2026-09-03"></a>

- **Integrator:** Codex on `build/leto-moirai-source-identity`; **lease:**
  `Cargo.toml`, `Cargo.lock`, `backlog.md`.
- **Outcome:** Advance Leto's workspace Hermes edge to PR #155 so consumers
  resolve one first-party SIMD and memory-provider source graph.
- **Acceptance:** Standalone lock resolves Hermes `5a399ee`, Mnemosyne
  `da5c6be`, and Eunomia `fdbf122`; workspace check, Clippy, nextest,
  doctests, rustdoc, and diff checks pass; no adapter or compatibility layer.
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

## ✅ LETO-F16-HERMES-ROUTING-2026-09-02 — Route F16 slice operations through hermes [minor] [perf] — done 2026-09-02

- **Finding:** `impl_simd_ops_unsupported!(F16)` was stale — hermes 0.7 serves every operation the `SimdStrategy` routes (elementwise add/sub/mul/div, sum, dot, axpy, axpy rows, gemv, tiled GEMM, abs-sum/abs-max, min/max, jaccard/hamming) at `F16` on the scalar, AVX2 (+F16C), AVX-512 and NEON backends. The follow-up Bf16 consumer route is tracked in `LETO-BF16-HERMES-ROUTING-2026-09-02` now that Hermes provides the same backend coverage.
- **Delivered:** `F16` takes `impl_simd_ops_native!` and the SIMD-routing `Scalar` impl (`impl_scalar_simd!` now takes the index conversion, so the reduced-precision constructor fits it); `f16_slice_operations_route_through_hermes_and_match_scalar_semantics` pins bitwise equality for the elementwise ops (single rounding from the `f32` intermediate on every backend), an `n·u` bound for sum/dot, exact min/max, and the public `add` reaching the same route. leto-ops suite and clippy `-D warnings` green.

## LETO-BF16-HERMES-ROUTING-2026-09-02 — Route Bf16 slice operations through Hermes [minor] [perf] — done 2026-09-02

- **Integrator:** Codex atlas-session; **branch:** `perf/leto-bf16-hermes`;
  **lease:** `crates/leto-ops/src/domain/strategy.rs`,
  `crates/leto-ops/src/domain/scalar/impls.rs`,
  `crates/leto-ops/tests/ops/elementwise.rs`; **last-update:** 2026-09-02.
- **Finding:** Hermes 0.7 now exposes Bf16 `LaneScalar` and already supplies
  the scalar, AVX2, AVX-512, and NEON `BackendKernel` implementations. Leto's
  Bf16 `Scalar` implementation still bypasses `impl_scalar_simd!`, so every
  Bf16 slice operation uses the scalar fallback while F16 takes Hermes.
- **Acceptance:** Bf16 uses the existing generic `impl_simd_ops_native!` and
  `impl_scalar_simd!` paths; elementwise results match the Bf16 scalar
  contract bitwise, reductions satisfy the derived `n·u` bound with
  `u = 2⁻⁸`, min/max remain exact, and the public operation path is covered.
  No new arithmetic or allocation path is introduced.
- **Non-goals:** Hermes provider implementation, GEMM algorithm changes,
  runtime dispatch redesign, or allocator changes.
- **Disposition:** completed by removing the Bf16 scalar-only implementation
  and routing it through the existing generic Hermes path. The reduced-
  precision conformance helper now instantiates both F16 and Bf16. Evidence:
  `cargo nextest run --offline -p leto-ops --no-fail-fast` 549/549 passed;
  doctests 21/21 passed; Rustdoc generated; Clippy and diff checks passed;
  `cargo semver-checks` passed 196/196 applicable checks with 58 skipped.

## ✅ LETO-TILE-WIDTH-RUNTIME-GEOMETRY-2026-09-01 [minor] [perf] — done 2026-09-02

- **Outcome:** strided unary and binary map kernels now use the process-cached
  `CacheGeometry::cache_line_bytes()` value to derive their micro-tile side;
  explicit `map_into_with_cache_geometry` and
  `binary_map_with_cache_geometry` entry points support authoritative external
  topology and injected policy measurements. `CacheGeometry::with_cache_line_bytes`
  validates positive widths and retains conservative capacity fallbacks.
- **Evidence:** injected, pinned one-core Criterion runs over transposed
  `256 × 256` `f64` input consistently favored line-128. The clean 10-sample
  confirmation measured line-64 at `[21.901, 22.042] µs` (median 21.945)
  and line-128 at `[19.339, 19.474] µs` (median 19.402); a 20-sample run
  measured `[23.488, 24.633] µs` (median 23.937) versus `[18.198, 18.473]
  µs` (median 18.315). The exact medians vary with host frequency, but the
  direction and separated intervals persist. `cargo nextest run -p leto-ops`:
  548/548; doctests 21/21; Clippy `--all-targets -D warnings`, rustdoc, and
  196/196 applicable minor SemVer checks pass.
- **Integrator:** Codex atlas-session; **branch:** `perf/leto-runtime-tile-geometry`; **lease:** none; **commit:** pending PR.

- **Context:** `ATLAS-LETO-CACHE-LINE-TOPOLOGY` (PR #137) made
  `cache_line_bytes()` report the width themis actually observes instead of an
  unconditional 64. That accessor has **no in-tree consumer**, so the merged
  fix changes no kernel decision. The decision that does depend on line width
  is `line_elements<T>()` (`crates/leto-ops/src/application/index.rs:178`),
  which hardcodes `64 / size_of::<T>()` as the micro-tile side and is
  documented as such at lines 105-113. It is a `const fn`, so it cannot read
  runtime geometry without restructuring `TileGeometry` construction.
- **Impact is bounded, not a defect.** Under-estimating is the conservative
  direction for tiling: a 64-derived tile on a 128-byte-line part touches each
  line twice rather than consuming it once — lost reuse, never wrong results.
  Over-estimating is the harmful direction (a 128-assumption on a 64-byte part
  quadruples a 2-D tile's working set and can overflow the L1 budget), which is
  why 64 stays the fallback. So this item buys measured throughput on
  wide-line parts; it does not fix a bug.
- **Acceptance oracle:** a benchmark on a part reporting 128-byte lines shows
  the runtime-geometry tile beating the const-64 tile beyond noise, on pinned
  cores, with the baseline stored. **If the delta is inside noise, the correct
  outcome is to close this item unfixed and record 64 as measured-adequate** —
  do not land the restructuring on the strength of the reasoning above.
- **Constraint:** this host reports 64-byte lines at every level, so it
  **cannot** produce the discriminating measurement. Either source a
  128-byte-line part or drive `TileGeometry` from an injected geometry in a
  bench harness so the tile side varies without the topology varying. State
  which was used; a 64-byte-host measurement is not evidence either way.
- **Non-goals:** false-sharing padding — a different width whose safe error
  direction is the opposite one, and which must not reuse this constant.

## ATLAS-LETO-CACHE-LINE-TOPOLOGY-2026-09-01 — Cache-line width read from topology [patch] — done 2026-09-01

- **Outcome:** `CacheGeometry::cache_line_bytes` reports the platform width
  `themis` already detects on both backends, instead of always returning the
  64-byte fallback constant.
- **Scope/non-goals:** `geometry_from_cache_levels` and its unit tests in
  `crates/leto-ops/src/infrastructure/cache.rs`, plus CHANGELOG. No change to
  `CacheGeometry`'s shape or accessors, to L1/L2/L3 capacity selection, or to
  any tiling policy that consumes the geometry.
- **Acceptance:** a reported `line_bytes` reaches `cache_line_bytes()`
  (including widths narrower than the fallback); the widest width across levels
  governs when several are reported; typed absence (`None` or zero) resolves to
  `FALLBACK_CACHE_LINE_BYTES` at one documented site.
- **Evidence:** the three line-width tests were run against the reverted
  production path and all three fail there returning `64` (expected `128`,
  `128`, `32`), so they discriminate the fix rather than pin the old behavior;
  the two absence tests pass under both, which is their contract. Gates at
  `HEAD`: fmt clean, Clippy `--all-targets -D warnings` clean, Nextest 542/542,
  21/21 doctests.
- **Residual:** `line_elements<T>()` in `crates/leto-ops/src/application/index.rs`
  still hardcodes the 64-byte line as a `const fn`; it is the only consumer-side
  cache-line constant found and is out of this item's scope. `cache_line_bytes()`
  itself currently has no in-tree consumer, so this fix corrects the reported
  value without changing any kernel decision yet.
- **Integrator:** Claude session 5050c72a; **lease:** none.
  **Last-update:** 2026-09-01.
- **Independent review (2026-09-01, Claude):** the evidence claim reproduced.
  With `cache_line_bytes` reverted to the constant, exactly the three
  line-width tests fail — `cache_levels_override_capacities_and_line_width`,
  `widest_reported_line_width_wins_across_levels`,
  `reported_line_width_narrower_than_the_fallback_is_honoured` — and the eight
  others pass, so they discriminate the fix rather than pin old behaviour.
  Merged as PR #137. The `line_elements<T>()` residual is tracked as
  `LETO-TILE-WIDTH-RUNTIME-GEOMETRY-2026-09-01`.

## ✅ ATLAS-LETO-HERMES-COMPLEX-TRANSPOSE-2026-09-01 — Register-tiled complex matrix batches [minor, perf] — done 2026-09-02

- **Outcome:** add one Leto-owned, allocation-free C-destination/F-source
  complex matrix-batch operation that uses Hermes register-resident square
  tiles only for the measured high-count small-matrix regime and preserves the
  existing generic transpose for every other shape, scalar, and capability.
- **Scope/non-goals:** `crates/leto-ops` layout operation, exact value
  and allocation tests, a bounded provider instrument, API/Rustdoc, ADR,
  CHANGELOG, and Apollo consumer integration. No change to generic `assign`,
  FFT arithmetic, scheduler policy, or public compatibility layer.
- **Acceptance:** validate all lengths before mutation; preserve source-major
  matrix order and ragged/asymmetric tails for `Complex<f32>` and
  `Complex<f64>`; allocate zero times after caller storage exists; select
  Hermes only at the operation boundary when native complex lanes are useful;
  retain a value-identical generic fallback; and reproduce the measured Apollo
  3-D improvement without regressing its warm allocation census.
- **Integrator:** Codex `/root`; **lease:** none. **Evidence:** provider source
  `63d5cab` passes 3/3 value and failure-atomicity cases for f32/f64/F16, 1/1
  warmed zero-allocation census, release repeats, AArch64 warning-denied
  compilation, all-target/all-feature Clippy, 540/540 package Nextest, 21/21
  doctests, Rustdoc, and 196/196 minor SemVer checks; two local AVX2 Criterion
  runs retain the selected regime (ADR 0027).
- **Closed 2026-09-02.** The integrator's claim went stale in `review`;
  re-verified against the current tree rather than re-run: the operation is
  published (`leto_ops::transpose_complex_matrices`), routed through hermes
  register tiles in `application/layout/complex/tile.rs`, covered by
  `tests/complex_transpose_allocations.rs`, recorded in ADR 0027 and the
  CHANGELOG, and consumed by apollo at
  `apollo-fft/.../plan/fft/layout.rs:57`. `cargo nextest run -p leto-ops`:
  546/546 pass. Every acceptance clause is satisfied by the merged tree, so
  the item closes on evidence.

## ATLAS-LETO-QR-REFLECTOR-ACCESSORS-2026-08-31 — Read side for the QR compact reflector storage [minor] — done 2026-08-31

- **Delivered:** `QrDecomposition::{packed, heads, betas}` — the read
  counterpart to the existing `from_raw_parts`, so the hephaestus WGPU backend
  accumulates `Q` device-side rather than materializing it host-side via `q()`.
  Indexing pinned by `reflector_accessors_reproduce_q` (bitwise match to `q()`).
  Gates: fmt clean, clippy `-D warnings` clean, nextest 535/535, doctests 20/20.
  PR open, unmerged.
- **Integrator:** Claude session 5050c72a.

## LETO-MATMUL-PARITY-VERDICT-2026-08-28 — Dense matmul oracle parity re-measured and closed [patch] — done 2026-08-28

- **Delivered:** the open "dense matmul remains slower than ndarray/nalgebra"
  thread closes on fresh evidence, not new kernel work. A pinned same-binary
  external probe (leto path-dep beside ndarray 0.16 / nalgebra 0.34, outside
  the repo per the dependency policy; best-of-24 blocks, per-core pinning,
  three-engine value agreement < 1e-6) at HEAD `f527685` measures the SERIAL
  kernel 15–23 % ahead of ndarray at every oracle shape on the P-core (64²
  0.77x … 512² 0.85x) and 2–3x ahead of nalgebra; the default parallel entry
  is 1.8–15x ahead of the references' single-threaded execution (neither
  reference threads at default features, so that ratio measures Leto's
  runtime advantage, not kernel parity — the serial row above is the
  kernel-parity evidence). The recorded deficit predated the re-landed dense
  `T::tiled_gemm` route and the Hermes lane overhaul. Full table, method, and
  limits: `gap_audit.md` (dense matmul oracle parity — CLOSED). The
  packing-scratch / register-micro-kernel lever is retired; re-open trigger
  is an `oracle_compare/matmul_leto_*` median regression or a fresh external
  re-comparison, with the rejected-candidate list still binding.
- **Integrator:** Claude session 5050c72a.

## ATLAS-LETO-QUALITY-2026-08-27 — Mutable-iteration soundness + kernel fast paths [patch] — done 2026-08-27

- **Delivered:** PR #129 (`dfea36f`..merge) — injectivity gates on
  `lanes_mut`/`axis_iter_mut`, window-exclusivity tracking with gated
  whole-window accessors, one shared `validate_mutable_output` across 18
  leto-ops entry points, `sum` storage validation, `kron` checked size math,
  dense `fill` fast path; PR #130 — `scaled_add` slice path, F-dense
  `to_contiguous` via the tiled transpose, shared-dense-order map fast paths,
  single-zero matmul scratch, rustfmt cure for #129's fmt red. Gates:
  nextest 888/888, clippy `-D warnings` clean, fmt clean, doctests green.
- **Integrator:** claude-fable session 03d80d33 (atlas
  ATLAS-PROVIDER-CHAIN-QUALITY-2026-08-27).

## ATLAS-LETO-SUM-RESULT-UNIFICATION — `sum` panics where `reduce_all` returns Result [minor] — done 2026-09-01

- Owner: unclaimed. Evidence: audit 2026-08-27 + PR #129 — `sum` keeps its
  infallible `T` signature (now asserting storage validity) while
  `reduce_all` returns `Result<T>` for the same input class; one reduction
  surface should carry one failure contract. Breaking ([minor] 0.x); needs a
  consumer sweep (coeus backend adapters) in the same co-evolution unit.

- **Resolved (2026-09-01, Claude) — one implementation, two honest signatures.**
  The two surfaces differ in failure class, not by accident: `reduce_all`
  returns `Err` for an empty input under an op with no identity (`MinAxis`,
  `MaxAxis`), a genuine input-dependent failure; `sum` has an identity, so
  its only failure was the malformed-view invariant that validated
  construction cannot reach — a programmer error, which the panic policy
  keeps as a panic. So the contract stays split, but the *implementation*
  no longer is: `sum` was a second copy of `reduce_all`'s traversal (same
  fast rows, same element order, so bitwise-identical results) and now
  delegates to `reduce_all::<SumAxis>` with the impossible failure
  `expect`ed at the boundary. Net −34 lines. No consumer sweep needed:
  `coeus_ops::sum` already wraps in `Result` on its side. Workspace gate
  clean: 910/910, Clippy, Rustdoc, fmt.

## ATLAS-LETO-OP-PERF-2026-08-28 — Operator buffer reuse and single-write reductions [patch] — in-progress

- **Integrator:** claude-fable session 03d80d33 subagent.
- **Lease:** `crates/leto/src/application/arithmetic.rs`,
  `crates/leto-ops/benches/kernels.rs`, `CHANGELOG.md`, `backlog.md`.
- **Last-update:** 2026-08-28.
- **Members:** `ATLAS-LETO-OPERATOR-OWNED-LHS` (done),
  `ATLAS-LETO-REDUCE-SINGLE-WRITE` (stays filed — see its entry).

### ATLAS-LETO-OPERATOR-OWNED-LHS — operator chains allocate per term [minor] — done 2026-08-28

- **Delivered:** `62a0434` — owned-receiver `Add`/`Sub`/`Mul`/`Div`/`Neg` impls
  writing through the consumed lhs allocation. Allocation counts (counting
  global allocator, `crates/leto/tests/operator_allocations.rs`): 3-term chain
  2 → 1, 5-term 4 → 1, owned scalar/neg → 0. Pinned criterion (P-cores 0-3):
  3-term 64x64 1.505 → 1.323 us (−12%), 5-term 64x64 3.658 → 2.481 us (−32%);
  3-term 256x256 unchanged within CI (bandwidth-bound). Miri clean over the
  arithmetic suite (14/14).
- **Correction to the filed evidence:** the audit's "closure traversal bypasses
  the leto-ops SIMD `apply_slice` kernels" is not actionable in core —
  `leto-ops` depends on `leto`, so the SIMD tier is strictly downstream and
  core calling it would be circular. Routing operators through those kernels
  needs the ADR 0001 kernel-relocation option that ADR 0004 explicitly
  declined; not reopened here.

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

## ✅ ATLAS-LETO-MINMAX-NAN-CONTRACT — axis min/max NaN semantics undocumented and route-dependent [patch] — done 2026-09-02

- **Delivered:** `MinAxis`/`MaxAxis` seed their fold from `T::MAX_VALUE`/`T::MIN_VALUE`, so a leading NaN is rejected like any other; hermes #129 gave `min_slice`/`max_slice` the same contract (NaN ignored, all-NaN → identity) and the lock advances to it. Contract documented on both markers.
- **Evidence:** `min_max_axis_ignore_nan_lanes_on_both_routes` (f32, f64) — NaN leading/interior/trailing and an all-NaN lane, on the contiguous (SIMD) and strided (fold) routes, bitwise-equal expectations.
## ATLAS-LETO-MNEMOSYNE-SINGLE-WRITE-2026-08-27 — Initialize final provider storage once [patch, complete]

**Outcome:** let allocation-sensitive consumers initialize final
`MnemosyneStorage` directly without an intermediate `Vec` or default-fill pass.

**Scope:** the Mnemosyne storage constructor, Leto array construction boundary,
generic value-semantic and drop-safety tests, Rustdoc/CHANGELOG, and Apollo's
3-D GPU consumer. No allocator replacement, uninitialized public storage, or
GPU orchestration changes.

**Acceptance:** one panic-safe generic initializer writes each element exactly
once; `from_mnemosyne_shape_fn` uses it; values, zero length, non-Copy drops, and
generator panic cleanup are verified; strict provider gates and SemVer pass;
Apollo consumes the merged provider without an intermediate output allocation.

**Integrator:** Codex `01a0253c-6013-7552-99cc-36bbbcf77f6d`.
**Lease:** none. **Last update:** 2026-08-27.

**Evidence:** provider PR #127 merged as `fb70cb6`; `a3b7d44` initializes final
Mnemosyne storage once and makes partial construction panic-safe. Locked checks
and strict Clippy pass; Nextest passes 326/326; doctests pass 2/2; Rustdoc is
warning-clean; patch SemVer passes 223/223. Consumer PR #150 merged in Apollo as
`0536c9c8`, with its code revision passing 482/482 Nextest and public SemVer.
Focused Miri passes normal-drop, panic-cleanup, and zero-length cases; three
exposed-provenance warnings originate in the pinned Mnemosyne allocator.

## ATLAS-LETO-STACK-STORAGE-ORACLE-2026-08-20 — Assert valid stack construction [patch, complete]

**Finding:** `from_stack_validates_capacity` asserted only `is_ok()` for the
valid constructor result and only `is_err()` for the invalid result. Neither
branch proved the returned value or the typed failure contract.

**Outcome:** the valid branch now consumes the constructor result and checks
shape `[2, 2]`, size `4`, and all four inline values. The invalid branch
matches `LetoError::StorageError` and checks its exact capacity/shape reason.

**Evidence (2026-08-20):** clean lane branch `fix/leto-stack-storage-oracle`
is based on fetched `origin/main` `c1c8ab2`. Format, locked all-target check,
warning-denied Clippy, focused Nextest (`1/1`), and full Nextest (`984/984`)
pass. Doctests pass (`leto` `2/2`; `leto-ops` `20/20` plus one intentional
ignored case), and Rustdoc completes with 42 pre-existing `leto-ops` warnings;
the touched test adds none. The conformance scan reports
`existence_only_assertions: 7` versus `9` on the fetched default, with every
other class unchanged. A temporary constructor mutation that rejected valid
shapes failed the focused test (`0/1`), then the source was restored. The
provider commit and publication status are recorded below; the primary Leto
checkout remains untouched.

**Delivery:** commit `b682cd8` is published on
`fix/leto-stack-storage-oracle`. GitHub compare confirms exact base
`c1c8ab2` → head `b682cd8` is one commit ahead with four intended files.
Draft PR creation was rejected by the GitHub connector with HTTP 403
`Resource not accessible by integration`; no hosted gate or merge is claimed.
Re-open publication when repository write authorization is available.

## ATLAS-LETO-CONTRACT-100 — Make shutdown regression value-semantic [patch, complete]

**Owner:** Atlas session; scope is the Leto-ops parallel test assertion and
these provider-local PM records. No Moirai source, consumer, or release scope
is included.

**Finding:** the shutdown regression added by `508962d` asserts only
`ExecutorResult::is_err()`, so it cannot distinguish the required
`ExecutorError::ShuttingDown` contract from another failure.

**Acceptance:** assert the exact shutdown error variant, reduce Leto's
`existence_only_assertions` count from 10 to the committed baseline of 9, and
pass the focused locked provider gates without changing the baseline.

**Outcome:** provider commit `6463f4a` replaces the `is_err()` assertion with
`Err(moirai::ExecutorError::ShuttingDown)`. The provider scan returns 9;
formatting, strict Clippy, and focused locked Nextest pass 550/550. Hosted
exact-head CI `32021076930` and Pages `32021074899` pass. No baseline edit,
compatibility path, or Moirai change is required.

## ATLAS-ORPHAN-MODULES-096-LETO — Remove the uncompiled transform module [patch, complete]

**Owner:** Atlas session; scope is `crates/leto/src/application/transform.rs`
and the provider-local checklist entry. No `leto-ops`, consumer, or release
files are in scope.

**Finding:** the file is not reachable from any `mod` declaration. Its
`mapv`, `zip_map`, and `fill` methods duplicate compiled implementations in
`application/array.rs`; its additional `to_vec`, `fold`, and `mapv_inplace`
methods have no compiled callers because the file is not part of the crate.

**Acceptance:** delete the unreachable duplicate rather than wiring a second
implementation into the public surface; the source orphan count for Leto
falls by one, the package's value-semantic tests remain green, and the exact
Atlas orphan-module detector records the new count. If a missing public
contract is discovered during the audit, stop deletion and move the operation
to the canonical `application` leaf with tests and documentation in the same
change.

**Outcome:** `crates/leto/src/application/transform.rs` is deleted because it
had no module declaration or compiled callers and duplicated `Array` methods.
The direct detector reports `orphan_modules=0`; `git diff --check` passes.
From outside the Atlas configuration ancestry, the pinned MSVC toolchain
passes format, locked check, warning-denied Clippy, configured Nextest
`314/314`, two doctests, and rustdoc against the standalone lock graph. The
Atlas overlay still rejects the same `--locked` invocation before compilation
because it would rewrite the lockfile; no lockfile churn is committed.

## LETO-CROSS-ENTROPY-PROVIDER-1 — Own CPU classification loss [minor, arch, complete]

**Owner:** Codex on `codex/leto-cross-entropy-provider`; claimed 2026-08-04.

**Outcome:** own stable mean cross-entropy forward and additive backward over
borrowed Leto views and caller-owned outputs so CPU consumers retain no host
formula, staging allocation, or duplicate validation.

**Acceptance:** one scalar-generic contract validates rank, shape, targets,
storage reachability, and writable aliasing before mutation; contiguous and
strided f32/f64 value and failure-atomic contracts pass through Nextest;
warning-denied package gates, doctests, SemVer classification, independent
review, and exact-head CI pass before merge.

**Non-goals:** accelerator kernels, a Coeus adapter, reduction variants beyond
mean, class weights, ignored labels, or performance claims without matched
measurement.

**Evidence (2026-08-04):** ADR 0023 is accepted. Focused Nextest passes
14/14 analytical and failure-atomic contracts, including f32/f64 parity,
padded and permuted layouts, stable finite extremes, overflow-safe mean and
additive backward, gamma-bounded probability validation, and non-finite
inputs. Full `leto-ops` Nextest passes 511/511; all-target warning-denied
Clippy passes; doctests pass 18/18 runnable cases with one existing ignored
case; `cargo-semver-checks` passes 196/196 applicable minor-release checks.
Standalone locked all-feature metadata resolves after the dependency-ordered
Eunomia 0.8 sweep through Aequitas, Mnemosyne, and Hermes. Three falsification
passes found and closed tolerance, overflow, stride-oracle,
allocation, API-documentation, and additive-destination preflight defects.
Rustdoc adds no new diagnostic and retains the tracked baseline of 36 unrelated
broken/private-link warnings. Exact-head Rust verification passed and PR #94
merged as `c743a60`.

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

## LETO-MOIRAI-PACKAGE-1 — Restore Moirai resolution [patch, complete]

**Owner:** Codex on `release/leto-package-aliases`.

**Outcome:** bind the existing Rust crate alias to package `moirai-runtime`
0.4.0, refresh the lockfile, and pass focused checks before Hephaestus CI.

**Acceptance:** locked metadata and package archives resolve the provider's
published facade identity, focused package gates pass, and exact-head hosted
checks pass before merge.

**Status:** the exact external graph resolves `moirai-runtime`; format,
warning-denied Clippy, and 492/492 focused Nextest cases pass. PR #89 merged as
`a5d53ca` after exact-head hosted verification passed.

## LETO-MNEMOSYNE-PACKAGE-1 — Restore Mnemosyne resolution [patch, complete]

**Owner:** Codex on `codex/leto-themis-package`.

**Outcome:** bind the existing Rust crate alias to package `mnemosyne-memory`
0.6.0, refresh the lockfile, and pass focused checks before Hephaestus CI.

## LETO-THEMIS-PACKAGE-1 — Restore Themis resolution [patch, complete]

**Owner:** Codex on `codex/leto-themis-package`.

**Outcome:** bind the existing Rust crate alias to upstream package
`themis-topology` 0.10.1, refresh the lockfile, pass focused checks, and merge
before dependent Hephaestus provider CI is retried.

## LETO-STATEFUL-ZERO-LR-1 — Preserve warmup updates [patch, complete]

**Owner:** Codex on `codex/leto-pm-closeout`; PM closeout claimed 2026-08-13.

**Outcome:** every stateful-update parameter contract accepts a finite zero
learning rate while retaining strict positive epsilon and finite-domain checks.

**Acceptance:** all five rules construct at zero learning rate, negative and
non-finite rates remain rejected, focused Nextest and warning-denied Clippy
pass, and exact-head hosted checks pass before merge.

**Status:** complete. PR #86 merged as `7d8c98f`; Rust verification run
`30716401746` passed. The current default head `8c4e609` also passes the full CI
run `31645757949`.

## LETO-STATEFUL-UPDATE-1 — Provider-owned CPU updates [minor, arch, complete]

**Owner:** Codex on `codex/leto-pm-closeout`; PM closeout claimed 2026-08-13.

**Driver:** Coeus currently owns CPU optimizer formulas while Hephaestus owns
the corresponding accelerator seam. Leto must own the scalar-preserving CPU
contract so Coeus selects a provider instead of retaining consumer formulas.

**Scope:** one generic borrowed stateful-update entry point in `leto-ops`, SGD,
Adam, AdamW, RMSProp, and AdaGrad rule markers and validated parameters,
arbitrary injective const-rank layouts, value-semantic f32/f64 conformance,
ADR 0022, and synchronized release records. Coeus and Hephaestus changes are
separate dependency-ordered consumer increments.

**Acceptance:** every rule monomorphizes through the existing mutable-zip
provider with no tensor-sized allocation or copy; complete validation precedes
mutation; invalid parameters, shape/storage/layout contracts, and state
cardinality return typed errors; f32/f64 dense and strided values match an
independent oracle; warning-denied, Nextest, doctest, Rustdoc, SemVer, and
independent-review gates pass.

**Status:** complete. The provider API and f32/f64 value/failure contracts pass
full Leto/Leto Ops Nextest, warning-denied Clippy, doctest, Rustdoc, standalone
minimal-feature, SemVer, independent-review, and exact-head hosted gates.

## LETO-STABLE-VECTOR-NORM-1 — Range-stable Euclidean geometry [patch, complete]

**Owner:** Codex on `codex/leto-pm-closeout`; PM closeout claimed 2026-08-13.

**Driver:** Gaia polyline arc length delegates to `Vector::norm`, whose direct
sum of squares overflows or underflows for finite vectors with representable
Euclidean lengths. RITK tractography therefore cannot use the provider contract
without losing valid physical path lengths.

**Scope:** Leto Euclidean vector norm and its f32/f64 value-semantic tests.
Other linear-algebra norm families and the active `leto-ops` changes are
non-goals.

**Acceptance:** `Vector::norm` and `distance` preserve finite representable
large and subnormal-scale lengths without widening precision; ordinary and
non-finite IEEE behavior remains explicit; format, focused check, warning-denied
Clippy, Nextest, doctest, Rustdoc, and SemVer gates pass.

**Status:** complete. PR #84 merged as `aa5c283`; the current default head
`8c4e609` passes CI run `31645757949`, including the range-stable norm
implementation and regression suite.

## LETO-MUTABLE-ZIP-PROVIDER-1 — Generalize mutable zip outputs [major, arch, complete]

**Owner:** Codex; claimed 2026-07-31.

**Driver:** Kwavers still carries provider-neutral mutable-output zip traits and
indexed traversal wrappers even though `leto-ops` owns the array-view layout,
source tuple, and traversal contracts. Move output arity into one sealed
monomorphized provider trait so consumers use `zip_mut_with` and
`indexed_zip_mut_with` directly.

**Scope:** `leto-ops` mutable zip output trait and tests, its public exports and
ADR, plus the Kwavers consumers and superseded local zip wrappers. Moirai's
execution primitives remain outside the array-layout contract; no new executor
abstraction or domain-kernel migration is in scope.

**Acceptance:** one provider API covers one, two, and three mutable outputs,
zero or more statically typed read-only sources, dense and strided layouts, and
indexed/non-indexed traversal; Kwavers has no local `MutableZipOutputs`, zip
wrapper, or indexed source wrapper residue; value-semantic provider and
consumer tests, format, focused checks, and Nextest pass.

**Evidence:** `leto-ops` check, warning-denied clippy, format, and diff checks
pass; provider Nextest passes 37/37 and provider doctests pass 16/16. Kwavers
consumer package compilation and affected package Nextest lanes pass. No
runtime or benchmark gain is claimed without controlled measurements.

## LETO-SPARSE-OWNED-FACTOR-1 — Owned sparse LU factor for preconditioner caching [minor, complete]

**Owner:** Claude (atlas session 0161539d); last-update: 2026-07-31.

**Driver:** CFDrs `cfd-math` commit 63e49604 landed `ComponentBlockPreconditioner`
against a leto surface that was never built (`OwnedNumericLu`,
`SparseLuSolver::factor_sparse_with_symbolic`), leaving the CFDrs workspace
uncompilable. Upstream ownership: the capability lands here, not downstream.

**Scope:** `leto-ops` sparse module only (`lu_numeric.rs`, `lu_sparse.rs`,
`sparse/mod.rs`) — disjoint from LETO-ATTENTION-PROVIDER-1's claimed files.
`OwnedNumericLu<T>` (owned symbolic + values, dense `LuDecomposition` arm via
the solver's existing dispatch/pivot-fallback contract), `NumericLu::solve_into`,
shared `triangular_solve_into` core (SSOT), and a partial-pivot-scan magnitude
fix in `factor_numeric` (raw-value comparison misreported negative-pivot
columns as singular). Root `lib.rs` re-export sweep completed 2026-08-02 once
the attention item released the file: the root sparse export list now carries
the full LU family (`factor_numeric`, `factor_symbolic`, `NumericLu`,
`OwnedNumericLu`, `SymbolicLu`), and CFDrs's interim
`leto_ops::application::sparse` deep-path import reverts to the root in the
same co-evolution unit.

**Acceptance:** cfd-math compiles against local leto; owned factor
value-matches `solve_view` differentially; pivot-requiring and small matrices
route through the dense arm; shape mismatches are typed errors; full leto-ops
Nextest + doctests green.

## LETO-ATTENTION-PROVIDER-1 — Scaled dot-product attention [major, arch, complete]

**Owner:** Codex on `codex/leto-attention-provider`; last-update: 2026-07-31.

**Driver:** Coeus must dispatch CPU attention through Leto and accelerator
attention through Hephaestus without consumer-owned kernels, host fallback, or
erased provider errors.

**Scope:** one scalar-preserving rank-3 scaled dot-product attention family in
`leto-ops`; borrowed strided query/key/value/mask views; caller-owned output and
weights; additive optional query/key/value gradient targets; typed validation;
generic f32/f64 contracts; ADR 0002, exports, Rustdoc, changelog, and active PM
artifacts. Accelerator kernels and the downstream Coeus cutover are non-goals.

**Acceptance:** forward and backward validate every operand before mutation;
unmasked, causal, broadcast-mask, and causal-plus-mask modes preserve value
semantics for contiguous and strided views; prefilled selected gradients are
accumulated rather than overwritten; failures are typed and atomic; focused
format, check, warning-denied Clippy, Nextest, doctest, Rustdoc, and SemVer
classification pass.

**Risk/change class:** `[major][arch]`; revises ADR 0002's CPU-kernel ownership
boundary while preserving its const-rank and monomorphized dispatch decision.

**Evidence (2026-07-31):** focused attention Nextest 15/15 and full `leto-ops`
Nextest 469/469 pass; f32/f64 closed-form forward, negative and positive
strides, exact injective/aliased mutable-layout validation, broadcast/causal
masks, fully masked rows, finite-difference backward,
optional additive targets, non-finite input, and typed failure atomicity are
covered. Format, all-target warning-denied Clippy, and 16/16 runnable doctests
pass. `cargo-semver-checks` reports 196/196 applicable minor-release checks
passing against `origin/main`. Warning-denied Rustdoc reaches the unchanged
repository baseline of 36 unrelated broken/private-link diagnostics; the new
attention Rustdoc produces none. Three independent falsification passes found
and closed arithmetic overflow, probability validation, optional-workspace,
mask-aware preflight, and extreme finite-output defects. Delivered by PR #82.

## LETO-GENERIC-ZIP-SOURCES-1 — Generic tuple source sets for multi-input zips [major, arch, complete]

**Owner:** Codex on `codex/leto-real-sparse-lu`

**Scope:** Replace the duplicated `leto-ops` multi-input zip implementations
with one sealed tuple-source contract, migrate all in-repo callers and tests,
and synchronize ADR, changelog, and public re-exports. Non-goals: changing the
optimized single-source zip contract or claiming a runtime improvement without
matched measurements.

**Acceptance:** Multi-input zips use one monomorphized traversal for dense and
strided layouts; tuple element types remain heterogeneous and statically
dispatched; indexed and non-indexed paths preserve value semantics; removed
arity-specific names have no live residue; format, check, Nextest, doctest,
Rustdoc, and warning-denied Clippy evidence is recorded.

**Evidence:** `cargo check -p leto-ops --all-features`, format, and warning-denied
all-target Clippy pass. The configured `cargo nextest run -p leto-ops
--all-features` lane passes 448/448, including two-, three-, four-, and
five-source tuple cases, heterogeneous source types, strided views, indexed
traversal, and finite-difference callers. `cargo test --doc -p leto-ops
--all-features` passes 13/13 runnable doctests; `cargo doc -p leto-ops
--all-features --no-deps` builds. Warning-denied Rustdoc remains blocked by the
36 pre-existing unrelated broken/private links listed by that command. The
semver audit reports one expected major failure for the removed arity-specific
functions. `cargo check --locked` remains blocked by the pre-existing stale
lockfile versus the active stack patch overlay; this change does not modify
`Cargo.lock`.

**Delivery watchpoint:** PR [#81](https://github.com/ryancinsight/leto/pull/81)
is pushed and ready for review. The repository exposes no Actions workflow for
this branch; merge state is currently `UNSTABLE` because the external
`recurseml/analysis` status reports `Error occurred during analysis
(4c584e3a..4a836cd8)`, while CodeRabbit review remains pending.

## LETO-CONVOLUTION-PROVIDER-1 — Generic convolution contracts [major, arch, complete]

**Owner:** Atlas integration; implementation delivered through PRs #78, #79,
and #80.

**Driver:** Coeus ADR-0046 requires CPU backend selection to execute directly
through Leto before Coeus can delete its transposed-convolution host default
and make `ConvOps` fallible.

**Scope:** one generic, scalar-preserving regular and transposed-convolution
operation family under `leto-ops`; validating shape/stride/padding/dilation
contracts; allocation-reusing output APIs; generic conformance and analytical
value tests; Rustdoc, changelog, gap audit, and this item. Accelerator kernels,
Coeus call-site migration, and unrelated linear algebra are non-goals.

**Acceptance:** 1-D, 2-D, and 3-D regular forward/backward plus 1-D, 2-D, and 3-D
transposed forward/backward operations execute through one dimension-parameterized
implementation family; invalid contracts return typed errors without partial
output mutation; every supported scalar instantiates the same value-semantic
suite; format, warning-denied Clippy, configured Nextest, doctests, Rustdoc,
and SemVer classification pass.

**Claimed files:** `crates/leto-ops/src/application/convolution/`,
`crates/leto-ops/src/{application/mod.rs,lib.rs}`, focused `leto-ops` tests,
`CHANGELOG.md`, `gap_audit.md`, `backlog.md`, and the governing ADR/index.

**Current evidence:** the allocation-reusing N-dimensional regular and
transposed forward/additive-backward kernels validate ranks, shapes, storage,
all requested gradient targets, and checked dimension arithmetic before
mutation. Generic f32/f64/F16/Bf16 conformance; 1-D, 2-D, and 3-D transposed
semantics; output-padding gradient behavior; exact typed errors; and failure
atomicity, including strided input/weight/output/gradient views, pass under
focused Nextest (18/18). Package test targets compile warning-free on Rust
1.97 GNU; doctests pass 11/11 with one existing ignored case; and all 196
applicable minor-release SemVer checks pass. The exact current provider default
is `e525d8dd5ee52d12de0bf61987e8af6bf896700f`: PR #78 merged as `4137a1c`,
PR #79 as `aa958be`, and PR #80 as `f896c43`. Hosted run `31663241086` passes
formatting, minimal-feature compilation, warning-denied Clippy, configured
Nextest, doctests, and documentation at that exact head. Local Rustdoc retains
33 pre-existing broken/private-link warnings; the convolution family adds no
new diagnostic. ADR 0019 records the provider contract.

Coeus consumer integration is complete independently under
`ATLAS-COEUS-CONVOLUTION-020`: current Coeus default
`aabdec67a0f5baa415c4abb6dded69db41b2f2d6` routes CPU convolution directly
through Leto, routes accelerators through Hephaestus, and deletes the former
host kernels and autograd loops. Coeus hosted run `31672329963` passes at that
exact head. The provider item is therefore complete; accelerator execution
remains owned by Hephaestus and is outside this CPU provider item.

## LETO-COMPARISON-OPS-1 — Broadcast comparison markers [minor, complete]

**Owner:** Codex

**Scope:** `leto-ops` sealed `BinaryOp` markers for equality and ordering
comparisons, root re-exports, and broadcast elementwise regression coverage.

**Acceptance:** `EqOp`, `NeOp`, `LtOp`, `GtOp`, `LeOp`, and `GeOp` produce
zero-or-one masks through the existing broadcast-aware `binary_map` traversal
for every `Scalar` implementation; downstream consumers can use the public
markers without defining local operation adapters.

**Evidence:** `cargo check --locked -p leto-ops`, focused Nextest
`elementwise` (19/19), warning-denied all-target Clippy, and `cargo test
--locked -p leto-ops --doc` (10/10) pass. `cargo doc --no-deps` completes with
18 pre-existing unrelated link warnings.

## LETO-SPARSE-LU-VIEW-1 — Preserve native RHS views at the sparse-LU seam [minor, complete]

**Owner:** Codex `/root`

**Scope:** `leto-ops` sparse direct-solver input-view API, its value-semantic
regression coverage, Rustdoc, and the downstream `CFDrs` direct-solver call
site. Non-goals: changing the dense-backed LU algorithm, sparse matrix
storage, pivot policy, or the legacy slice-returning API.

**Acceptance:** the provider exposes one `ArrayView1`-based solve path that
validates the existing size and shape contracts, the `CFDrs` consumer passes
its native `Array1` view without an intermediate RHS or solution `Vec`, and
provider plus consumer tests preserve exact value semantics for supported
`f32`/`f64` solves. Warning-denied format, check, Clippy, configured Nextest,
doctest, and Rustdoc gates pass on the exact revisions. No compatibility
adapter or concrete-precision duplicate is introduced.

**Claimed files:** `crates/leto-ops/src/application/sparse/lu_sparse.rs`,
`crates/leto-ops/src/application/sparse/mod.rs`, and the corresponding
`CFDrs` direct-solver source, tests, and active PM entries.

**Current evidence:** provider check, warning-denied all-target Clippy, focused
sparse Nextest (29/29), doctests (8/8), Rustdoc, and public-surface SemVer
classification (196/196 checks, 57 skipped) pass. Consumer integration passes
its focused check, lib Clippy, direct-solver Nextest (4/4), doctest, and
Rustdoc gates against merged Leto `b24fc860864abad84af3118aa2bb27c32bb81265`.
Leto PR #70 and CFDrs PR #309 are merged; Atlas pins both merged child heads.

## LETO-PARITY-HARNESS-1 — Runnable migration evidence [patch, complete]

**Owner:** Codex `/root/implement_horae` (stale-claim takeover at
2026-07-22 13:26 ET)

**Scope:** `leto-ops` ndarray/nalgebra parity examples, their oracle-only
dev-dependencies, README and completeness evidence, and owner-local checklist
state. Production kernels and dependency ownership are non-goals.

**Acceptance:** both examples run deterministic, value-semantic differential
checks with explicit error magnitudes and analytically scaled bounds; claims
match the APIs actually exercised; single-shot timings and nonexistent SSOT
references are absent; format, warning-denied Clippy, configured Nextest,
doctest, Rustdoc, and example execution gates pass; available hosted review
checks pass (this repository defines no PR test workflow); the branch merges
and leaves one clean `main` worktree.

**Current evidence:** both executables pass against ndarray 0.16.1 and nalgebra
0.35.0 and report eleven bounded observations. Focused example Nextest passes
7/7; warning-denied all-target/all-feature Clippy passes; configured
all-target/all-feature Nextest passes 688/688; doctests pass 8/8;
warning-denied Rustdoc passes. The normal dependency graph contains neither
ndarray nor nalgebra, while the dev graph contains the intended oracle edges.
Greptile's single P2 version-alignment finding was fixed and resolved on PR
#69; no repository PR test workflow exists.

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

## LETO-NDARRAY-BOUNDARY-1 — Retire public ndarray compatibility [major, done]

**Owner:** Codex `/root`

**Scope:** remove the public `ndarray-compat` feature, conversion module,
conversion-only tests, and downstream feature requests; retain `ndarray` only
as a test/benchmark oracle. Correct the pre-existing `Tiles` Rustdoc link that
blocks the package documentation gate. Non-goals: removing independent
differential oracles or changing Leto array semantics.

**Acceptance:** production manifests and Rust sources contain no `ndarray`
dependency or conversion surface; all retained array, view, stride, mutation,
and storage contracts pass their canonical suites; the breaking boundary and
consumer migration are documented in ADR 0017; Apollo consumes native Leto
without the removed feature; format, warning-denied Clippy, configured Nextest,
doctest, Rustdoc, dependency, and SemVer gates pass.

**Current evidence:** Leto 0.40.0 releases the removed compatibility surface;
the configured provider gates and the expected major SemVer classification pass.
Apollo commit `324f380` consumes native Leto arrays, and its manifests and
resolved Rust graph contain no `ndarray` or `ndarray-compat` dependency edge.

## LETO-LAPLACIAN-1 — Typed Cartesian stencil ownership [minor, done]

**Owner:** Codex `/root`

**Scope:** `leto` typed grid, boundary, spacing, and polarity contracts;
`leto-ops` CPU evaluation; dependency lock; tests; ADR and PM artifacts.

**Acceptance:** one native-precision CPU implementation covers `f32` and `f64`;
the typed contract is reusable by Hephaestus; invalid dimensions, spacing, and
array lengths return typed failures; the anisotropic Neumann quadratic matches
its closed form; focused package gates pass. **Evidence:** all-target/all-feature
check and warning-denied Clippy; configured Nextest 575/575; doctests 9/9; and
warning-denied rustdoc.

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

## LETO-PARALLEL-INTENSITY-1 — Arithmetic-intensity-aware parallel thresholds [minor, done]

**Owner:** unclaimed

**Scope:** `leto-ops` `PARALLEL_THRESHOLD` gating in `map.rs` (binary), `unary.rs`,
and `reduction.rs`. Replace the uniform element-count gate with a per-op-intensity,
cache-aware threshold: bandwidth-bound elementwise ops parallelize only when the
working set exceeds shared LLC (`themis::CpuTopology`); compute-bound ops keep a
low threshold. See gap_audit `2026-07-19 Parallel Threshold Ignores Arithmetic
Intensity`.

**Done:**
- Binary (`map.rs` — `add`/`sub`/`mul`/`div`): gates on
  `working_set > CacheGeometry::l3_bytes()` via `cached_cache_geometry()` (L3 from
  `themis`). Diagnosed 64k `f64` `add` 43 µs → 16 µs.
- Unary + scalar-broadcast into caller-owned output (`unary_map_into`,
  `scalar_map_into`): `UnaryOp::COMPUTE_BOUND` const marks transcendentals eager
  and `neg`/`abs` bandwidth-bound; the shared `map_into_gated` gates bandwidth-
  bound ops on working-set-vs-LLC. 64k `f64` `scalar_map_into` add ~73 µs → 9.4 µs.
  Raw-closure `map_into`/`mapv` keep the eager default (closure intensity is
  unknowable); the serial `mapv` path is unaffected.
- Reductions: measured competitive (`sum` @64k serial 3.3 µs ≈ parallel 3.6 µs) —
  the efficient parallel tree-reduction does not over-parallelize, so the existing
  threshold is kept (no change needed). 305/305 tests green throughout.

- **f32 coverage (2026-09-01).** The sweep below validated f64 only, and the
  committed `parallel_crossover` bench instantiated f64 only — a generic
  instantiation gap in the instrument: the gate scales by `size_of::<T>()`, so
  one scalar validates one row of its model. A same-binary three-arm probe
  (gate / sequential slice loop / naive scoped-thread add; best of 9) confirms
  the model at f32 as well. Naive-parallel over sequential: f64 1.29x at 1M,
  **0.55x at 1.57M** (the model's f64 crossover); f32 2.84x at 1M, 1.69x at
  1.57M, 1.06x at 2.1M, **0.63x at 3.1M** (the model's f32 crossover). The gate
  is at most ~1.3x conservative for f32, not the 3x an element-driven model
  would imply. The bench is now generic over the scalar and runs both rows.
- Crossover sweep (`parallel_crossover` bench, `add` 512k → 8M, gate vs
  `--no-default` serial) validates the L3-working-set threshold on a 36 MiB-L3
  host: below L3 the gate is serial and matches the serial baseline (512k 942 vs
  897 µs; 1M 1709 vs 1596 µs — within noise), above L3 it parallelizes and wins
  (2M 1447 vs 2579 µs = 1.78×; 4M 3686 vs 4645 µs = 1.26×; 8M 5380 vs 9267 µs =
  1.72×), with non-overlapping CIs. The crossover lands at the L3 boundary — 2M
  parallel (1447 µs) even beats 1M serial (1709 µs) despite 2× the data. The
  cache-residency default is confirmed optimal; no threshold refinement needed.

**Remaining:** none for the threshold policy. A latent option (not required): flip
the raw-closure `map_into`/`mapv` default to bandwidth-bound — currently eager to
avoid regressing rare compute-bound closures, which should use typed ops anyway.

**Acceptance:** met for the converted set — binary, `unary_map_into`,
`scalar_map_into`, and reductions gate on cache residency; the sweep confirms
parallel wins above L3 and is correctly avoided below; no compute-bound
regression; full warning-denied gate green.

**Correction (2026-09-01):** this previously read "every bandwidth-bound
elementwise path", which overstated the converted set. `map_inplace` was not
converted and still gates on the flat element count; measured at up to 9.8x
slower for a bandwidth-bound closure while cache-resident. Carried by
`LETO-INPLACE-INTENSITY-GATE-2026-09-01` above; "Remaining: none" above applies
to the converted set only.

## LETO-EUNOMIA-PRECISION-1 — Reduced-precision ownership [major, done]

**Owner:** Codex `/root`

**Scope:** Leto/Leto Ops scalar contracts, arithmetic markers, reduced-precision
tests, direct dependency manifests, provider lock, synchronized PM artifacts,
and composition of the peer-owned matrix-trait, oracle-parity, and Schur
rustfmt-only edits.

**Acceptance:** production and test Rust sources contain no raw `half` types;
manifests contain no direct `half` dependency; Leto exposes Eunomia
`F16`/`Bf16` natively through its scalar and real contracts; the lock resolves
merged Eunomia and Hermes defaults; format, warning-denied all-target/all-feature
Clippy, configured Nextest, doctests, rustdoc, no-default-feature compilation,
residue audits, and Rust-crate semver classification pass; any Python semver
tooling failure is isolated with exact evidence. Delivered by PR #46 as merge
commit `0afece5`.

## LETO-EUNOMIA-0.4-REFRESH — Provider lock [patch, done]

**Owner:** Codex `/root`

**Scope:** Eunomia reproducibility pin and synchronized provider evidence.

**Acceptance:** the lock resolves Eunomia 0.4.0 from its merged default commit;
Leto's full warning-denied compile, test, doctest, and rustdoc gates pass.

**Evidence:** `Cargo.lock` resolves `49dc115`; formatter, warning-denied
all-target/all-feature Clippy, configured Nextest 593/593, doctests 9/9, and
warning-denied rustdoc pass.

## LETO-EUNOMIA-COMPLEX-1 — Complex oracle ownership [patch, done]

**Owner:** Codex `/root`

**Scope:** workspace numeric dependency ownership, `leto-ops` migration and
decomposition oracles, dependency lock, and synchronized PM artifacts.

**Acceptance:** no Leto manifest or Rust source directly references
`num-complex`/`num_complex`; tests use Eunomia's complex representation
natively; full affected package gates pass.

**Evidence:** direct manifest/source and production graph residue are zero;
warning-denied all-target/all-feature Clippy; Nextest 305/305; doctest 8/8;
warning-denied rustdoc; 196/196 applicable SemVer checks. Merged as PR #42
(`cf47686`).

## LETO-EXTERNAL-ORACLE-1 — Reconcile legacy oracle ownership [arch, done]

**Owner:** Codex `/root` (complete)

**Claimed files:** `backlog.md`, `checklist.md`, `gap_audit.md`,
`crates/leto-ops/Cargo.toml`, and the current oracle import sites.

**Scope:** reconcile the stale removal plan against the current `leto-ops`
test, benchmark, example, manifest, and normal dependency graph. Production
graph ownership and the retained independent oracle role are in scope;
deleting active evidence is not.

**Acceptance:** prove that `ndarray`, `ndarray-rand`, and `nalgebra` do not enter
the normal dependency graph; enumerate their active dev-only tests, examples,
and benchmark roles; retain independent value-semantic evidence where no
analytical replacement exists; and update this item so it no longer claims
that active oracle code is obsolete. No compatibility wrapper or production
dependency is introduced.

**Evidence:** the normal `leto-ops` graph has no `ndarray`, `ndarray-rand`, or
`nalgebra` edge. The dev graph resolves `ndarray 0.16.1`, `ndarray-rand 0.15.0`,
and `nalgebra 0.35.0`. Active references are limited to seven source files:
the `kernels.rs` benchmark, `ndarray_parity.rs`, `nalgebra_parity.rs`, and
four differential/parity test modules. The previous removal target was stale;
the active independent oracle boundary is intentional and remains dev-only.

## LETO-SPARSE-DUPLICATE-1 — Sparse conversion contract [patch, done]

**Owner:** Codex `/root`

**Scope:** stale shared-tree sparse conversion/test changes, package formatting,
dependency lock refresh, and synchronized PM artifacts.

**Acceptance:** duplicate COO coordinates sum or keep the last value exactly;
zero sums are removed; CSC construction normalizes arbitrary COO order;
transpose and column access preserve exact values; package diagnostics and
configured tests pass without warnings.

**Evidence:** sparse Nextest 18/18; full Leto Nextest 267/267; warning-denied
all-target/all-feature Clippy; doctest 1/1; warning-denied rustdoc; 196/196
applicable SemVer checks; value-semantic unordered duplicate, CSC column,
lookup, and transpose regressions. Merged as PR #41 (`9b22301`).

## LETO-SPARSE-DIRECT-1 — Sparse direct factorization [minor, done 2026-08-16]

**Owner:** unclaimed

**Consumer:** CFDrs `DirectSparseSolver`

**Scope:** `leto-ops` CSR factorization, solve, errors, tests, Rustdoc, and
consumer contract verification.

**DELIVERED 2026-08-16:** the native sparse LU (`SparseLuSolver`,
`factor_symbolic`/`factor_numeric`, AMD ordering) shipped in leto-ops with
generic f32/f64 value tests, typed failures, and the symbolic-phase storage
contracts covered by the lu_symbolic unit suite (PR #114, `a65fcfb`).
CFDrs's `DirectSparseSolver` consumes `leto_ops::SparseLuSolver` (ADR 0031)
with a dense fallback; `rsparse` and all old call sites were removed. All
acceptance bullets are met; the item is closed.


Add a generic sparse direct factorization over Leto-owned `CsrMatrix<T>`.
The implementation must preserve a failure mode independent from iterative
Krylov solvers: it may not call CG/GMRES, materialize the full coefficient
matrix as dense storage, or silently fall back to another solver. Before code
lands, a co-located design note must ground the ordering, pivoting, symbolic
analysis, and numeric-factorization contracts in an authoritative reference.

Acceptance:

- One native-precision entry point generic over every supported real scalar,
  with typed structural, non-finite, singular, and solve-dimension failures.
- Symbolic analysis and reusable numeric factors have explicit ownership and
  allocation contracts; repeated right-hand sides reuse the factorization.
- Generic `f32`/`f64` value tests cover nonsymmetric pivoting, singular input,
  repeated solves, and residual bounds derived from scalar epsilon and matrix
  conditioning.
- Differential conformance covers the independent retained CFDrs sparse-LU
  provider and Leto's dense partial-pivot LU on small matrices; the CFDrs
  direct-after-GMRES consumer regression passes before `rsparse` is removed.
- Provider fmt, warning-denied Clippy, configured Nextest, doctest, rustdoc,
  and SemVer gates pass; CFDrs removes `rsparse` and all old call sites in the
  same consumer increment, without an adapter or iterative fallback.

## Provider default-source convergence (DELIVERED 2026-07-16)

[minor] Leto 0.37.0 follows merged provider default branches for Mnemosyne,
Moirai, Hermes, Eunomia, and Themis. The lockfile remains the reproducibility
pin. Mnemosyne 0.5/Core 0.2 requires the declared Rust 1.95 MSRV across every
published workspace package. Formatter, explicit-nightly warning-denied release
Clippy, 568/568 release Nextest, 9/9 doctests, rustdoc, source-identity scan,
and offline rustdoc SemVer checks for `leto` and `leto-ops` pass. Hephaestus and
Apollo now refresh their locks against the merged Leto source contract.

## CR-4 SSOT rebind: `leto_ops::Scalar` over `eunomia::NumericElement` (DONE 2026-07-05)

[minor] Leto `leto_ops::Scalar` is now bound as `pub trait Scalar: NumericElement { fn from_usize(...) -> Self; /* default-bodied slice kernels */ }` per `atlas/docs/adr/0005-eunomia-scalar-ssot.md`. The local maintainer branch was rebased onto `origin/main` (PR #30 feat/array-to-vec, 47 commits ahead), resolving file-level merge conflicts at `crates/leto-ops/src/domain/scalar.rs`, `crates/leto/src/lib.rs`, `crates/leto/src/application/array.rs`, and `crates/leto-ops/src/application/sparse/mod.rs`. The old standalone `Scalar` trait methods (`ZERO/ONE/add/sub/mul/div/bitand/bitor/bitxor/count_ones/to_f64`) are inherited from `NumericElement`; `RealScalar` inherits from `FloatElement`. Leto keeps only `from_usize` and default-bodied slice kernels. No compatibility shims.

- Downstream fallout remains consumer-owned: Apollo/Coeus code that explicitly names removed Leto UFCS items should import Eunomia traits directly.
- Evidence: `rustup run nightly cargo check -p leto-ops --all-features`; `rustup run nightly cargo fmt --package leto-ops --check`; `rustup run nightly cargo clippy -p leto-ops --all-targets --all-features -- -D warnings`; `rustup run nightly cargo nextest run -p leto-ops --all-features` (271/271 tests) pass. Clippy also reports the pre-existing upstream `hermes-simd-core::sparse::ValidatedData::new_unchecked` dead-code warning while exiting successfully for the `leto-ops` gate.

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

## Replacement Position
- [x] [arch] Use `leto` as the Atlas shared N-dimensional strided-array and layout crate. It sits below Apollo and Coeus and above Mnemosyne/Moirai/Hermes. It should replace `ndarray` only after parity and verification gates are met.
- [x] [patch] Naming assessment: `leto` is appropriate. The crate's intended responsibility is the shared array substrate between Coeus and Apollo, matching both functionality and the existing mythological naming scheme. Rename only if the crate changes scope into autodiff/tensors proper or Apollo-specific signal arrays.

## Current Evidence
- [x] [patch] ndarray/nalgebra oracle gates added for current Leto replacement
  claims. `leto-ops` oracle tests compare LU solve/determinant/inverse,
  symmetric eigenvalues, Cholesky lower factors, singular values, and
  reverse-last-axis reductions against nalgebra/ndarray. Criterion oracle
  comparison shows reverse reductions at parity or faster than ndarray, while
  dense 128x128 matmul is slower than ndarray/nalgebra and remains open.
  **Superseded 2026-08-28** — see `LETO-MATMUL-PARITY-VERDICT-2026-08-28`.
- [x] [patch] `cargo test --all-features` passes: 34 `leto` core tests, 28 `leto-ops` tests, and 5 `leto-python` tests pass. Evidence tier: value-semantic, property, differential, PyO3 boundary, and downstream-shape migration fixture tests.
- [x] [patch] The 2026-06-10 Apollo scan identified its public and internal
  `ndarray` usage; the completed migration replaces those array, shape, mapping,
  and Python ownership boundaries with native Leto arrays at commit `324f380`.
- [x] [patch] `cargo fmt --check` is clean after formatting the workspace.
- [x] [patch] `cargo clippy --all-targets --all-features -- -D warnings` is clean after fixing `mnemosyne-alloc` allocator use and public module docs.
- [x] [patch] `cargo test --all-features` is clean.
- [x] [patch] `CowStorage` is available for Leto arrays that borrow read-only Apollo/Coeus inputs and clone only when mutable access is requested. Evidence tier: value-semantic tests assert pointer identity on read-only borrowed storage, source preservation after mutation, and owned-detach output values.
- [x] [patch] Full `cargo doc --workspace --all-features --no-deps` no longer
  hits the `leto-python`/`numpy-0.23.0` rustdoc ICE. `leto-python` is a PyO3
  extension boundary with no public Rust API, so its library target has
  `doc = false`; Cargo still checks and tests the Rust crate, but rustdoc no
  longer walks NumPy 0.23's broken intra-doc link path. Verification:
  `cargo doc -p leto-python --all-features --no-deps`;
  `cargo doc --workspace --all-features --no-deps`;
  `cargo doc --no-deps`;
  `cargo clippy -p leto-python --all-targets --all-features -- -D warnings`;
  `cargo nextest run -p leto-python --all-features` (21 tests).

## Phase 1: Sound Core Layout and Storage [patch]
- [x] Add ndarray-style slicing for full-axis selection, optional signed range bounds, negative indices, negative steps, integer axis removal, new-axis insertion, ellipsis expansion, and implicit trailing axes. Verification: three value-semantic tests over rank-preserving, rank-dropping, rank-adding, reverse, ellipsis, and implicit-tail cases.
- [x] Replace unchecked negative-offset casts with checked signed arithmetic across `Layout` and `Array` validation. Verification: value-semantic tests cover valid negative strides, rejected negative physical offsets, and one-past-storage rejection.
- [x] Make externally constructed `ArrayView` and `ArrayViewMut` layouts bounds-checked against their backing slices through `try_new` constructors. Verification: invalid external layouts return `StorageError`.
- [x] Add copy-on-write storage for zero-copy read-only interop and mutation-time detachment. Verification: core tests cover borrowed pointer identity, owned-detach transition, unchanged source backing, and mutated owned values.
- [x] Remove or constrain mutable broadcast views that introduce zero-stride write aliasing. Verification: mutable broadcast rejects aliasing expansion and permits same-shape non-aliasing writes.
- [x] Add overflow-checked shape product and stride multiplication for core constructors and derived layout validation. Verification: property tests cover bounded generated offset, empty-axis, negative-stride, and composed-slice cases.
- [x] Add property tests for C/F layouts, negative strides, singleton axes, transposes, slices, broadcasts, and offset ranges. Verification: generated tests cover C/F offset formulas, transpose value preservation, reverse slicing, composed slicing, empty-axis storage validation, singleton-axis broadcast stride/value contracts, and negative-stride storage span validation. Remaining risk: broad adversarial composition over larger dimensions still needs expansion.
- [x] Fix `MnemosyneStorage` initialization semantics. `new(len)` requires `T: Default` and initializes elements; `from_slice` copies initialized values; `Drop` drops elements before deallocation.
- [x] Add Mnemosyne-backed owned array constructors for Apollo replacement boundaries. `zeros_mnemosyne` and `from_mnemosyne_slice` construct C-contiguous Leto arrays over `MnemosyneStorage`, with ndarray differential validation for shape, strides, values, and length rejection.
- [x] Add Apollo ndarray-validation contract tests. Coverage validates Leto constructor, storage, transpose, broadcast, axis iteration, mutable view, slice metadata, ndarray conversion, negative-stride import, and bounds-rejection behavior against `ndarray`.
- [x] Align retained single-element range stride metadata with `ndarray`: `SliceArg::range` outputs stride `0` when the normalized range length is exactly one, while empty ranges preserve their computed stride.

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

## Phase 8: nalgebra Successor Policy [minor]
Source: `gap_audit.md` §B. Apollo's nalgebra removal is complete; this phase is demand-driven.
- [x] [minor] Generalize `symmetric_eigen_jacobi`/`SymmetricEigenDecomposition` over `T: RealScalar`; runs in native precision with no hidden widening (the wider-accumulator path is intentionally not introduced — a consumer needing higher working precision converts first). f32 genericity test added; f64 path unchanged. `RealScalar` is a segregated transcendental extension of `Scalar` (ISP).
- [x] [minor] Add eigenvalues-only symmetric Jacobi entry points so callers do not allocate eigenvectors when only sorted eigenvalues are needed. The implementation uses a `RotationTarget` strategy with a zero-sized no-vector target; no `dyn` dispatch and no fake generic widening.
- [x] [minor] LU/solve/det/inv, QR + least squares, Cholesky, and norms entered `leto-ops` with named CFDrs consumer drivers and nalgebra differential oracles.
- [x] [major] Full rank-revealing SVD/pseudoinverse delivered by ADR 0005 through one-sided Jacobi SVD and Moore-Penrose construction; the legacy Gram SVD remains full-rank-only by contract.
- [x] [minor] Non-symmetric eigenvalues delivered by ADR 0006 Phase 2a through shifted complex QR. Remaining [major] surface: Schur vectors (`Q`, quasi-triangular `T`) when a consumer needs them.
- [x] [minor] Add unpivoted symmetric indefinite `U D Uᵀ` decomposition with determinant/solve/inverse helpers. Remaining [major] surface: pivoted symmetric-indefinite factorization for zero-pivot cases.

## Apollo Migration Gate [arch]
- [x] Add Leto as a Git workspace dependency in Apollo only after a pushed Leto revision passes all default and all-feature gates. The initial Apollo boundary requested `["std", "ndarray-compat"]`; current Apollo consumes native Leto arrays without that retired feature and exposes Leto boundaries across its transform families.
- [x] [minor] Replace Apollo's nalgebra dependency: FrFT/GFT eigendecomposition migrated to `leto_ops::symmetric_eigen_jacobi`; GFT adjacency storage migrated to `leto::Array2<f64>`.
- [x] Add representative Leto-side Apollo and Coeus migration fixtures before direct consumer updates. Verification: fixtures cover Apollo FFT-like rank/complex/precision shapes and Coeus reduction/broadcast/matmul shapes.
- [x] Migrate Apollo transform crates to native Leto arrays with consumer-side
  value-semantic and differential coverage.
- [x] Record Apollo's public array-boundary migration in its changelog and
  update all in-repository callers in the same development line.
- [x] Remove Apollo's workspace `ndarray` dependency after its manifests,
  kernels, validation, and Python bindings consume native Leto arrays.

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
