# Leto Development Checklist

**Target version: 0.40.0** · **Phase: Closure**

## LETO-KERNEL-BENCHMARKS-1 [patch] — Owner: Codex `/root`

- [x] Extend the canonical `leto-ops` Criterion harness so elementwise,
      reduction, and matmul groups each expose a contiguous baseline and a
      genuinely strided fallback case without changing the timed workload.
- [x] Record the benchmark design, exact commands, and result limits in the
      matching `gap_audit.md` entry; do not claim a production optimization
      until a controlled baseline comparison identifies one.
- [x] Pass the package format, warning-denied check/Clippy, configured
      Nextest, doctest, and Rustdoc gates for the benchmark target.

**Evidence:** prepared-view default-feature Criterion coverage reports elementwise add
`11.796 µs` contiguous vs `49.229 µs` step-2 and sum `3.6693 µs` contiguous
vs `34.150 µs` step-2. Matmul reports `407.01 µs` dense vs `297.46 µs`
step-2 in the matched run, but a prior `217.13 µs` step-2 sample and build
contention make that row non-actionable until a quiet-host rerun. Provider
gates pass: 306/306 Nextest, 8/8 doctests, warning-denied Clippy/Rustdoc,
format, and diff checks.

## LETO-MATMUL-PERF-1 [minor] — Owner: Codex `/root` (complete)

- [x] Establish a quiet-host, counterbalanced dense matmul baseline against
      ndarray at 64×64, 128×128, and 256×256 before changing production code.
- [x] Profile the current row/block/column kernel and cache-topology decision;
      reject any tile, packing, dispatch, or allocation change without a
      statistically significant value-preserving improvement.
- [x] If the profile identifies a complete provider-owned fix, implement it
      in the canonical matmul module with differential tests and synchronized
      benchmark/PM evidence; otherwise close this item as an evidence-only
      audit with the measured blocker and no speculative rewrite.

**Evidence:** current default-feature release measurements report Leto versus
ndarray medians of `23.597/12.770 µs` (64×64), `123.63/113.07 µs`
(128×128), and `233.60/952.54 µs` (256×256). A focused threshold rerun reports
parallel `23.597 µs` versus serial `27.483 µs` at 64×64; serial is also
`223.69 µs` and `1.8522 ms` at 128×128 and 256×256. Flamegraph collection was
blocked by missing Windows `dtrace` and administrator-only `blondie`. No
production change is justified; future matmul work requires a working profile.

## LETO-STRIDED-REDUCE-1 [patch] — Owner: Codex `/root` (complete)

- [x] Establish the current `sum_strided_step2_256x256` baseline and inspect
      the fallback's allocation and memory-access behavior.
- [x] Implement one generic, zero-copy reduction fallback only if the
      measured model identifies a real loop-overhead or dependency-chain
      defect; preserve positive/negative stride value semantics.
- [x] Add focused regression coverage and update the benchmark/gap evidence;
      otherwise close this item as evidence-only with the blocker recorded.

**Evidence:** the quiet baseline measured `sum_strided_step2_256x256` at
`28.849 µs` [28.408, 29.110]. An order-preserving four-way generic loop
candidate measured `27.793 µs` [27.052, 28.853], `p = 0.06`, which is not a
significant improvement; the same candidate run moved the contiguous control
to `4.683 µs` [4.4823, 4.8297] from `4.1184 µs` [4.0946, 4.1298]. The helper
was removed. The zero-copy row-walk implementation remains canonical, and
`whole_reduction_preserves_non_unit_stride_values` adds value-semantic
coverage. After removal, the unchanged implementation measured `28.226 µs`
[27.481, 28.889] in a 20-sample run; an intervening run measured `31.633 µs`
[30.099, 33.701]. The spread is not attributed to the candidate.

## LETO-EXTERNAL-ORACLE-1 [arch] — Owner: Codex `/root` (complete)

- [x] Verify normal dependency ownership and enumerate active dev-only oracle
      imports, examples, and benchmark rows.
- [x] Reconcile the stale removal plan with the current independent evidence
      harnesses; do not delete active comparisons without an equivalent oracle.
- [x] Synchronize the backlog and gap audit, then run the graph and focused
      documentation gates.

**Evidence:** normal dependency graph has zero `ndarray`, `ndarray-rand`, and
`nalgebra` edges; the dev graph resolves `ndarray 0.16.1`, `ndarray-rand 0.15.0`,
and `nalgebra 0.35.0`. Seven active source files use these crates for
independent tests, examples, and benchmark comparisons. The stale removal
plan is closed without deleting verification evidence or changing production
ownership.

## LETO-SPARSE-LU-VIEW-1 [minor] — Owner: Codex `/root`

- [x] Add the provider-owned `ArrayView1` sparse-LU solve seam and preserve
      the legacy slice API through that canonical implementation.
- [x] Add generic value-semantic provider coverage for the view path.
- [x] Convert the `CFDrs` direct solver to pass an `Array1` view and return the
      provider-owned solution directly.
- [x] Synchronize the provider/consumer Rustdoc, changelog, and active PM
      evidence with the allocation ownership boundary.
- [x] Pass format, warning-denied checks, configured Nextest, doctest, and
      Rustdoc gates for both affected repositories.
- [x] Commit and publish the provider and consumer increments, then integrate
      the exact upstream revision in `CFDrs`.

## LETO-PARITY-HARNESS-1 [patch] — Owner: Codex `/root/implement_horae`

- [x] Cross the one-hour stale-claim threshold with no renewed Leto process,
      write, or commit; preserve the two existing harness commits and manifest
      clarification on a dedicated branch.
- [x] Replace boolean-only and inconsistent tolerance checks with reported,
      value-semantic differentials and analytically scaled bounds.
- [x] Correct exercised-API, SSOT, and performance-evidence claims.
- [x] Synchronize README and completeness evidence without duplicating the
      canonical test ownership.
- [x] Run focused example execution and proportional repository gates.
- [x] Publish as PR #69, resolve the single hosted review finding, merge, and
      remove the branch after the locally authoritative gates pass.

**Evidence:** both examples execute eleven bounded value differentials;
focused example Nextest passes 7/7; warning-denied all-target/all-feature
Clippy passes; configured all-target/all-feature Nextest passes 688/688;
doctests pass 8/8; warning-denied Rustdoc passes; ndarray 0.16.1 and nalgebra
0.35.0 are isolated to dev-only oracle ownership. The repository defines no PR
test workflow; Greptile's single P2 finding was fixed and resolved.

## LETO-PYTHON-RELEASE-1 [patch] — Owner: Codex `/root`

- [x] Add the pinned build-once GitHub Release and PyPI workflow.
- [x] Document the `leto-python` distribution, `leto_python` import, Cargo
      version source, supported CPython range, and OIDC publication contract.
- [x] Build, install, import, and inspect a production CPython 3.13 wheel
      locally as `leto-python` 0.39.0 / `leto_python`.
- [x] Create the protected `pypi` environment restricted to
      `leto-python-v*` tags.
- [ ] Pass hosted CI on the exact release-automation head.
- [ ] Register the PyPI pending trusted publisher.

## LETO-NDARRAY-BOUNDARY-1 [major] — Owner: Codex `/root`

- [x] Remove the public feature, conversion module, re-export, and
      conversion-only contract suite without weakening canonical Leto tests.
- [x] Record the ownership decision, unsafe-boundary removal, migration, and
      retained-oracle proof in ADR 0017 and synchronized public documentation.
- [x] Correct the `Tiles` constructor Rustdoc link that blocks documentation.
- [x] Verify production/dev dependency separation and the current Apollo native
      Leto consumer contract.
- [x] Pass format, warning-denied Clippy, configured Nextest, doctest, Rustdoc,
      dependency-residue, and SemVer classification gates.
- [x] Commit, publish, review, and merge the boundary in Leto 0.40.0; refresh
      Apollo to native Leto arrays. The meta-repository owns its gitlink update.

**Provider evidence:** format and warning-denied all-target/all-feature Clippy
pass for `leto` and `leto-ops`; configured Nextest passes 266/266 and 305/305;
doctests pass 1/1 and 8/8; warning-denied Rustdoc passes; six Atlas consumer
source/manifest scans contain no removed-surface residue; the normal dependency
graph contains no `ndarray` while the dev graph contains one oracle edge; and
`cargo-semver-checks` reports the removed feature and module as the two expected
major breaks, with explicit major-release classification passing.

## LETO-LAPLACIAN-1 [minor] — Owner: Codex `/root`

- [x] Define the validated Aequitas spacing, boundary, and polarity contract.
- [x] Implement one native-precision CPU stencil into caller-owned storage.
- [x] Add generic `f32`/`f64` closed-form Neumann coverage.
- [x] Pass focused format, check, Clippy, Nextest, doctest, and rustdoc gates.

**Evidence:** all-target/all-feature check and warning-denied Clippy pass;
configured Nextest passes 575/575; doctests pass 9/9; warning-denied rustdoc
passes; and the generic closed-form stencil regression passes for `f32` and
`f64`.

## LETO-EUNOMIA-PRECISION-1 [major] — Owner: Codex `/root`

- [x] Reconcile origin and isolate the migration from fresh peer-owned
  oracle-formatting edits in a bounded worktree.
- [x] Replace raw half types in scalar, real, arithmetic, and fixture contracts
  with Eunomia `F16`/`Bf16`; remove direct `half` dependencies.
- [x] Refresh the lock to merged Eunomia and Hermes defaults and prove one
  identity for each provider.
- [x] Pass format, warning-denied all-target/all-feature Clippy, configured
  Nextest, doctests, rustdoc, no-default-feature compilation, residue audits,
  and semver classification. `leto` and `leto-ops` baselines classify clean;
  `leto-python` semver extraction is externally blocked by a Rust 1.95 rustdoc
  ICE in NumPy's `ToPyArray::to_pyarray` link, while direct workspace rustdoc is
  green.
- [x] Publish, review, and merge as PR #46 (`0afece5`); preserve the peer-owned
  main-tree edits byte-for-byte. Worktree removal follows this closeout merge.

2026-07-18 [patch, complete]: Advanced the reproducibility lock from Eunomia
0.2.0 `6f431f2d` to Eunomia 0.4.0 `49dc115`, carrying the canonical
round-to-nearest-even sub-byte conversion and corrected reduced-format
constants into Leto without changing Leto's public surface. Formatter,
warning-denied all-target/all-feature Clippy, configured Nextest, doctest, and
warning-denied rustdoc pass; Nextest is 593/593 and doctests are 9/9.

2026-07-18 [patch, complete]: Took over the stale sparse duplicate-conversion
scope from the shared tree. Replaced HashMap and multi-vector duplicate
handling with one sorted streaming compaction; normalized arbitrary COO order
inside the CSC construction boundary; removed redundant caller sorting; and
replaced tautological sparse tests with exact column/value assertions.
Formatter and warning-denied all-target/all-feature Clippy pass; focused sparse
Nextest passes 18/18 and the full Leto package passes 267/267; doctest passes
1/1; warning-denied rustdoc and 196/196 applicable SemVer checks pass. The lock
also advances to Eunomia 0.2.0 `6f431f2d`, deleting its former `num-traits`
edge. Merged as PR #41 (`9b22301`).

2026-07-18 [patch, complete]: Removed Leto's restored direct `num-complex`
workspace dependency and bind every complex migration/eigenvalue/Schur oracle
to Eunomia's canonical representation. Direct manifest/source and production
graph residue are zero; warning-denied all-target/all-feature Clippy passes;
Nextest passes 305/305; doctests pass 8/8; warning-denied rustdoc and 196/196
applicable SemVer checks pass. Merged as PR #42 (`cf47686`).

2026-07-17 [patch, complete]: Registered `LETO-SPARSE-DIRECT-1` as the
upstream-owned replacement boundary for CFDrs sparse LU. Source inspection
confirms Leto 0.38 provides CSR CG/GMRES but no sparse direct factorization,
while CFDrs invokes its direct solver after GMRES stagnation or breakdown.
The backlog item pins native-precision genericity, independent direct-solver
semantics, reusable factors, typed failures, authoritative algorithm grounding,
value-semantic and differential verification, and same-increment consumer
removal of `rsparse`. This tracking increment changes no code or API.

2026-07-16 [minor, complete]: Removed direct revision quarantine for
Mnemosyne, Moirai, Hermes, Eunomia, and Themis. Leto 0.37.0 records Rust 1.95
in every published package because merged Mnemosyne 0.5/Core 0.2 require it.
Evidence: one locked source identity for each provider; Rust 1.95 accepts
`leto-ops` and Rust 1.94 rejects the graph; formatter; explicit-nightly
warning-denied release Clippy; configured release Nextest passes 568/568;
doctests pass 9/9; rustdoc is warning-clean; and offline rustdoc SemVer
comparisons pass all 196 applicable checks for each published Rust API crate
(`leto` and `leto-ops`).

2026-07-16 [minor, complete]: Added the Helios-driven checked
rotation-column-to-unit-quaternion provider contract in the dedicated
`geometry::rotation` leaf. Evidence: generic `f32`/`f64` value semantics;
exact non-orthogonal, left-handed, non-finite, and tolerance-rejection tests;
formatter; locked package check; warnings-denied all-target/all-feature Clippy;
249/249 configured Nextest; 1/1 doctest; warning-clean Leto rustdoc; and
repository-baseline SemVer comparisons for `leto` and `leto-ops` (no required
update). `leto-python` SemVer rustdoc remains toolchain-blocked by the existing
NumPy 0.23 intra-doc-link ICE; the package is already `doc = false`, and the
provider package documentation gate is clean.

2026-07-06 (provider worktree commit closeout). Split the dirty provider
worktree into `11722c4` (`leto-ops` sparse/traversal/linalg export provider
surface) and `3331eb1` (`leto` owned-array serde, assignment/indexing, aliases,
and fixed geometry coverage). Added the committed nextest timeout profile at
`.config/nextest.toml` so focused gates use the 30 s slow / 60 s terminate
budget. Evidence tier: empirical value tests plus compile/lint/rustdoc gates.
Provider gates run in this closeout: `leto-ops` focused sparse/structure/
elementwise/properties nextest filters (18/36/18/11) and full all-features
nextest (271/271), all-target all-features clippy, rustdoc with warnings
denied; `leto` focused serde/indexing/array API and geometry filters
(18/19), full all-features nextest (199/199), all-target all-features clippy,
rustdoc with warnings denied; `leto-python` all-target all-features clippy and
nextest (21/21); workspace rustdoc with warnings denied.

2026-07-05 (CR-4 scalar SSOT rebind complete). Rebased `leto_ops::Scalar`
onto `eunomia::NumericElement` per ADR 0005, resolving the 47-commit
divergence with `origin/main` PR #30. `Scalar: NumericElement` keeps only
`from_usize` + default-bodied slice kernels; `RealScalar: Scalar + FloatElement`.
Old standalone methods (`ZERO/ONE/add/sub/mul/div/bitand/bitor/bitxor/
count_ones/to_f64`) inherited from Eunomia. Merge conflicts resolved in
`scalar.rs`, `lib.rs`, `array.rs`, `sparse/mod.rs`. Added `Point1` geometry,
`X<T>` 1D swizzle view, and missing CSR utility methods (`zeros`, `values_mut`,
`scale_values/rows/columns`, `frobenius_norm`, `is_strictly_diagonally_dominant`,
`transpose`, `condition_estimate`). No compatibility shims. Evidence:
`rustup run nightly cargo check -p leto-ops --all-features`,
`rustup run nightly cargo fmt --package leto-ops --check`,
`rustup run nightly cargo clippy -p leto-ops --all-targets --all-features -- -D warnings`,
and `rustup run nightly cargo nextest run -p leto-ops --all-features` (271/271)
pass. Clippy also reports the pre-existing upstream
`hermes-simd-core::sparse::ValidatedData::new_unchecked` dead-code warning
while exiting successfully for the `leto-ops` gate.

2026-07-04 (CFDrs sparse extension CSR utility provider gap). Added
`CsrMatrix` provider methods for diagonal extraction, scalar/value scaling,
row scaling, column scaling, Frobenius norm, strict diagonal dominance, and a
diagonal-dominance condition estimate. CFDrs consumes these from
`SparseMatrixExt`, removing downstream CSR traversal loops while the public
storage boundary remains `nalgebra_sparse::CsrMatrix`. Evidence:
`rustup run nightly cargo fmt -p leto-ops --check`, `cargo check -p leto-ops`,
`cargo nextest run -p leto-ops --test ops_tests sparse --status-level fail`
(18/18), and `cargo clippy -p leto-ops --all-targets -- -D warnings` pass.
Downstream CFDrs `cfd-math` fmt/check/focused sparse nextest/all-target clippy
also pass.

2026-07-04 (CFDrs AMG CSR transpose provider gap). Added
`CsrMatrix::transpose()` so CFDrs AMG can construct restriction operators
(`R = P^T`) through Leto-owned CSR storage instead of `nalgebra_sparse`
CSR-to-CSC conversion. The implementation count-prefix-scans output rows and
scatters source entries while preserving sorted row-column invariants. Evidence:
`rustup run nightly cargo fmt -p leto-ops --check`, `cargo check -p leto-ops`,
`cargo nextest run -p leto-ops --test ops_tests sparse --status-level fail`
(16/16), `cargo clippy -p leto-ops --all-targets -- -D warnings`, and `cargo
doc -p leto-ops --no-deps` pass. Downstream CFDrs `cfd-math` check, focused
sparse/AMG nextest, and all-target clippy also pass.

2026-07-04 (CFDrs AMG sparse product provider gap). Added
`leto_ops::spgemm` for CSR×CSR products plus `CsrRow::nnz`, so CFDrs AMG can
replace its `nalgebra_sparse` Galerkin product (`R * A * P`) through Leto-owned
sparse operators instead of retaining a downstream multiply. The implementation
uses per-row sorted accumulation and drops exact-zero cancellations while
preserving CSR row-order invariants. Evidence: `rustup run nightly cargo fmt -p
leto-ops --check`, `cargo check -p leto-ops`, `cargo nextest run -p leto-ops
--test ops_tests sparse --status-level fail` (14/14), `cargo clippy -p
leto-ops --all-targets -- -D warnings`, and `cargo doc -p leto-ops --no-deps`
pass.

2026-07-04 (CFDrs mesh rotation provider gap). Added
`FixedMatrix<T, 3, 3> * leto::geometry::Vector3<T>` so consumers can replace
nalgebra `Matrix3 * Vector3` mesh transforms with Leto-owned fixed geometry.
CFDrs uses this in `cfd-core::geometry::mesh::MeshOperations::rotate` while
moving mesh storage to Leto points/vectors. Evidence: `rustup run nightly cargo
fmt -p leto --check`, `cargo check -p leto`, `cargo nextest run -p leto
--status-level fail` (171/171), `cargo clippy -p leto --all-targets -- -D
warnings`, and downstream cfd-core no-default check/clippy/full nextest pass.

2026-07-04 (CFDrs Domain Point1 provider gap). Added `Point1<T>` to
`leto::geometry`, added conditional `Eq` derives to fixed geometry values, and
wired Leto's `std`/`alloc` features through to serde so Vec-backed serde
surfaces compile in direct provider checks. CFDrs uses this to move
`cfd-core::geometry::shapes::Domain` and its boundary/domain contract from
nalgebra point/vector/scalar types to Leto/Eunomia. Evidence: `rustup run
nightly cargo fmt -p leto --check`, `rustup run nightly cargo check -p leto`,
`rustup run nightly cargo nextest run -p leto --status-level fail` (170/170),
`rustup run nightly cargo clippy -p leto --all-targets -- -D warnings`, and
downstream `cfd-core` no-default check/nextest/clippy pass.

2026-07-04 (Kwavers Layout serde rank gap). Replaced the `Layout<N>` Serde
derive with a manual implementation that serializes shape/stride slices and
validates decoded rank before rebuilding `[usize; N]` and `[isize; N]`. This
keeps owned-array serde provider-owned for Kwavers and other consumers at ranks
above serde's fixed-array impl limit. Evidence: `rustup run nightly cargo fmt
-p leto --check`, `rustup run nightly cargo check -p leto --all-targets`,
`rustup run nightly cargo nextest run -p leto layout --status-level fail
--no-fail-fast` (15/15), and `rustup run nightly cargo clippy -p leto
--all-targets --no-deps -- -D warnings` pass.

2026-07-04 (CFDrs serialized Array1 provider gap). Added serde support for
owned `Array<T, S, N>`, `VecStorage<T>`, and const-rank `Layout<N>`, with array
deserialization routed through `Array::new` so layout/storage bounds are
validated before reconstruction. CFDrs `FieldData::Scalar` uses this to move
serialized state storage from nalgebra `DVector` to `leto::Array1` without a
downstream wrapper. Evidence: `rustup run nightly cargo fmt -p leto --check`,
`rustup run nightly cargo nextest run -p leto
owned_array_round_trips_shape_and_values_through_serde --status-level fail`,
`rustup run nightly cargo clippy -p leto --all-targets -- -D warnings`, and
downstream `cfd-core` no-default check/clippy/state nextest pass.

2026-07-04 (Kwavers CPML Array1 provider gap). Added rank-1 `Array1` indexing
by `usize` and owned-array `PartialEq`/`Eq` value semantics so Kwavers CPML
profiles and `PmlExpFactors` can move from ndarray `Array1` to Leto without a
downstream helper. Evidence: `rustup run nightly cargo fmt -p leto`,
`rustup run nightly cargo nextest run -p leto
test_owned_array_equality_checks_shape_and_values --status-level fail
--no-fail-fast`, downstream `kwavers-boundary` CPML nextest 15/15,
downstream `kwavers-gpu --features cuda-provider` PSTD nextest 24/24, and
downstream `kwavers-solver` PML nextest 45/45 pass.

2026-07-03 (CFDrs FVM fixed-vector provider gap). Added the
`leto::geometry::Vector2<T>` alias plus generic fixed-vector
`norm_squared`, `norm`, `try_normalize`, and `normalize` methods so CFDrs FVM
face geometry can replace nalgebra `Vector2` without a downstream wrapper.
Evidence: `cargo check -p leto`,
`cargo nextest run -p leto fixed_vector_norm_and_normalization_are_value_semantic`,
`cargo check -p cfd-2d`, and `cargo nextest run -p cfd-2d --lib fvm` pass.

2026-07-03 (CFDrs serialized geometry provider gap). Added Serde derives to
Leto fixed geometry value types (`Point2`, `Point3`, `Vector3`, `UnitVector3`,
and `Isometry3`) so CFDrs can replace serialized nalgebra `Vector3` velocity
storage with `leto::geometry::Vector3` without a downstream wrapper. Evidence:
`rustfmt --edition 2021 --check crates/leto/src/geometry.rs` passes and
`cargo check -p cfd-core` in `D:/atlas/repos/CFDrs` compiles Leto plus the
consumer boundary. `cargo nextest run -p leto geometry` passes 3/3.

2026-07-02 (Atlas special-functions provider lane). Added `leto-ops`
`ErfOp`, `ErfcOp`, and `LgammaOp` as named elementwise unary markers over the
Eunomia real-math surface. This is the provider plumbing Coeus consumes for
exact GELU and `torch.special`-style `erf`/`erfc`/`gammaln` parity without
local special-function approximations. Evidence: `rustup run nightly cargo
check -p leto-ops`, `rustup run nightly cargo nextest run -p leto-ops
special_unary_ops_match_eunomia_reference_values`, and `rustup run nightly cargo
fmt -p leto-ops --check` pass.

2026-07-02 (scalar SSOT audit). Closed the first Leto-side scalar ownership
slice by rebasing `leto_ops::Scalar` on `eunomia::NumericElement` and
`RealScalar` on `eunomia::FloatElement`. Leto now owns only operation-local
slice/SIMD hooks plus `Scalar::from_usize`; Eunomia owns constants, primitive
arithmetic/bit contracts, finite predicates, and real transcendental methods.
No compatibility shims were added. Downstream fallout remains consumer-owned:
Apollo/Coeus code that explicitly names removed Leto UFCS items should import
Eunomia traits directly, while platform-sized index values should stay domain
indices unless Eunomia adds numeric-element support upstream. Evidence:
`cargo fmt -p leto-ops --check`, `cargo check -p leto-ops`,
`cargo clippy -p leto-ops --all-targets -- -D warnings`, and
`cargo nextest run -p leto-ops scalar_traits_are_eunomia_extensions
special_unary_ops_match_eunomia_reference_values
integer_scalar_elementwise_ops_are_value_semantic` pass 3/3.

2026-07-02 (Eunomia platform-sized scalar extension). Extended Eunomia's sealed
primitive numeric SSOT to `isize`/`usize` with pointer-width metadata and
value-semantic integer contract coverage, then re-enabled the corresponding
`leto_ops::Scalar` impls through the Eunomia supertrait boundary. This resolves
the only Leto-side scalar type contraction from the initial SSOT slice without
adding Leto aliases or forwarding shims. Evidence: in `D:/atlas/repos/eunomia`,
`cargo check -p eunomia`, `cargo clippy -p eunomia --all-targets -- -D warnings`,
and `cargo nextest run -p eunomia` pass; in `D:/atlas/repos/leto`,
`cargo check -p leto-ops`, `cargo clippy -p leto-ops --all-targets -- -D warnings`,
`cargo fmt -p leto-ops --check`, and `cargo nextest run -p leto-ops
scalar_traits_are_eunomia_extensions integer_scalar_elementwise_ops_are_value_semantic`
pass 2/2.

2026-07-02 (Gaia/Kwavers FEM fixed-matrix inverse gap). Closed the
consumer-driven tetrahedral Jacobian inverse gap by adding
`FixedMatrix<T, 3, 3>::try_inverse(min_abs_det)`. The provider rejects
non-finite and near-singular determinants instead of forcing Kwavers to keep a
nalgebra inverse path. Evidence: `cargo fmt -p leto --check`,
`cargo check -p leto`, `cargo clippy -p leto --all-targets -- -D warnings`,
and `cargo nextest run -p leto fixed_matrix_inverse` pass 2/2. Downstream
Kwavers FEM source audit is clean for `Matrix3`/`Vector3` in the FEM core; full
Kwavers compile is blocked by broader lock/provider convergence outside this
provider primitive.

2026-07-02 (Kwavers self-adjoint sparse coordinate provider gap). Closed the
consumer-driven source-injection gap by adding `leto_ops::coordinate_map_inplace`,
the sparse logical-coordinate mutable traversal needed by Kwavers self-adjoint
FWI, plus `CoordinateMapPlan`/`coordinate_map_plan_inplace` for prevalidated
repeated sparse-coordinate updates. Repeated coordinates are visited in input
order, out-of-bounds coordinates return `LetoError::OutOfBounds`, plan
application rejects layout mismatch, and zero-stride mutable aliasing is
rejected. This is provider-owned CPU array capability, not a downstream Kwavers
helper.
Evidence: `cargo fmt --package leto-ops --check`,
`cargo check -p leto-ops`, `cargo clippy -p leto-ops --all-targets --no-deps
-- -D warnings`, and `cargo nextest run -p leto-ops coordinate_map`
pass 4/4; downstream `cargo check -p kwavers-solver`, `cargo clippy -p
kwavers-solver --lib --no-deps -- -D warnings`, focused self-adjoint nextest
6/6, and `cargo nextest run -p kwavers-solver inverse::fwi::time_domain`
59/59 pass. Kwavers remains on direct `coordinate_map_inplace`; a planned
downstream consumption attempt exceeded the 30 s focused-test budget and was
reverted pending profiling.

2026-07-01 (Kwavers FWI Fortran-order indexed provider gap). Closed the
consumer-driven recorder/source voxel-list gap by adding
`leto_ops::indexed_fold_fortran`, the one-view indexed reduction that visits
logical indices in column-major order. This is provider-owned CPU array
capability, not a downstream Kwavers compatibility helper. Evidence:
`cargo fmt --package leto-ops --check`,
`cargo clippy -p leto-ops --all-targets --no-deps -- -D warnings`, and
`cargo nextest run -p leto-ops
test_indexed_fold_fortran_uses_column_major_logical_order` passes 1/1;
downstream `cargo check -p kwavers-solver`, `cargo clippy -p kwavers-solver
--lib --no-deps -- -D warnings`, focused self-adjoint nextest, and
`cargo nextest run -p kwavers-solver inverse::fwi::time_domain` pass 59/59.

2026-07-01 (Kwavers MOFI multi-output indexed provider gap). Closed the
consumer-driven rigid-Jacobian fill gap by adding `leto_ops::indexed_map4_inplace`,
the four-mutable-output indexed traversal needed to replace a downstream
coordinate loop without recomputing the transform four times. This is
provider-owned CPU array capability, not a downstream Kwavers compatibility
helper. Evidence: `cargo fmt --package leto-ops --check`,
`cargo clippy -p leto-ops --all-targets --no-deps -- -D warnings`, and
`cargo nextest run -p leto-ops test_indexed_map4_inplace_fills_multiple_outputs`
passes 1/1; downstream `cargo check -p kwavers-solver`,
`cargo clippy -p kwavers-solver --lib --no-deps -- -D warnings`, focused MOFI
Jacobian nextest, and `cargo nextest run -p kwavers-solver
inverse::fwi::time_domain::mofi` pass 9/9.

2026-07-01 (Kwavers FWI indexed fold provider gap). Closed the
consumer-driven adjoint-gradient peak reduction gap by adding
`leto_ops::indexed_fold`, the one-view indexed reduction needed to replace
`ndarray::ArrayBase::indexed_iter().fold` at the provider boundary. This is
provider-owned CPU array capability, not a downstream Kwavers compatibility
helper. Evidence: `cargo fmt -p leto-ops --check`, `cargo check -p leto-ops`,
and `cargo nextest run -p leto-ops indexed_fold` passes 2/2; downstream
`cargo check -p kwavers-solver`, solver library clippy with `-D warnings`, and
`cargo nextest run -p kwavers-solver inverse::fwi::time_domain` pass 59/59.

2026-07-01 (Kwavers FWI all-elements extrema provider gap). Closed the
consumer-driven model-range reduction gap by adding checked `leto_ops::min` and
`leto_ops::max` over the existing reduction marker path. This is provider-owned
CPU array capability, not a downstream Kwavers compatibility helper. Evidence:
no-default consumer-feature `leto-ops` check and clippy pass,
`cargo nextest run -p leto-ops reduce_min_max` passes 3/3, and downstream
`cargo nextest run -p kwavers-solver inverse::fwi::time_domain` passes 59/59.

2026-07-01 (Kwavers FWI indexed map provider gap). Closed the final
FWI time-domain test-helper gap by adding `leto_ops::indexed_map_inplace`, the
one-view indexed mutable traversal needed by Kwavers' self-adjoint sponge
builder. This is provider-owned CPU array capability, not a downstream Kwavers
compatibility helper. Evidence: no-default consumer-feature `leto-ops` check and
clippy pass, `cargo nextest run -p leto-ops indexed_map_inplace` passes 1/1, and
downstream `cargo nextest run -p kwavers-solver inverse::fwi::time_domain` passes
59/59 with no remaining FWI time-domain `Zip::from`/`Zip::indexed` source hits.

2026-07-01 (Kwavers self-adjoint imaging zip provider gap). Closed the
consumer-driven FWI imaging-condition gap by adding `leto_ops::zip5_mut_with`
and `leto_ops::indexed_zip4_mut_with`, covering reconstructed five-read zips
and stored-history indexed four-read zips. These are provider-owned CPU array
capabilities, not downstream Kwavers compatibility helpers. Evidence:
no-default consumer-feature `leto-ops` check passes, focused nextest filters
pass (`zip5_mut_with` 2/2 and `indexed_zip4_mut_with` 1/1), no-default
consumer-feature `leto-ops` clippy passes with `-D warnings`, and downstream
`cargo nextest run -p kwavers-solver inverse::fwi::time_domain` passes 59/59.

2026-07-01 (Kwavers FWI two-view reduction provider gap). Closed the
consumer-driven FWI relative-model-change gap by adding `leto_ops::zip_fold`,
the reduction analogue of `zip_mut_with` for two read-only views. This is
provider-owned CPU array capability, not a downstream Kwavers compatibility
helper. Evidence: no-default consumer-feature `leto-ops` check passes,
`cargo nextest run -p leto-ops zip_fold` passes 3/3, no-default
consumer-feature `leto-ops` clippy passes with `-D warnings`, and downstream
`cargo nextest run -p kwavers-solver inverse::fwi::time_domain` passes 59/59.

2026-07-01 (Kwavers FWI four-view zip provider gap). Closed the
consumer-driven FWI pressure second-derivative gap by adding
`leto_ops::zip3_mut_with`, the four-operand mutable zip analogue for one output
view plus three read-only input views. This is provider-owned CPU array
capability, not a downstream ndarray compatibility helper in Kwavers. Evidence:
no-default consumer-feature `leto-ops` check passes,
`cargo nextest run -p leto-ops zip3_mut_with` passes 2/2, no-default
consumer-feature `leto-ops` clippy passes with `-D warnings`, and downstream
`cargo nextest run -p kwavers-solver inverse::fwi::time_domain` passes 58/58.

2026-07-01 (Gaia/Kwavers geometry and array API provider gap). Closed the
consumer-driven gap exposed by routing Kwavers through Gaia and Leto directly:
`leto` now exposes fixed vectors/matrices, `Point3`/`Vector3`/`UnitVector3` and
small isometry primitives, plus ndarray-compatible owned-array conveniences
needed by the Atlas consumer graph (`as_slice_mut`, memory-order mutable slice
access, `mapv`, `zip_map`, `fill`, `assign`, and `[usize; N]` indexing). This is
not an ndarray compatibility helper in Kwavers; the missing provider surface was
filled in Leto. Evidence: `cargo nextest run -p leto fixed geometry` (5 passed),
`cargo nextest run -p leto array_index_fill_map_and_zip_are_value_semantic` (1
passed), and the downstream `cargo check -p kwavers-solver` gate passes through
Gaia/Ritk/Apollo/Kwavers provider routing.

2026-06-23 (batched-matmul aliasing UB). Closed the recorded follow-on: the
parallel `batched_matmul` closure formed a full-buffer `&mut [T]` per task
(`from_raw_parts_mut(out_ptr, out_len)`), writing only its batch sub-region —
runtime-disjoint but Tree-Borrows UB to hold N concurrent full-range `&mut`.
Each task now borrows only its batch's physical span (`min_max_offsets`) with a
rebased offset; a disjointness guard (`batch_stride ≥ per-matrix span`, non-empty)
routes interleaved-batch outputs to the sound sequential loop. A full-surface
sweep confirmed this was the only such site (per-row/per-block/per-chunk kernels
already disjoint). New tests: interleaved-output fallback (vs C-contiguous
reference) + empty-output boundary. Evidence: `cargo fmt --all --check`; `cargo
clippy --workspace --all-targets -- -D warnings`; `cargo nextest run --workspace`
(407 passed); `cargo test --doc --workspace` (5 doctests). Soundness is by
Tree-Borrows reasoning + differential oracles (Miri unavailable on this
Windows/moirai-asm env). Note: the shared `moirai` path-patched dep was
transiently broken mid-session by a concurrent agent and self-recovered; not
touched per non-interference.

2026-06-23 (matmul offset-routing). Audited the highest-unsafe-density paths
(`view.rs` aliasing, storage exception-safety, parallel `matrix.rs`). Storage is
clean; parallel output writes are disjoint. Fixed two real perf/memory defects:
(1) `matmul`/`route_matmul` now route dense-but-offset views (every batch `b>0`
of `batched_matmul`, any sliced sub-array output) through the in-place fast
kernels via new offset-independent `is_c_dense`/`is_f_dense` predicates instead
of the offset-pinned `is_c_contiguous` — eliminating the per-batch scratch
allocation + operand copy + copy-back; (2) `batched_matmul`'s parallel path
replaced the per-batch `Mutex` poll with a relaxed `AtomicBool` early-out.
Offset-0 contiguous inputs take the identical branch (no codegen change on
benchmarked paths). New test `test_matmul_into_offset_c_dense_view_writes_in_place`
pins value-correctness + no out-of-view writes. Recorded residual risk: the
batched parallel closure's full-buffer `&mut` aliasing (Tree-Borrows UB though
runtime-disjoint) — filed as a follow-on [patch]. Removed the stale 1.5 GiB
`target_ag/` orphan target tree (disk was 100% full). Evidence: `cargo fmt --all
--check`; `cargo clippy --workspace --all-targets -- -D warnings`; `cargo nextest
run --workspace` (405 passed); `cargo test --doc --workspace` (5 doctests);
`cargo doc -p leto` (3 pre-existing unrelated warnings; the later
`leto-python`/`numpy 0.23` rustdoc ICE is now resolved by the 2026-07-05
PyO3-extension doc-target exclusion).

Sprint phase: Execution (performance track). Target version: 0.35.1 [patch]
(Cargo.toml bumped; CHANGELOG synced). Unreleased minor/patch lane: Leto owns
narrow CPU CSR sparse-dense parity kernels while Coeus keeps higher sparse
formats/backends; trace, rank, Kronecker, and keep-dim axis reductions use
validated stride walks instead of repeated logical offset recomputation in hot
loops. Negative-stride regression tests cover reverse views; trace/rank/Kronecker
now have crate-root and `application` export coverage plus free-function
doctests. Evidence: `rustup run nightly cargo fmt -p leto-ops --check`;
`rustup run nightly cargo check -p leto-ops --all-features`; `rustup run nightly
cargo clippy -p leto-ops --all-targets --all-features -- -D warnings`; `rustup
run nightly cargo nextest run -p leto-ops --test ops_tests properties
--all-features --status-level fail` (11 tests); `rustup run nightly cargo
nextest run -p leto-ops --all-features` (265 tests); `rustup run nightly cargo
test -p leto-ops --doc --all-features` (8 doctests); `rustup run nightly cargo
doc -p leto-ops --no-deps --all-features`; focused reverse-axis reduction
benchmark improved 11.742% median. Prior unreleased patch:
`Scalar::tiled_gemm` now defaults to scalar GEMM
and concrete real/half scalar impls opt into `SimdStrategy`, fixing generic
integer `Scalar` builds exposed by Hephaestus WGPU nextest. Evidence: `cargo fmt
-p leto-ops --check`; `cargo clippy -p leto-ops --all-targets -- -D warnings`;
Hephaestus dependent gate `cargo nextest run -p hephaestus-wgpu` (43 passed).
Delivered 0.35.1: `matexp` even/odd Padé
split (Paterson–Stockmeyer) — N=U+B·V, D=U−B·V via even powers B²/B⁴/B⁶ + one
B·V product → **4 matmuls instead of 6** for the Padé step; added shared dense
`sub`; compile-time assert ties the unrolling to q=6. Evidence: provable op-count
reduction + unchanged matexp battery (7 tests, ops_tests 186). Honest perf note:
wall-clock within criterion noise at 32–64 (small-norm ⇒ s=0; LU-inverse +
remaining products dominate); benefit grows with n / when s>0. Collision-free
(`matrix_function/`; peer agent active in svd/eig/francis lane). Delivered 0.35.0:
**full thin SVD via
bidiagonal QR** (`svd_via_bidiagonal`, Golub–Reinsch) with U/V Givens
accumulation in `svd/bidiagonal_qr.rs`. Const-generic `VEC`: values-only DCE's
the U/V rotations (zero cost); full path accumulates into the bidiagonalization
factors. Wide handled via σ(A)=σ(Aᵀ)+swap; sign-normalized σ≥0, descending sort
carrying U/V columns. `svd_decompose` rerouted to it (Gram path **deleted** +
dead `singular_value_or_zero` removed — SSOT), full-rank-rejection contract
preserved. **Closes the headline SVD perf gap**: full SVD 32×32 292→103.6 µs
(10.6×→3.5×), 64×64 3.1 ms→758.8 µs (18×→4.1×). Values-only
`singular_values` now skips factor accumulation through the same const-generic
kernel: 57.1 µs (32×32) and 275.0 µs (64×64), reducing the local criterion
median by 25.5% and 39.5%. Accuracy improves versus Gram (`κ(A)`, not `κ(A)²).
Evidence: reconstruction + orthonormality + nalgebra σ across tall/square/wide
(svd 14 tests, ops_tests 186; svd_decompose contract tests green). Jacobi
(`svd_rank_revealing`) retained for rank-deficient/max-accuracy. Residual ~3.5–4.1×
= scalar bidiagonalization + scalar Givens (vectorization next lever, shared with
eigenvalues residual). Delivered 0.34.3: `singular_values` moved
to implicit-shift bidiagonal QR (Golub–Kahan) in new `svd/bidiagonal_qr.rs`
(reuses `bidiagonalize`, SSOT) — no `AᵀA`, so accuracy κ(A) not κ(A)²
(diag(1,1e-6)→1e-15). Removed the
dead Gram `singular_values` (one impl, SSOT). Evidence: σ-preservation theorem +
21-matrix nalgebra battery + closed-form + rank-deficient + wide-dynamic-range
(ops_tests 185; matrix_rank + all consumers green). Delivered
0.34.2: eigenvalues no-Q Francis
path — `francis::run` const-generic over `ACCUMULATE_Q`; eigenvalues-only passes
`false` so the Schur-vector update is DCE'd (zero-cost) and standardization is
skipped; block extraction factored to one shared `eigenvalues_from_quasi_triangular`
helper (SSOT, used by `RealSchur::eigenvalues` + the no-Q path). Cumulative eig
speedup: 32×32 992→397.0 µs (~2.5×), 64×64 ~4.8→2.52 ms (~1.9×); eig 8 + schur
7 tests green. Gap vs nalgebra now ~5.8–7.4× (residual = scalar reflector application;
vectorization deferred). Delivered 0.34.1: performance gap analysis
+ first optimization. Added `decomposition_compare` criterion baselines
(leto vs nalgebra, LU/QR/Cholesky/SVD/eig/matexp/matpow) → finding: largest gaps
are SVD (~10–18×, one-sided Jacobi) and eigenvalues (~16×), NOT matmul (~2×);
recorded in gap_audit "Performance gap analysis". Resolved the eigenvalues gap
(partial): consolidated `eigenvalues` onto the real Schur (Francis) iteration,
**deleted the complex single-shift QR** (`eigenvalues/{complex,qr}.rs` + `Cplx`) —
one QR iteration in the crate (SSOT), real arithmetic; 32×32 992→581 µs (~1.7×),
contract-preserving (eig 8 + schur 7 tests green). Residual eig gap (~8.8×) needs
a no-Q Francis path (const-generic over Q-accumulation) which lands in
`francis.rs` (peer-agent-active) → deferred/coordinated. Open perf items: SVD
(bidiagonal-QR rewrite reusing `bidiagonalize`); matmul (register-blocked GEMM
micro-kernel, upstream hermes primitive, peer lane). Prior target 0.34.0 [minor]
(Cargo.toml bumped;
CHANGELOG synced). Delivered 0.34.0: PyO3 runtime-rank interop
(`leto_python.sum_dyn`) realizing the ADR 0007 boundary at the binding edge —
arbitrary-rank numpy array → **zero-copy** `ArrayD` (borrowing via
`SliceStorage`) → `into_dimensionality::<N>()` bridge (bounded `match` on
`ndim()`, ranks 1–6) → existing rank-generic `sum` kernel (SSOT, no per-rank
binding code); GIL released around compute; non-contiguous rejected. Removes the
prior compile-time-rank-2 numpy-boundary constraint. Evidence tier: embedded-
CPython 3.13 integration tests (ranks 1/2/3 + non-contiguous rejection; crate's
established binding-test convention — no maturin/pytest harness exists, so the
Rust embedded-Python tests are authoritative; 7 leto-python tests). Closes the
ADR 0007 consumer-driven PyO3 follow-up. Delivered 0.33.0: stack-allocated
`StackStorage<T, CAP>`
backing (inline `[T; CAP]`, no heap, `no_std`/`Copy`) + `Array::from_stack`/
`from_stack_elem`. Reuses the **full** op surface via the `Storage` trait
(DIP/SSOT — zero per-backend code; reductions/iteration/transpose all verified on
stack-backed arrays, 6 tests). ADR 0008 resolves the parity matrix's two
`Excluded?` rows: stack allocation delivered; compile-time fixed *shape*
Excluded(architecture) (leto is const-rank/runtime-dims per ADR 0002); geometry
Excluded(bounded-context) (downstream domain crate, not the array substrate).
**This closes the parity program's open exclude-vs-implement decisions** — §A and
§B are fully resolved (Verified/Complete/Excluded-with-rationale). Remaining:
performance (Verified→Complete via criterion baselines), consumer-driven PyO3
`ArrayD` interop (ADR 0007). Delivered 0.32.0: real Schur decomposition
`A = Q T Qᵀ`
(`schur`, `RealSchur`, `MatrixDecompose::schur`) in new
`linalg/schur/{mod,francis,standardize}.rs` leaf (nalgebra `Schur` parity) — the
Schur **vectors** (orthogonal Q + real quasi-triangular T), the capstone §B gap.
Francis double-shift implicit QR in real arithmetic; reuses Hessenberg + shared
Householder reflectors (SSOT); precision-exact deflation; real-2×2 standardization.
Theorem+proof (implicit-Q) in rustdoc. Evidence tier: exact reconstruction
`A = Q T Qᵀ`, Q orthogonality, quasi-triangular structure (2×2 only for complex
pairs), spectrum vs `eigenvalues` kernel + nalgebra (7 tests; ops_tests 183 green).
Concurrent-agent note: the Francis bulge-chase initial-reflector-size fix was
applied cooperatively by the peer agent; module structure/wiring/tests are mine.
**This closes the last substantive §B nalgebra-decomposition gap.** Remaining §A
Partial: random-constructor distribution-oracle depth; cross-cutting: PyO3 `ArrayD`
interop. Delivered 0.31.0: symmetric-indefinite Bunch–Kaufman
`P A Pᵀ = L D Lᵀ` with partial pivoting (`bunch_kaufman`,
`BunchKaufmanDecomposition`, `MatrixDecompose::bunch_kaufman`) in new
`linalg/bunch_kaufman/{mod,decompose,solve}.rs` leaf — the stable general form of
the unpivoted UDU; 1×1/2×2 pivot blocks via the α=(1+√17)/8 test, succeeds on
zero-diagonal indefinite matrices. Theorem+proof in rustdoc; exposes l/d/perm/
is_two_by_two/det/solve/inv + fluent method. Evidence tier: **exact reconstruction
identity** `P A Pᵀ = L D Lᵀ` (machine precision, definite+indefinite), det/solve/
inverse differential vs LU, zero-diagonal 2×2-pivot case, 1×1 symmetric
interchange, rejection (8 tests; ops_tests 176 green). Closes the §B "pivoted
Bunch-Kaufman" Missing item.
Remaining §B Missing: Real Schur form (Q,T vectors — needs real Francis
double-shift QR, [major]); §A Partial: random-constructor distribution-oracle
depth. Delivered 0.30.0: matrix functions (`matpow`, `matexp`,
`MatrixFunction` fluent trait) in new `linalg/matrix_function/{dense,power,
exponential,mod}.rs` leaf hierarchy (nalgebra `pow`/`exp` parity). `matpow`:
exp-by-squaring `Θ(log k)`, generic over `Scalar` (exact for integer matrices),
binary-decomposition theorem+proof. `matexp`: scaling-and-squaring + diagonal
Padé(6), documented identity/construction and empirical/differential evidence
tier. Both reuse
`matmul` + LU-inverse (SSOT, no new contraction/solve path); shared dense
helpers in `dense.rs`. Evidence tier: closed-form oracles (zero/diagonal/
nilpotent/skew→rotation) + nalgebra `exp`/`pow` differential + rejection
(12 tests; ops_tests 168 green). Closes the §B "Matrix exp/power" Missing row.
Remaining §B Missing: Real Schur form (Q,T vectors — needs real Francis
double-shift QR), pivoted Bunch-Kaufman; §A Partial: random-constructor
distribution-oracle depth (leto-ops). Delivered 0.29.0: runtime-rank (`IxDyn`)
support via a
boundary carrier + zero-copy rank bridge (ADR 0007), NOT a parallel compute
substrate (keeps ADR 0002's const-rank compute invariant). New `domain/dynamic/`
(`LayoutDyn`) and `application/dynamic/` (`ArrayD<T,S>`, bridge) leaf hierarchies;
`Array::into_dyn` / `ArrayD::into_dimensionality::<N>` move storage unchanged and
translate only O(ndim) shape/stride scalars (allocation-free; compute via rank
recovery → existing const-rank kernels, SSOT). Also refactored strided-layout
arithmetic into shared slice-based kernels (`domain/layout/kernels.rs`) that both
`Layout<N>` and `LayoutDyn` delegate to (SSOT; behavior-preserving — full suite +
leto-ops 156 ops_tests regression-free). Evidence tier: 12 dynamic tests
(round-trip, strided, runtime-rank dispatch, exact rejection contracts) + docs
warning-clean. This closes the last **Missing** §A array/ndarray parity row; the
remaining §A Partial row is random-constructor distribution-oracle depth.
Remaining cross-cutting: PyO3 `ArrayD` interop (consumer-driven follow-up).
Delivered 0.28.2: axis chunk streaming
(`Array`/`ArrayView::axis_chunks_iter` -> `AxisChunks`) in
`application/iter/chunks.rs` (ndarray `axis_chunks_iter` parity). Each yielded
chunk is a non-overlapping zero-copy view along one axis; the final chunk keeps
the remainder, and every other axis keeps the parent extent and strides.
Evidence tier: coverage theorem, remainder values, transposed stride
preservation, double-ended meet-once, invalid-axis rejection, and zero-length
rejection in core/chunks. Delivered 0.28.1: exact chunk streaming
(`Array`/`ArrayView::exact_chunks` -> `ExactChunks`) in
`application/iter/chunks.rs` (ndarray `exact_chunks` parity). Each yielded chunk
is a non-overlapping zero-copy block view of fixed shape; per-axis remainders
are skipped, and transposed/sliced inputs preserve parent strides. Documents the
`prod floor(s_i / c_i)` chunk-count theorem with proof. Evidence tier:
count theorem, skipped remainder values, transposed stride preservation,
double-ended meet-once, empty oversize stream, and zero-extent rejection in
core/chunks. Delivered 0.28.0: zero-copy lane iteration
(`Array`/`ArrayView::lanes`/`lanes_mut` -> `Lanes`/`LanesMut`) in
`application/iter/lanes.rs` (ndarray `lanes`/`lanes_mut` parity). Each lane along
axis `a` is a 1-D view parallel to `a`; mut iteration enforces non-aliasing
layout to safely yield disjoint mutable views. Documents the lane partition
theorem with proof. Evidence tier: partition theorem, count and content across
shapes, dual to rows/columns equivalence, transposed/strided zero-copy
correctness, double-ended iteration, and mutable write disjointness (8 tests,
100 core_tests green). Remaining §A: `IxDyn` (ADR 0002). Delivered 0.27.0: zero-copy sliding-window iteration
(`Array`/`ArrayView::windows` → `Windows`) in `application/iter/windows.rs`
(ndarray `windows` parity). Each window reuses parent strides + shifted offset
(no copy; overlapping windows share storage via shared borrows);
`DoubleEndedIterator`+`ExactSizeIterator` over a linear start counter decoded by
`index_from_flat` (SSOT). Documents the `∏(sᵢ−wᵢ+1)` window-count theorem with
proof. Evidence tier: count theorem across shapes, row-major content,
full-window-equals-original, transposed/strided zero-copy correctness,
double-ended meet-once, zero/oversize rejection (6 tests, 92 core_tests green).
remaining §A: `IxDyn` (ADR 0002). Delivered 0.26.1:
`Array`/`ArrayViewMut::indexed_iter_mut` -> `IndexedIterMut` (ndarray
`indexed_iter_mut` parity), yielding `([usize; N], &mut T)` in logical
row-major order. The iterator is `DoubleEndedIterator`+`ExactSizeIterator` and
rejects layouts whose logical offsets are not provably disjoint before yielding
mutable references. Evidence tier: value-semantic mutation by index,
transposed-view index parity, double-ended meet-once, and alias rejection in
core/iteration. Delivered 0.26.0: logical-order element iteration
(`Array`/`ArrayView::iter`/`indexed_iter` -> `ElementIter`/`IndexedIter`,
`IntoIterator for &ArrayView`; ndarray `iter`/`indexed_iter` parity), both
`DoubleEndedIterator`+`ExactSizeIterator`, strided/transposed logical order via
the view strides; shared `elem_at` (SSOT). Refactored `application/iter.rs` into
a vertical `application/iter/{axis,element,mod}.rs` leaf hierarchy with stable
public paths (all AxisIter consumers, incl. leto-ops 156 ops_tests, green).
Evidence tier: row-major/transposed-order oracles, indexed pairs, double-ended
meet-once + rev-equals-reverse, `&view` for-loop, empty (7 tests). Remaining §A
iterator gap: `windows`/`lanes` (sliding windows + 1-D lane views, GAT lending
follow-up); remaining §A: `IxDyn` (ADR 0002). Delivered 0.25.0: whole-array
argmin/argmax (`argmin_all`/`argmax_all`) in `leto` core
`application/reduction/min_max.rs` (ndarray-stats
`argmin`/`argmax` parity), returning the const-generic `[usize; N]` multi-index
of the global extremum; first-occurrence tie-break; one shared `arg_reduce_all`
kernel (SSOT). Evidence tier: rank-1/rank-2 multi-index oracles, tie-break,
value-agrees-with-`min_all`/`max_all` cross-check, empty rejection (5 new tests,
23 reduction lib tests green). Promotes the argmin/argmax parity row to Verified;
the §A array/stats surface now has only `IxDyn` (ADR 0002) and the full iterator
surface audit open. Delivered 0.24.0: covariance and Pearson correlation
(`covariance`/`pearson_correlation`) in `leto` core
`application/statistics/`, following the ndarray-stats / numpy `rowvar = true`
contract. Evidence tier: theorem/proof sketches in rustdoc plus closed-form
sample/population oracles, diagonal == `var_axis`, symmetry, perfect +/-1
correlation, normalized-covariance identity, and exact empty/ddof rejection
(7 tests). Delivered 0.23.0: quantile and median reductions
(`quantile_all`/`median_all`/`quantile_axis`/`median_axis`) with an
`Interpolation` strategy enum (Linear/Lower/Higher/Nearest/Midpoint) in `leto`
core `application/reduction/quantile.rs` (ndarray-stats / numpy parity). One
shared `quantile_of_slice` kernel backs both whole-array and per-axis paths
(SSOT); axis path reuses one `out_size × axis_len` scratch buffer. Evidence
tier: fractional-rank theorem in rustdoc plus closed-form analytical oracles for
every interpolation method, per-lane equivalence, and empty/range/NaN rejection
(7 tests). Also 0.23.0 [patch]: `var_axis` no longer allocates a redundant
per-output gather buffer (indexes the C-contiguous `mean_axis` result directly).
Delivered 0.22.0: variance and standard-deviation reductions
(`var_all`/`std_all`/`var_axis`/`std_axis`) in `leto` core with finite `ddof`
validation and a two-pass numerical-stability theorem in rustdoc. Evidence
tier: theorem/proof sketch plus closed-form, invalid-input, and ndarray
differential tests. Delivered 0.21.0: unpivoted symmetric indefinite
`U D Uᵀ` factorization (`udu_decompose`, `MatrixDecompose::udu`) with
determinant, solve, and inverse helpers in `linalg/udu/{mod,decompose,solve}.rs`.
Evidence tier: theorem/proof sketch in rustdoc plus value-semantic tests for
reconstruction, determinant/solve/inverse parity against nalgebra, and invalid
contract rejection. Delivered 0.20.0: fluent rank-2 LA trait layer (ADR 0003)
consolidating the ndarray strided-array and nalgebra matrix-method models onto
the existing `Array2`/`ArrayView2` — `MatrixProduct`/`MatrixNorm`/
`MatrixDecompose`/`MatrixSolve` blanket-impl'd via the `AsMatrixView` bridge,
each method a zero-cost delegator to the existing free-function kernel (no kernel
duplicated; operators still deferred per ADR 0001). Differential tests
(`tests/ops/matrix_traits.rs`, 6) assert method == kernel == nalgebra/ndarray
plus a strided transposed-receiver case; 4 doctests; full ndarray/nalgebra
completeness program in `docs/completeness/`. Also in 0.20.0: elementwise
operators on `Array` (ADR 0004, supersedes ADR 0001) — `&a op &b`, `&a op
scalar` (sealed `ScalarOperand`), `-&a`, in leto core as the allocating
convenience tier; `*` is elementwise (matmul stays a method); 7 differential
tests in `tests/core/arithmetic.rs`. Dependency-resolution note re-verified:
`--locked` gates PASS (lock satisfies the floating themis spec); only fresh
`cargo generate-lockfile` is blocked because hermes (`efac0454`) and mnemosyne
(`1e014d25`) both pin unpinned `themis ^0.8.0` transitively — a coordinated
themis-0.9 co-evolution (upstream fixes already pushed) deferred to avoid
regressing the tuned matmul (gap_audit §D). Prior 0.19.7 [patch]. Delivered this cycle: Hermes fused multi-row AXPY consumed
by dense row-blocked matmul and direct Hermes pinned to the pushed provider
revision; post-0.19.7 generic 4x4 registered dense tiles rejected and removed
after benchmark regression. Result
parity remains covered for LU solve/determinant/inverse, symmetric eigenvalues,
Cholesky lower factors, singular values, and reverse-last-axis reductions.
Performance parity remains mixed: reverse reductions are faster than ndarray;
the current dense matmul audit finds Leto slower at 64x64, near ndarray at
128x128, and faster at 256x256. The post-0.19.7 128-row batched Hermes
row-panel AXPY path remains in the canonical dense route.
Remaining open: non-unit truly strided
reductions still row-walk (per-lane accumulators needed); melinoe ThreadCached
consolidation filed.

Stage A1 progress: norms (0.8.0), LU/solve/det/inv (0.9.0), QR + least
squares (0.10.0), Cholesky factor/solve/det/inv (0.12.0), thin SVD for
tall/square full-column-rank matrices (0.13.0), eigenvalues-only symmetric
Jacobi (0.14.0), wide full-row-rank SVD support (0.14.1), and
rank-deficient singular values (0.14.2), rank-revealing SVD/pseudoinverse,
non-symmetric eigenvalues, Hessenberg, bidiagonalization, full-pivot LU,
column-pivoted QR, trace/rank/Kronecker, and unpivoted UDU all delivered with
value-semantic reconstruction, identity, or differential parity checks.
Remaining nalgebra surface: Schur vectors/quasi-triangular form, pivoted
symmetric-indefinite factorization, matrix functions, and consumer-driven
fixed-size/geometry decisions.
Array-statistics surface: variance/std, quantile/median, and
covariance/correlation are closed for the current ndarray-stats parity rows.

Stage A2 progress: indexed zip parity (0.11.0) delivered through
`indexed_zip_mut_with` and `indexed_zip2_mut_with`, closing the current
`Zip::indexed` Apollo/Coeus migration blocker.

Parallel cross-repo track: Coeus CPU consolidation onto coeus-leto; the shared
GPU substrate `hephaestus` (atlas ADR 0001, wgpu + composed cuda-oxide/cutile)
consumed by coeus MS-60+ Stage D and apollo Stage D4; apollo ndarray retirement.

## Atlas ndarray replacement readiness [arch]
- [x] [minor] Route default thin SVD through implicit-shift bidiagonal QR
  (`svd_via_bidiagonal`) and remove the former Gram-backed SVD leaf; values-only
  `singular_values` now reuses bidiagonal reduction without U/V factor
  accumulation. Verification: `cargo fmt --check`; `cargo clippy --workspace
  --all-targets --all-features -- -D warnings`; `cargo nextest run --workspace
  --all-features` (384 tests); `cargo test --doc --workspace --all-features` (5
  doctests); `cargo doc -p leto -p leto-ops --all-features --no-deps`;
  criterion SVD and singular-values benchmark groups; `git diff --check`.
  Follow-up 2026-07-05: full workspace docs no longer hit the tracked
  `numpy 0.23.0` rustdoc ICE because `leto-python` is excluded as a Rust doc
  target while remaining checked/tested as a PyO3 extension crate.
- [x] [patch] Complete Stage C2 Hermes SIMD coverage audit for leto-ops hot kernels. Current coverage: dense elementwise slice ops and dense sum/dot/min/max route through Hermes via `Scalar`; matmul remains scalar because the current Hermes public surface lacks a zero-allocation scalar-AXPY/fused row-update provider. Rejected measured candidates: const-generic dense blocking regressed matmul (`64x64` ~48.5 µs, `256x256` ~3.37 ms); generic `mul_add` regressed matmul (`64x64` ~245.6 µs, `256x256` ~12.5 ms). Verification: focused matmul tests passed during both experiments; regressing source changes reverted; final gate run recorded in CHANGELOG/backlog.
- [x] [minor] Extend cache-line micro-tiling to unary `map_into` strided fallbacks (serial + parallel) through the shared `TileGeometry`/`line_elements` policy. Value tests: cache-line-sized transposed f64 `map_into` exact logical output; strided zero-sized input maps without divide-by-zero. Criterion: transposed unary `map_into` 57.631 µs (56.477–58.379 µs CI) → 35.303 µs (34.221–36.468 µs CI), −38.7% median with non-overlapping confidence intervals. Contiguous `map_into` remains within observed run-to-run noise. Version: 0.15.0.
- [x] [patch] Split `leto-ops::singular_values` from the full-vector `svd_decompose` contract so finite rank-deficient matrices return zero singular values through the smaller Gram-matrix eigenvalue path while `svd_decompose` still rejects rank-deficient inputs. Verification: `cargo metadata --no-deps --locked --format-version 1`; `cargo fmt --check`; `cargo check --workspace --all-features --locked`; `cargo test --workspace --all-features --locked`; `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`; `cargo doc --workspace --exclude leto-python --all-features --no-deps --locked`; `git diff --check`.
- [x] [patch] Generalized `leto-ops::svd_decompose`/`singular_values` from tall-or-square full-column-rank inputs to all full-rank thin SVD shapes, adding the wide full-row-rank `A A^T` path and deriving right singular vectors with `V = A^T U Σ^-1`. Verification: `cargo metadata --no-deps --locked --format-version 1`; `cargo fmt --check`; `cargo check --workspace --all-features --locked`; `cargo test --workspace --all-features --locked`; `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`; `cargo doc --workspace --exclude leto-python --all-features --no-deps --locked`; `git diff --check`.
- [x] [patch] All Leto package manifests now default both `parallel` and `mnemosyne-memory`; `leto` maps Mnemosyne memory to its existing Mnemosyne-backed storage implementation, `leto-ops` forwards memory into `leto`, and `leto-python` forwards both provider features to its Rust dependencies. Verification: manifest audit confirmed every package default includes both feature contracts; `cargo metadata --no-deps --locked`; `cargo fmt --check`; `cargo check --workspace --all-features`; `cargo test --workspace --all-features`; `cargo clippy --workspace --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude leto-python --all-features --no-deps`.
- [x] [minor] Add `leto-ops` eigenvalues-only symmetric Jacobi entry points (`symmetric_eigenvalues_jacobi`, `symmetric_eigenvalues_jacobi_with_tolerance`) that share the full decomposition's diagonalization logic through a monomorphized `RotationTarget` strategy and a zero-sized no-vector target. Verification: `cargo fmt --check`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo test --workspace --all-features`; `cargo nextest run --workspace --all-features`; `cargo doc -p leto -p leto-ops --all-features --no-deps`. Current note: the reopened `numpy 0.23` rustdoc ICE is resolved by the 2026-07-05 `leto-python` doc-target exclusion.
- [x] [minor] Add `leto-ops` thin SVD (`svd_decompose`, `svd_decompose_with_tolerance`, `singular_values`, `SvdDecomposition`) for tall/square full-column-rank matrices via `A^T A` + symmetric Jacobi; unsupported wide or rank-deficient inputs reject explicitly. Verification: `cargo fmt --check`; `cargo test -p leto-ops --test ops_tests svd --all-features`; `cargo test -p leto-ops --all-features`; `cargo clippy -p leto-ops --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude leto-python --all-features --no-deps`; `cargo test --workspace --all-features`.
- [x] [minor] Add unpivoted symmetric indefinite `U D Uᵀ` factorization (`udu_decompose`, `MatrixDecompose::udu`, `UduDecomposition`) with determinant, solve, and inverse helpers. Verification: `cargo test -p leto-ops --test ops_tests udu --all-features`.
- [x] [minor] Add variance and standard-deviation reductions (`var_all`/`std_all`/`var_axis`/`std_axis`) with finite `ddof` validation. Verification: `cargo test -p leto --test core_tests variance --all-features`.
- [x] Repository structure exists: `leto`, `leto-ops`, and `leto-python`.
- [x] Core C/F-contiguous `Layout<const N: usize>` construction, offset lookup, slicing, transpose, and broadcast have value-semantic tests.
- [x] Core storage exists for borrowed slices, mutable borrowed slices, `Vec`, and feature-gated Mnemosyne allocation.
- [x] Core `Array`, `ArrayView`, and `ArrayViewMut` wrappers exist for const-rank layouts.
- [x] Basic elementwise binary ops, `sum`, and 2D `matmul` exist with value-semantic tests.
- [x] [patch] Added ndarray-style slicing with full-axis ranges, optional signed bounds, negative indices, negative strides, axis-dropping integer indices, inserted new axes, ellipsis expansion, and implicit trailing axes through `SliceArg` and `slice_with`.
- [x] [patch] Run `cargo fmt` and keep `cargo fmt --check` clean across all workspace crates.
- [x] [patch] Fixed `mnemosyne-alloc` feature compilation by importing the allocator trait surface used by `MnemosyneStorage`.
- [x] [patch] Fixed `MnemosyneStorage` initialization semantics: `new(len)` now requires `T: Default` and initializes elements; `from_slice` copies initialized elements; `Drop` runs element destructors before deallocation.
- [x] [patch] Make mutable broadcast writes structurally impossible when the resulting layout has zero-stride aliasing.
- [x] [patch] Replace negative-offset casts with checked signed offset validation before any `usize` conversion in `Layout::offset_of`, `Layout::min_max_offsets`, and sliced layout construction.
- [x] [patch] Add property tests for C/F offset formulas, transposes, reverse slices, composed slices, empty axes, singleton-axis broadcasts, and negative-stride storage spans.
- [x] [patch] Add validated `ArrayView::try_new` / `ArrayViewMut::try_new` constructors so externally supplied layouts cannot index past the backing slice.
- [x] [patch] Add overflow-checked shape product and storage-span validation through `Layout::checked_size`, `checked_min_max_offsets`, and `validate_storage_len`.
- [x] [patch] Collapse duplicated `add`/`sub`/`mul`/`div` traversal into one generic zero-cost binary map skeleton with operation ZSTs.
- [x] [patch] Add axis-aware reductions required by Apollo and Coeus: `sum_axis_into`, `mean_axis_into`, `min_axis_into`, `max_axis_into`, and caller-owned output variants.
- [x] [patch] Add allocating keep-dim axis reduction wrappers: `sum_axis`, `mean_axis`, `min_axis`, and `max_axis`.
- [x] [patch] Add ndarray-parity constructors used by Apollo: `zeros`, `from_elem`, `from_vec`, `from_shape_fn`, `from_shape_vec`, and `into_vec`.
- [x] [patch] Add row/column/axis iteration APIs with contiguous fast paths and strided fallbacks.
- [x] [patch] Add named rank-2 `rows`, `columns`, `rows_mut`, and `columns_mut` wrappers over the axis iterator APIs.
- [x] [patch] Add shape aliases or type aliases for `Array1`, `Array2`, `Array3`, `ArrayView1`, `ArrayView2`, `ArrayView3` if Apollo migration keeps rank-specific readability.
- [x] [patch] Add `map`, `map_into`, `mapv`-equivalent, and precision-conversion APIs without hidden widen-and-narrow computation.
- [x] [patch] Add ndarray differential tests for map-style contiguous/transposed traversal.
- [x] [patch] Add zip-map APIs without duplicating the shared binary/unary traversal strategy.
- [x] [patch] Add BLAS/matrixmultiply replacement gates: contiguous `matmul`, strided `matmul`, transposed inputs, caller-owned output, and differential tests against `ndarray`.
- [x] [patch] Add ndarray differential tests for keep-dim axis reductions over contiguous and transposed inputs.
- [x] [patch] Add Python output conversion that avoids `Vec` clone round-trips where NumPy ownership transfer or direct allocation is available.
- [x] [patch] Add Python boundary tests for value parity, shape validation, C-contiguous input, and rejected non-contiguous inputs.
- [x] [patch] Add representative Leto-side Apollo and Coeus migration fixtures for rank aliases, complex precision mapping, keep-dim reduction/broadcast, and dense matmul.
- [x] [patch] Add `CowStorage` so Leto can borrow Apollo/Coeus read-only buffers without copying and detach into owned storage on mutation.
- [x] [patch] Add `CowStorage::as_borrowed` and `as_owned` accessors so callers can inspect backing state without cloning or forcing detachment.
- [x] [patch] Split storage infrastructure into SRP leaf modules for traits, borrowed slices, owned vectors, Cow, and Mnemosyne allocation while preserving the public storage API.
- [x] [patch] Fix ndarray-to-Leto zero-copy view conversion for negative strides by preserving signed strides and anchoring the borrowed backing slice at the minimum physical address.
- [x] [patch] Add Apollo ndarray-validation contract coverage for constructors, C-order storage, transpose, broadcast, axis iteration, mutable views, owned ndarray round trips, negative-stride views, slice-with metadata, and storage-bound rejection.
- [x] [minor] Add Mnemosyne-backed owned constructors (`zeros_mnemosyne`, `from_mnemosyne_slice`) so Apollo can return Leto arrays with provider-owned allocation instead of ndarray-owned storage. Verified against ndarray C-order values and storage-bound rejection.
- [x] [patch] Fix reduction module rustdoc links so `cargo doc -p leto --features mnemosyne-alloc --no-deps` is warning-clean.
- [x] [patch] Match ndarray retained single-element range stride metadata by setting the sliced axis stride to `0` when `SliceArg::range` selects exactly one logical element; empty ranges keep their computed stride.
- [x] [patch] Add Apollo migration test coverage for Mnemosyne-backed Leto owned constructors as the first FFT replacement prerequisite.
- [x] [minor] Add indexed mutable zip traversal (`indexed_zip_mut_with`, `indexed_zip2_mut_with`) to cover ndarray `Zip::indexed`-style Apollo/Coeus position-aware call sites without allocation.
- [x] [patch] Add Apollo migration tests proving Leto can replace current `Array1`/`Array2`/`Array3` usage in FFT, DHT, NTT, NUFFT, SHT, WGPU verification, and Python bindings. Added explicit Apollo FFT three-axis mutable rank-1 lane slicing over rank-3 Leto arrays so ndarray-free 3D axis-pass mutation is covered. Verification: `cargo fmt --check`; `cargo test -p leto-ops --test migration_fixtures --all-features`; `cargo clippy -p leto-ops --all-targets --all-features -- -D warnings`; `cargo test --workspace --all-features`; `cargo doc --workspace --exclude leto-python --all-features --no-deps`.
- [x] [patch] Coeus migration tests covering tensor layout, broadcast, elementwise ops, reductions, matmul, and non-differentiable storage boundaries: DONE on the coeus side as `coeus-leto/tests/contract.rs` (cross-repo behavior contracts) plus `coeus-ops/tests/*_leto_diff.rs` and `coeus-tensor/tests/*_leto_diff.rs` differential suites (verified 2026-06-15).
- [x] [major] Retire the transitional `ndarray` compatibility feature after
  consumers migrate; retain `ndarray` only as Leto's differential oracle.
- [x] [minor] Publish Leto 0.40.0 after format, warning-denied Clippy, configured
  Nextest, doctest, Rustdoc, dependency, and SemVer gates pass.

## Gap analysis: ndarray/nalgebra replacement [arch]
- [x] [patch] Audit Leto against `ndarray` 0.16, `nalgebra`, Apollo usage, and
  Coeus backend requirements; record the 2026-06-10 baseline in `gap_audit.md`.
  That audit found partial Apollo migration and no Coeus Leto references; both
  consumer migrations are now complete, while the recorded layer boundary
  remains authoritative.
- [x] [patch] Sync README role, layer boundary, linear-algebra features, and replacement status with the audited state.

## Next increments (ordered)
- [x] [minor] Contiguous-slice view access (`as_slice`/`as_mut_slice` now offset-independent C-dense, `as_slice_memory_order`/`as_mut_slice_memory_order`, `is_c_contiguous`/`is_f_contiguous`/`is_contiguous` queries) — unblocks Apollo FFT hot kernels. Value tests: offset-contiguous subview, F-order block, strided-gap rejection, mutable offset-block write.
- [x] [patch] `map_inplace` (mapv_inplace analogue) and 1D `dot` (contiguous + strided). Value tests in `ops/unary_math.rs`.
- [x] [major] ADR: const-rank vs dynamic-rank boundary for Coeus integration — `docs/adr/0002-coeus-rank-boundary.md` (const-generic dispatch shim at the Coeus boundary; Leto stays const-rank).
- [x] [minor] Unary math-op ZST suite (`ExpOp`/`LnOp`/`SinOp`/`CosOp`/`SqrtOp`/`AbsOp`/`NegOp`/`RecipOp`/`PowfOp`) via `UnaryOp` + `unary_map`/`unary_map_into`, on the new segregated `RealScalar` trait. Routed through the existing traversal kernel.
- [x] [minor] `scalar_map`/`scalar_map_into` array–scalar arithmetic reusing `BinaryOp` markers.
- [x] [minor] Generalize `symmetric_eigen_jacobi` over `T: RealScalar` (native precision, no hidden widening). f32 genericity test added; f64 path unchanged.
- [x] [minor] Add `symmetric_eigenvalues_jacobi` for sorted eigenvalues without eigenvector allocation; implemented with a ZST no-vector rotation target and shared Jacobi diagonalization kernel.
- [x] [arch] std::ops operator overloading decision — `docs/adr/0001-elementwise-operator-overloading.md` (deferred; orphan rule; `scalar_map` covers the scalar case).
- [x] [minor] Broadcast-aware binary ops into caller-owned output layouts: `binary_map`/`add`/`sub`/`mul`/`div` broadcast each input to the output shape, preserve the equal-shape contiguous fast path, and reject zero-stride aliased mutable output layouts. Value tests cover dense and strided broadcast inputs; ndarray differential coverage validates broadcasted add.
- [x] [minor] `reshape`/`permute`/`to_contiguous`: dense row-major reshape/into_shape on layouts, arrays, and views; permute aliases over transpose; row-major materialization for strided/transposed/broadcasted arrays and views. Value tests and ndarray contract coverage added.
- [x] [minor] `concat`/`pad`/`split` (leto core `structure/`), batched rank-3 `matmul`, `cumsum`/`scan_axis`, seeded RNG (`uniform_with_seed`/`normal_with_seed`), and `zip2_mut_with` (3-operand). Value tests for each; RNG validated against closed-form mean/variance. `stack` deferred (needs `InsertAxis` rank helper — stable Rust lacks const-generic `N+1`).
- [x] [minor] `stack` via an `InsertAxis` rank helper mirroring `RemoveAxis` (rank `N -> N+1`, ranks 0..=7). Value tests: new leading/trailing axis, rank-2→3, transposed-input logical order, shape-mismatch rejection.
- [x] [patch] Leto-internal ndarray differential coverage for the new ops: `unary_map` (exp/sqrt), `scalar_map`, `concat`, `stack`, `batched_matmul` (per-batch ndarray dot), and `cumsum` (reference accumulate). `ops_tests` differential suite now 57 green.
- [x] [minor] Indexed zip parity: `indexed_zip_mut_with` and `indexed_zip2_mut_with` pass logical row-major `[usize; N]` coordinates into zip closures while preserving zero-copy view traversal and mutable-output alias rejection.
- [x] [arch] Push Leto rev 9d5a2bf (0.7.0) and verify consumers: Apollo (already pinned at 9d5a2bf) builds clean — `apollo-frft`/`apollo-gft` eigensolver consumers check green against the generic eigensolver. Coeus integration started — new `coeus-leto` const-rank dispatch shim (ADR 0002) committed+pushed (coeus cdaaeb9) with 6 cross-repo contract tests; leto/leto-ops pinned at 9d5a2bf.
- [x] [arch] Coeus consolidation: COMPLETE (verified 2026-06-15 against coeus
  HEAD `037fdd5`). coeus's CPU `BackendOps` (elementwise binary/unary, matmul,
  batched matmul, axis reductions, argmax/argmin, cumsum/suffix scan,
  concat/pad/split/stack, seeded RNG, to_contiguous/reshape/permute,
  cross-backend transfer, from_fn/eye/arange/linspace) all route through the
  `coeus-leto` const-rank dispatch shim (ADR 0002) into leto/leto-ops kernels,
  with cross-repo contract tests (`coeus-leto/tests/contract.rs`) and per-op
  differential tests (`coeus-ops/tests/*_leto_diff.rs`); coeus workspace 255
  tests green. Framing correction: `coeus-tensor` is NOT a duplicated layout
  layer to retire — it is the autodiff-integrated `Tensor`/COW wrapper over
  coeus-core's dynamic-rank layout, with CPU compute delegated to leto. The
  array-primitive duplication is what was retired (routed to coeus-leto); the
  tensor/autograd wrapper legitimately remains coeus-owned. coeus-specific NN
  kernels (conv/pool/attention/optimizers) and higher sparse formats/backends
  stay in coeus by the layer boundary; Leto owns narrow CPU sparse parity
  kernels such as CSR SpMV/SpMM. No leto-side capability gap remains for the CPU
  re-base.
- [x] [minor] Apollo internal FFT-kernel migration off ndarray using Leto's
  memory-order slice access. Apollo commit `324f380` exposes native Leto arrays
  across its transform families; its manifests and resolved Rust graph contain
  no `ndarray` or `ndarray-compat` edge.
- [ ] [arch] Stack-wide themis-0.9 re-pin cascade (downstream-blocking,
  meta/stack-owned). All leaf upstreams are pushed on themis-0.9
  (themis `7c38eb2` 0.9.11; mnemosyne `0174b80`; moirai `4aa94f1`; hermes
  `e6761ac` 0.9.9), and apollo already migrated. leto cannot move unilaterally:
  fresh `cargo generate-lockfile` fails because the pinned upstream revs
  cross-reference each other's OLD (themis `^0.8.0`) revs — e.g. hermes `e6761ac`
  still pins mnemosyne `1e014d25`. Resolution must re-pin + re-push in dependency
  order (themis → mnemosyne → moirai/hermes → leto → apollo/coeus); apollo only
  builds on 0.9.11 today via local path-patches that bypass the git revs. Until
  then leto stays on the themis-0.8.0 lock (`--locked` builds/tests pass;
  consumer rev-bumps to leto 0.24.0 wait on this cascade). See gap_audit §D.
- [x] [patch] Current Leto 0.5.0 artifact verification: `cargo fmt --check`; `cargo test --all-features`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude leto-python --all-features --no-deps`. Historical note: full workspace docs were previously blocked by the tracked `numpy 0.23`/rustdoc ICE in `leto-python`; 0.19.6 updates the Python FFI dependencies and rechecks full docs.
- [x] [patch] Add ndarray/nalgebra oracle validation gates for current linalg
  and reduction contracts. Verification: `oracle_parity` compares Leto LU,
  Cholesky, symmetric eigenvalues, singular values, and reverse reductions
  against nalgebra/ndarray with value-semantic assertions. Gates run:
  `cargo fmt --check`; `cargo test -p leto-ops --test ops_tests oracle_parity
  --all-features`; `cargo check -p leto-ops --benches --all-features`;
  `cargo clippy --workspace --all-targets --all-features -- -D warnings`;
  `cargo test --workspace --all-features`; `cargo nextest run --workspace
  --all-features`; `cargo doc --workspace --exclude leto-python
  --all-features --no-deps`; `cargo test --doc --workspace --all-features`;
  `git diff --check`.
- [x] [minor] Close dense matmul oracle performance gap: the historical 0.19.7
  gate consumed
  Hermes fused multi-row AXPY and improves Leto medians to 17.430 µs
  (64x64), 108.98 µs (128x128), and 1.0631 ms (256x256), but ndarray/nalgebra
  remain faster at 8.492/8.775 µs, 66.527/62.935 µs, and 495.95/505.35 µs.
  Rejected: removing the dense row-block zero-skip branch, RHS-column packing
  plus `Scalar::dot_slice`, replacing Hermes AXPY with a generic scalar row
  update, existing Hermes `tiled_gemm` for f64 dense matmul, reducing parallel
  row-block scheduling for small dense matrices, `MATMUL_ROW_BLOCK=16`, and
  first-shared-row output initialization. Rejected after 0.19.7:
  Hermes column-chunk `axpy_rows`, `MATMUL_ROW_BLOCK=64`, and row-block
  fused-branch/alpha-buffer hoisting, and generic 4x4 registered dense tiles.
  Added after 0.19.7: `hermes_simd::axpy_rows_batch` is consumed only for the
  measured 128-row dense regime (212.64 µs → 98.853 µs on the local themis-0.9
  stack); broad depth-batched routing was rejected after 64x64/256x256
  regression.
  Current corrective gate: `cargo fmt --check`; `cargo clippy --workspace
  --all-targets --all-features -- -D warnings`; `cargo test --workspace
  --all-features`; `cargo nextest run --workspace --all-features`; `cargo doc
  -p leto -p leto-ops --all-features --no-deps`; `git diff --check`. Full
  workspace docs are no longer blocked by the reopened `leto-python`/`numpy
  0.23` rustdoc ICE after the 2026-07-05 PyO3-extension doc-target exclusion.
  The 2026-07-23 current-build audit closes this evidence item without a
  production change: default-feature Leto measures 23.597/123.63/233.60 µs
  against ndarray 12.770/113.07/952.54 µs at 64/128/256; disabling parallelism
  measures 27.483/223.69/1.8522 ms. Flamegraph collection is blocked on this
  Windows session by missing dtrace and administrator-only blondie. A future
  kernel increment should target an allocation-controlled
  reusable packing scratch or a verified external micro-kernel provider with
  profile evidence.
- [x] [patch] Direct registry dependencies were audited and later aligned with
  the current NumPy FFI constraint: workspace manifests now use `ndarray` 0.16,
  `pyo3` 0.23, and `numpy` 0.23. The reopened `leto-python` rustdoc ICE is
  resolved without moving the FFI constraint: `leto-python` is no longer a Rust
  doc target, while full workspace docs still build. Full Git dependency update
  is still blocked upstream by
  Mnemosyne's `themis ^0.8.0` requirement vs Themis main 0.9.5.

## Naming decision [patch]
- [x] Keep `leto` as the crate name. Functionally, Leto is a non-differentiable shared strided-array substrate between Coeus and Apollo; mythologically, Leto bridges Coeus and Apollo as parent/child context. The name is appropriate if the crate remains the shared array/memory vocabulary, not an autodiff engine or spectral-transform crate.
