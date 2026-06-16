# Leto Development Checklist

Sprint phase: Execution. Target version: 0.34.0 [minor] (Cargo.toml bumped;
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
Delivered 0.28.0: zero-copy lane iteration
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
remaining §A: `IxDyn` (ADR 0002). Delivered 0.26.0:
logical-order element iteration
(`Array`/`ArrayView::iter`/`indexed_iter` → `ElementIter`/`IndexedIter`,
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
Performance parity remains mixed: reverse reductions are faster than ndarray,
and dense 64x64/128x128/256x256 matmul improved materially, including a
post-0.19.7 128-row batched Hermes row-panel AXPY path, but it remains slower
than ndarray/nalgebra.
Remaining open: close dense matmul oracle performance gap before claiming
replacement performance parity; non-unit truly strided
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
- [x] [patch] Complete Stage C2 Hermes SIMD coverage audit for leto-ops hot kernels. Current coverage: dense elementwise slice ops and dense sum/dot/min/max route through Hermes via `Scalar`; matmul remains scalar because the current Hermes public surface lacks a zero-allocation scalar-AXPY/fused row-update provider. Rejected measured candidates: const-generic dense blocking regressed matmul (`64x64` ~48.5 µs, `256x256` ~3.37 ms); generic `mul_add` regressed matmul (`64x64` ~245.6 µs, `256x256` ~12.5 ms). Verification: focused matmul tests passed during both experiments; regressing source changes reverted; final gate run recorded in CHANGELOG/backlog.
- [x] [minor] Extend cache-line micro-tiling to unary `map_into` strided fallbacks (serial + parallel) through the shared `TileGeometry`/`line_elements` policy. Value tests: cache-line-sized transposed f64 `map_into` exact logical output; strided zero-sized input maps without divide-by-zero. Criterion: transposed unary `map_into` 57.631 µs (56.477–58.379 µs CI) → 35.303 µs (34.221–36.468 µs CI), −38.7% median with non-overlapping confidence intervals. Contiguous `map_into` remains within observed run-to-run noise. Version: 0.15.0.
- [x] [patch] Split `leto-ops::singular_values` from the full-vector `svd_decompose` contract so finite rank-deficient matrices return zero singular values through the smaller Gram-matrix eigenvalue path while `svd_decompose` still rejects rank-deficient inputs. Verification: `cargo metadata --no-deps --locked --format-version 1`; `cargo fmt --check`; `cargo check --workspace --all-features --locked`; `cargo test --workspace --all-features --locked`; `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`; `cargo doc --workspace --exclude leto-python --all-features --no-deps --locked`; `git diff --check`.
- [x] [patch] Generalized `leto-ops::svd_decompose`/`singular_values` from tall-or-square full-column-rank inputs to all full-rank thin SVD shapes, adding the wide full-row-rank `A A^T` path and deriving right singular vectors with `V = A^T U Σ^-1`. Verification: `cargo metadata --no-deps --locked --format-version 1`; `cargo fmt --check`; `cargo check --workspace --all-features --locked`; `cargo test --workspace --all-features --locked`; `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`; `cargo doc --workspace --exclude leto-python --all-features --no-deps --locked`; `git diff --check`.
- [x] [patch] All Leto package manifests now default both `parallel` and `mnemosyne-memory`; `leto` maps Mnemosyne memory to its existing Mnemosyne-backed storage implementation, `leto-ops` forwards memory into `leto`, and `leto-python` forwards both provider features to its Rust dependencies. Verification: manifest audit confirmed every package default includes both feature contracts; `cargo metadata --no-deps --locked`; `cargo fmt --check`; `cargo check --workspace --all-features`; `cargo test --workspace --all-features`; `cargo clippy --workspace --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude leto-python --all-features --no-deps`.
- [x] [minor] Add `leto-ops` eigenvalues-only symmetric Jacobi entry points (`symmetric_eigenvalues_jacobi`, `symmetric_eigenvalues_jacobi_with_tolerance`) that share the full decomposition's diagonalization logic through a monomorphized `RotationTarget` strategy and a zero-sized no-vector target. Verification: `cargo fmt --check`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo test --workspace --all-features`; `cargo nextest run --workspace --all-features`; `cargo doc -p leto -p leto-ops --all-features --no-deps`. Current note: the `numpy 0.23` rustdoc ICE is reopened by the FFI dependency downgrade, so full workspace docs remain blocked.
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
- [x] [patch] Fix reduction module rustdoc links so `cargo doc -p leto --features mnemosyne-alloc,ndarray-compat --no-deps` is warning-clean.
- [x] [patch] Match ndarray retained single-element range stride metadata by setting the sliced axis stride to `0` when `SliceArg::range` selects exactly one logical element; empty ranges keep their computed stride.
- [x] [patch] Add Apollo migration test coverage for Mnemosyne-backed Leto owned constructors as the first FFT replacement prerequisite.
- [x] [minor] Add indexed mutable zip traversal (`indexed_zip_mut_with`, `indexed_zip2_mut_with`) to cover ndarray `Zip::indexed`-style Apollo/Coeus position-aware call sites without allocation.
- [x] [patch] Add Apollo migration tests proving Leto can replace current `Array1`/`Array2`/`Array3` usage in FFT, DHT, NTT, NUFFT, SHT, WGPU verification, and Python bindings. Added explicit Apollo FFT three-axis mutable rank-1 lane slicing over rank-3 Leto arrays so ndarray-free 3D axis-pass mutation is covered. Verification: `cargo fmt --check`; `cargo test -p leto-ops --test migration_fixtures --all-features`; `cargo clippy -p leto-ops --all-targets --all-features -- -D warnings`; `cargo test --workspace --all-features`; `cargo doc --workspace --exclude leto-python --all-features --no-deps`.
- [x] [patch] Coeus migration tests covering tensor layout, broadcast, elementwise ops, reductions, matmul, and non-differentiable storage boundaries: DONE on the coeus side as `coeus-leto/tests/contract.rs` (cross-repo behavior contracts) plus `coeus-ops/tests/*_leto_diff.rs` and `coeus-tensor/tests/*_leto_diff.rs` differential suites (verified 2026-06-15).
- [x] [minor] Add optional `ndarray` compatibility feature for differential tests and transitional conversions only; core crates must not depend on `ndarray`.
- [ ] [minor] Publish a pushed Git revision only after `fmt`, `clippy --all-targets --all-features -- -D warnings`, `cargo test --all-features`, docs, and differential ndarray parity tests pass.

## Gap analysis: ndarray/nalgebra replacement [arch]
- [x] [patch] Audit Leto against `ndarray` 0.16, `nalgebra`, Apollo usage, and Coeus backend requirements; record in `gap_audit.md` (2026-06-10). Findings: Apollo partially migrated (Git-pinned Leto, `forward_leto` boundaries, nalgebra removed via `symmetric_eigen_jacobi`); Coeus has zero Leto references and duplicates the layout/storage layer; layer-boundary decision recorded in `gap_audit.md` §C/`README.md`.
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
  kernels (conv/pool/attention/optimizers/sparse) stay in coeus by the layer
  boundary. No leto-side capability gap remains for the CPU re-base.
- [ ] [minor] Apollo internal FFT-kernel migration off ndarray using the new
  memory-order slice access (boundary `forward_leto`/`inverse_leto` APIs already
  in place). Apollo (HEAD `db76ca2`) still uses ndarray as its internal CPU
  compute substrate; leto boundaries exist but end-to-end kernel migration is
  apollo-owned work.
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
- [ ] [minor] Close dense matmul oracle performance gap: 0.19.7 consumes
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
  workspace docs are blocked by the reopened `leto-python`/`numpy 0.23`
  rustdoc ICE. Next kernel increment should target an allocation-controlled
  reusable packing scratch or a verified external micro-kernel provider with
  profile evidence.
- [x] [patch] Direct registry dependencies were audited and later aligned with
  the current NumPy FFI constraint: workspace manifests now use `ndarray` 0.16,
  `pyo3` 0.23, and `numpy` 0.23. This reopens the `leto-python` rustdoc ICE;
  keep full workspace docs recorded as blocked until the FFI constraint can
  move forward again. Full Git dependency update is still blocked upstream by
  Mnemosyne's `themis ^0.8.0` requirement vs Themis main 0.9.5.

## Naming decision [patch]
- [x] Keep `leto` as the crate name. Functionally, Leto is a non-differentiable shared strided-array substrate between Coeus and Apollo; mythologically, Leto bridges Coeus and Apollo as parent/child context. The name is appropriate if the crate remains the shared array/memory vocabulary, not an autodiff engine or spectral-transform crate.
