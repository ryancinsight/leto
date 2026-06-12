# Leto Development Checklist

Sprint phase: Execution. Target version: 0.16.1 [patch] (Cargo.toml/Cargo.lock
bumped; CHANGELOG synced). Delivered this cycle: hermes AXPY matmul rows +
dense memory-order sum (0.16.0, Stage C2 gate closed), zip line micro-tiling
(0.16.1). Next increment: hermes abs-sum/abs-max kernels would vectorize the
norm_l1/norm_max dense scalar fold (file in hermes Stage C2); matmul blocking
may be revisited only on top of the AXPY row kernel with criterion evidence.

Stage A1 progress: norms (0.8.0), LU/solve/det/inv (0.9.0), QR + least
squares (0.10.0), Cholesky factor/solve/det/inv (0.12.0), thin SVD for
tall/square full-column-rank matrices (0.13.0), eigenvalues-only symmetric
Jacobi (0.14.0), wide full-row-rank SVD support (0.14.1), and
rank-deficient singular values (0.14.2) all delivered with value-semantic
identity/reconstruction or full-vs-values parity checks. Remaining nalgebra
surface: full rank-revealing SVD vectors and any consumer-driven non-symmetric
eigensolver.

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
- [x] [minor] Add `leto-ops` eigenvalues-only symmetric Jacobi entry points (`symmetric_eigenvalues_jacobi`, `symmetric_eigenvalues_jacobi_with_tolerance`) that share the full decomposition's diagonalization logic through a monomorphized `RotationTarget` strategy and a zero-sized no-vector target. Verification: `cargo fmt --check`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo test --workspace --all-features`; `cargo nextest run --workspace --all-features`; `cargo doc -p leto -p leto-ops --all-features --no-deps`. Full workspace docs remain blocked by the tracked `numpy 0.23` rustdoc ICE in `leto-python`.
- [x] [minor] Add `leto-ops` thin SVD (`svd_decompose`, `svd_decompose_with_tolerance`, `singular_values`, `SvdDecomposition`) for tall/square full-column-rank matrices via `A^T A` + symmetric Jacobi; unsupported wide or rank-deficient inputs reject explicitly. Verification: `cargo fmt --check`; `cargo test -p leto-ops --test ops_tests svd --all-features`; `cargo test -p leto-ops --all-features`; `cargo clippy -p leto-ops --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude leto-python --all-features --no-deps`; `cargo test --workspace --all-features`.
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
- [ ] [patch] Add Coeus migration tests covering tensor layout, broadcast, elementwise ops, reductions, matmul, and gradient-adjacent non-differentiable storage boundaries.
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
- [ ] [arch] Coeus consolidation: route `coeus-ops` CPU backend through `coeus-leto` and retire the duplicated `coeus-tensor` traversal once parity is proven (tracked in coeus docs/backlog MS-59).
- [ ] [minor] Apollo internal FFT-kernel migration off ndarray using the new memory-order slice access (boundary `forward_leto`/`inverse_leto` APIs already in place).
- [x] [patch] Current Leto 0.5.0 artifact verification: `cargo fmt --check`; `cargo test --all-features`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude leto-python --all-features --no-deps`. Full `cargo doc --workspace --all-features --no-deps` remains blocked by the tracked `numpy 0.23`/rustdoc ICE in `leto-python`.

## Naming decision [patch]
- [x] Keep `leto` as the crate name. Functionally, Leto is a non-differentiable shared strided-array substrate between Coeus and Apollo; mythologically, Leto bridges Coeus and Apollo as parent/child context. The name is appropriate if the crate remains the shared array/memory vocabulary, not an autodiff engine or spectral-transform crate.
