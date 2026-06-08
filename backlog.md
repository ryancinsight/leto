# Leto Work Backlog

## Replacement Position
- [x] [arch] Use `leto` as the Atlas shared N-dimensional strided-array and layout crate. It sits below Apollo and Coeus and above Mnemosyne/Moirai/Hermes. It should replace `ndarray` only after parity and verification gates are met.
- [x] [patch] Naming assessment: `leto` is appropriate. The crate's intended responsibility is the shared array substrate between Coeus and Apollo, matching both functionality and the existing mythological naming scheme. Rename only if the crate changes scope into autodiff/tensors proper or Apollo-specific signal arrays.

## Current Evidence
- [x] [patch] `cargo test --all-features` passes: 24 `leto` core tests and 15 `leto-ops` tests pass. Evidence tier: value-semantic unit tests.
- [x] [patch] Apollo scan confirms `ndarray` is still a public and internal dependency across many crates, including `Array1`/`Array2`/`Array3`, `zeros`, `from_shape_fn`, `from_vec`, `from_shape_vec`, `mapv`, shape checks, axis semantics, and Python `numpy` ownership conversion.
- [x] [patch] `cargo fmt --check` is clean after formatting the workspace.
- [x] [patch] `cargo clippy --all-targets --all-features -- -D warnings` is clean after fixing `mnemosyne-alloc` allocator use and public module docs.
- [x] [patch] `cargo test --all-features` is clean.
- [ ] [patch] Full `cargo doc --no-deps` is blocked by a rustdoc internal compiler error in the `leto-python`/`numpy-0.23.0` documentation path. `cargo doc --no-deps -p leto -p leto-ops` passes.

## Phase 1: Sound Core Layout and Storage [patch]
- [x] Add ndarray-style slicing for full-axis selection, optional signed range bounds, negative indices, negative steps, integer axis removal, new-axis insertion, ellipsis expansion, and implicit trailing axes. Verification: three value-semantic tests over rank-preserving, rank-dropping, rank-adding, reverse, ellipsis, and implicit-tail cases.
- [x] Replace unchecked negative-offset casts with checked signed arithmetic across `Layout` and `Array` validation. Verification: value-semantic tests cover valid negative strides, rejected negative physical offsets, and one-past-storage rejection.
- [x] Make externally constructed `ArrayView` and `ArrayViewMut` layouts bounds-checked against their backing slices through `try_new` constructors. Verification: invalid external layouts return `StorageError`.
- [x] Remove or constrain mutable broadcast views that introduce zero-stride write aliasing. Verification: mutable broadcast rejects aliasing expansion and permits same-shape non-aliasing writes.
- [x] Add overflow-checked shape product and stride multiplication for core constructors and derived layout validation. Remaining risk: property tests still need broad adversarial coverage over large dimensions and slice/broadcast composition.
- [ ] Add property tests for C/F layouts, negative strides, empty axes, singleton axes, transposes, slices, broadcasts, and offset ranges.
- [x] Fix `MnemosyneStorage` initialization semantics. `new(len)` requires `T: Default` and initializes elements; `from_slice` copies initialized values; `Drop` drops elements before deallocation.

## Phase 2: ndarray API Parity Required by Apollo [minor]
- [x] Add rank-specific aliases for `Array1`, `Array2`, `Array3` and corresponding view types. Verification: value test constructs `Array1` and `Array2` aliases and reads through views.
- [x] Add a stable `RankMarker` / `RemoveAxis` helper for rank-dropping shape and stride calculations over ranks 1 through 4. Verification: value tests cover rank-3 axis removal and out-of-bounds rejection.
- [x] Add `zeros`, `from_elem`, `from_vec`, `from_shape_fn`, `from_shape_vec`, and `into_vec` equivalents. Verification: value tests cover filled/generated/vector constructors, length mismatch rejection, and zero-copy contiguous `into_vec`.
- [x] Add axis iteration APIs that cover row/column traversal without forcing copies. Verification: value test iterates matrix rows as read-only subviews; mutable iterator rejects zero-stride aliasing layouts at construction.
- [ ] Add named row and column convenience wrappers after axis iterator ergonomics are settled.
- [x] Add `mapv`/typed conversion APIs for scalar storage used by Apollo verification and Python outputs. Verification: value tests cover caller-owned `map_into`, allocating `mapv`, explicit f64-to-f32 conversion, and strided transposed inputs.
- [x] Add mutable zip-map traversal for Apollo migration call sites. Verification: value tests cover contiguous shape-matched mutation, shape mismatch rejection, and strided transposed views.
- [ ] Add complex-storage map tests after complex scalar aliases and Apollo differential fixtures are added.
- [ ] Add caller-owned output variants for all constructors and operations used in Apollo to preserve zero-copy and allocation control.
- [ ] Add differential tests against `ndarray` for every Apollo-facing API before replacing a downstream crate dependency.

## Phase 3: Coeus Tensor Substrate Requirements [minor]
- [ ] Add shape/stride/layout contracts suitable for tensor batches, channels, and rank-generic model activations.
- [ ] Add broadcast semantics compatible with tensor elementwise operations, including no mutable aliasing.
- [x] Add reductions over axes with keep-dim output modes required by Coeus: `sum_axis_into`, `mean_axis_into`, `min_axis_into`, and `max_axis_into`. Verification: value tests cover row/column reductions, strided transposed inputs, shape mismatch rejection, and empty-axis behavior.
- [ ] Add allocating convenience wrappers for axis reductions only after storage constructors are complete.
- [ ] Add matmul coverage for transposed inputs, batched 2D cases, and caller-owned output.
- [ ] Keep Leto non-differentiable. Coeus owns autodiff graph, gradient storage, and optimizer state; Leto owns layout/storage/views only.

## Phase 4: Operations, Performance, and Architecture [minor]
- [x] Replace duplicated elementwise functions with one generic binary traversal kernel selected by ZST operation markers. Verification: direct `binary_map::<AddOp>`/`binary_map::<MulOp>` tests and transposed strided-view elementwise test.
- [x] Extract shared logical flat-index conversion helpers for core constructors and leto-ops traversals. Verification: all constructor, map, elementwise, and reduction tests pass after the split.
- [ ] Add contiguous fast paths and strided fallback benchmarks for elementwise ops, reductions, and matmul.
- [ ] Verify Moirai scheduling uses bounded work partitioning without raw-pointer aliasing hazards.
- [ ] Integrate Hermes SIMD through sealed scalar/vector traits, not ad hoc per-operation dispatch.
- [ ] Keep Mnemosyne allocation optional and feature-gated; no downstream Apollo/Coeus crate should need allocator-specific types in public domain structs.

## Phase 5: Python and Interop [minor]
- [ ] Keep Python as a thin PyO3/NumPy boundary over Rust operations.
- [ ] Resolve or route around the `numpy-0.23.0` rustdoc ICE for `leto-python` without weakening Rust crate documentation gates.
- [ ] Replace current Python result construction that clones through `Vec` after computation.
- [ ] Add Python tests for shape validation, C-contiguous input, rejected non-contiguous inputs or zero-copy strided support, and value parity with NumPy.

## Apollo Migration Gate [arch]
- [ ] Add Leto as a Git workspace dependency in Apollo only after a pushed Leto revision passes all default and all-feature gates.
- [ ] Migrate one low-risk Apollo crate first, preferably a verification-only or WGPU verification path, and keep differential tests against `ndarray`.
- [ ] Migrate public Apollo APIs only after compatibility/migration notes are in Apollo CHANGELOG because replacing `ndarray::Array*` public types is a breaking API change.
- [ ] Remove Apollo's workspace `ndarray` dependency only after all crate manifests and Python bindings no longer expose or construct `ndarray` arrays except under a temporary compatibility feature.
