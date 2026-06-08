# Leto Work Backlog

## Replacement Position
- [x] [arch] Use `leto` as the Atlas shared N-dimensional strided-array and layout crate. It sits below Apollo and Coeus and above Mnemosyne/Moirai/Hermes. It should replace `ndarray` only after parity and verification gates are met.
- [x] [patch] Naming assessment: `leto` is appropriate. The crate's intended responsibility is the shared array substrate between Coeus and Apollo, matching both functionality and the existing mythological naming scheme. Rename only if the crate changes scope into autodiff/tensors proper or Apollo-specific signal arrays.

## Current Evidence
- [x] [patch] Default `cargo test` passes: 7 `leto` core tests and 3 `leto-ops` tests pass. Evidence tier: value-semantic unit tests.
- [x] [patch] Apollo scan confirms `ndarray` is still a public and internal dependency across many crates, including `Array1`/`Array2`/`Array3`, `zeros`, `from_shape_fn`, `from_vec`, `from_shape_vec`, `mapv`, shape checks, axis semantics, and Python `numpy` ownership conversion.
- [x] [patch] `cargo fmt --check` is clean after formatting the workspace.
- [x] [patch] `cargo clippy --all-targets --all-features -- -D warnings` is clean after fixing `mnemosyne-alloc` allocator use and public module docs.
- [x] [patch] `cargo test --all-features` is clean.

## Phase 1: Sound Core Layout and Storage [patch]
- [x] Add ndarray-style slicing for full-axis selection, optional signed range bounds, negative indices, negative steps, integer axis removal, new-axis insertion, ellipsis expansion, and implicit trailing axes. Verification: three value-semantic tests over rank-preserving, rank-dropping, rank-adding, reverse, ellipsis, and implicit-tail cases.
- [ ] Replace unchecked negative-offset casts with checked signed arithmetic across `Layout` and `Array` validation.
- [ ] Make externally constructed `ArrayView` and `ArrayViewMut` layouts bounds-checked against their backing slices.
- [ ] Remove or constrain mutable broadcast views that introduce zero-stride write aliasing.
- [ ] Add overflow-checked shape product and stride multiplication for all constructors and derived layouts.
- [ ] Add property tests for C/F layouts, negative strides, empty axes, singleton axes, transposes, slices, broadcasts, and offset ranges.
- [x] Fix `MnemosyneStorage` initialization semantics. `new(len)` requires `T: Default` and initializes elements; `from_slice` copies initialized values; `Drop` drops elements before deallocation.

## Phase 2: ndarray API Parity Required by Apollo [minor]
- [ ] Add rank-specific aliases or constructors for `Array1`, `Array2`, `Array3` and corresponding view types.
- [ ] Add `zeros`, `from_elem`, `from_vec`, `from_shape_fn`, `from_shape_vec`, `as_slice`, `as_mut_slice`, `shape`, `dim`, and `into_vec` equivalents.
- [ ] Add row, column, and axis iteration APIs that cover Apollo's 1D/2D/3D transform loops without forcing copies.
- [ ] Add `mapv`/typed conversion APIs for f64/f32/f16 and complex storage used by Apollo verification and Python outputs.
- [ ] Add caller-owned output variants for all constructors and operations used in Apollo to preserve zero-copy and allocation control.
- [ ] Add differential tests against `ndarray` for every Apollo-facing API before replacing a downstream crate dependency.

## Phase 3: Coeus Tensor Substrate Requirements [minor]
- [ ] Add shape/stride/layout contracts suitable for tensor batches, channels, and rank-generic model activations.
- [ ] Add broadcast semantics compatible with tensor elementwise operations, including no mutable aliasing.
- [ ] Add reductions over axes with keep-dim or shape-preserving output modes if Coeus requires training/evaluation reductions.
- [ ] Add matmul coverage for transposed inputs, batched 2D cases, and caller-owned output.
- [ ] Keep Leto non-differentiable. Coeus owns autodiff graph, gradient storage, and optimizer state; Leto owns layout/storage/views only.

## Phase 4: Operations, Performance, and Architecture [minor]
- [ ] Replace duplicated elementwise functions with one generic binary traversal kernel selected by ZST operation markers.
- [ ] Add contiguous fast paths and strided fallback benchmarks for elementwise ops, reductions, and matmul.
- [ ] Verify Moirai scheduling uses bounded work partitioning without raw-pointer aliasing hazards.
- [ ] Integrate Hermes SIMD through sealed scalar/vector traits, not ad hoc per-operation dispatch.
- [ ] Keep Mnemosyne allocation optional and feature-gated; no downstream Apollo/Coeus crate should need allocator-specific types in public domain structs.

## Phase 5: Python and Interop [minor]
- [ ] Keep Python as a thin PyO3/NumPy boundary over Rust operations.
- [ ] Replace current Python result construction that clones through `Vec` after computation.
- [ ] Add Python tests for shape validation, C-contiguous input, rejected non-contiguous inputs or zero-copy strided support, and value parity with NumPy.

## Apollo Migration Gate [arch]
- [ ] Add Leto as a Git workspace dependency in Apollo only after a pushed Leto revision passes all default and all-feature gates.
- [ ] Migrate one low-risk Apollo crate first, preferably a verification-only or WGPU verification path, and keep differential tests against `ndarray`.
- [ ] Migrate public Apollo APIs only after compatibility/migration notes are in Apollo CHANGELOG because replacing `ndarray::Array*` public types is a breaking API change.
- [ ] Remove Apollo's workspace `ndarray` dependency only after all crate manifests and Python bindings no longer expose or construct `ndarray` arrays except under a temporary compatibility feature.
