# Leto Development Checklist

## Atlas ndarray replacement readiness [arch]
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
- [ ] [patch] Add Apollo migration tests proving Leto can replace current `Array1`/`Array2`/`Array3` usage in FFT, DHT, NTT, NUFFT, SHT, WGPU verification, and Python bindings.
- [ ] [patch] Add Coeus migration tests covering tensor layout, broadcast, elementwise ops, reductions, matmul, and gradient-adjacent non-differentiable storage boundaries.
- [x] [minor] Add optional `ndarray` compatibility feature for differential tests and transitional conversions only; core crates must not depend on `ndarray`.
- [ ] [minor] Publish a pushed Git revision only after `fmt`, `clippy --all-targets --all-features -- -D warnings`, `cargo test --all-features`, docs, and differential ndarray parity tests pass.

## Naming decision [patch]
- [x] Keep `leto` as the crate name. Functionally, Leto is a non-differentiable shared strided-array substrate between Coeus and Apollo; mythologically, Leto bridges Coeus and Apollo as parent/child context. The name is appropriate if the crate remains the shared array/memory vocabulary, not an autodiff engine or spectral-transform crate.
