# Leto Work Backlog

## Atlas in-house replacement roadmap — leto slice [arch]

Cross-repo program to eliminate ndarray, nalgebra, rayon, tokio, std::simd, and
burn from the Atlas stack using monomorphized zero-cost in-house crates. SSOT
map: ndarray→leto, nalgebra→leto-ops linalg, rayon/tokio→moirai, std::simd→hermes,
burn→coeus, alloc→mnemosyne, capabilities→melinoe, GPU=wgpu+cuda-oxide behind
coeus `ComputeBackend`. leto owns the CPU array substrate and stays CPU-only; GPU
backends live in coeus/apollo and index leto-style host-side layout metadata.

### Stage A1 — nalgebra linalg completion (leto-ops `application/linalg/`)
Each routine generic over `T: RealScalar`, native-precision accumulation (wider
accumulator only via a trait-encoded associated type with numerical justification),
admitted only with a named consumer driver (coeus/apollo) and a differential
oracle (nalgebra / ndarray-linalg as dev-dependency). SRP leaf modules.
- [x] [patch] Vector/matrix norms over `RealScalar`: `NormKind` ZST markers (`NormL1`/`NormL2`/`NormMax`) through one generic `norm` traversal in `application/linalg/norms.rs`; `norm_l2` covers Euclidean (rank-1) and Frobenius (rank-2+) in one entry point. Eigensolver consolidated into `linalg/` (re-export paths stable). Verification: nalgebra differential oracle, strided layout-independence, empty-view, and exact f16 tests.
- [x] [minor] LU with partial pivoting (`linalg/lu.rs`): `lu_decompose`/`LuDecomposition<T>` (packed factors, pivots, parity) with `solve`, `det`, `inv` — generic over `RealScalar`, native precision. Driver: CFDrs `cfd-math`. Verification: nalgebra oracle, pivot parity, `inv·A=I`, `det(Aᵀ)=det(A)` via strided view, singular/non-finite rejection, f32 genericity.
- [x] [minor] QR (Householder) + least-squares solve (`linalg/qr.rs`): compact packed reflectors, Q never materialized, least-squares via reflector application + back-substitution. Oracle: nalgebra SVD (independent path) + LU cross-check + residual-orthogonality property.
- [x] [minor] Cholesky (SPD) factorization + solve (`linalg/cholesky.rs`): lower-triangle-only reads, constructive positive-definiteness verification. Oracle: nalgebra cholesky().l() + LU cross-check + strided symmetry invariance.
- [ ] [major] SVD (Golub–Kahan) + pseudoinverse; ADR before implementation.
- [ ] [minor] Non-symmetric eigensolver only if a consumer drives it.

### Stage A2 — ndarray consolidation (support coeus/apollo)
- [ ] [minor] Provide any CPU kernel `coeus-leto` needs to retire coeus's
  duplicate traversal (reductions incl. argmax/cumsum already present; add gaps
  as coeus integration surfaces them).
- [ ] [patch] Keep ndarray strictly a dev-dependency differential oracle; core
  crates never depend on it in production.

### Stage C2 — hermes SIMD coverage audit
- [ ] [patch] Audit leto-ops hot kernels (matmul inner loop, reductions, scans,
  unary math) to ensure they dispatch through hermes `SimdOps` rather than
  ad-hoc scalar loops; file hermes coverage requests for any missing op/dtype.

## Replacement Position
- [x] [arch] Use `leto` as the Atlas shared N-dimensional strided-array and layout crate. It sits below Apollo and Coeus and above Mnemosyne/Moirai/Hermes. It should replace `ndarray` only after parity and verification gates are met.
- [x] [patch] Naming assessment: `leto` is appropriate. The crate's intended responsibility is the shared array substrate between Coeus and Apollo, matching both functionality and the existing mythological naming scheme. Rename only if the crate changes scope into autodiff/tensors proper or Apollo-specific signal arrays.

## Current Evidence
- [x] [patch] `cargo test --all-features` passes: 34 `leto` core tests, 28 `leto-ops` tests, and 5 `leto-python` tests pass. Evidence tier: value-semantic, property, differential, PyO3 boundary, and downstream-shape migration fixture tests.
- [x] [patch] Apollo scan confirms `ndarray` is still a public and internal dependency across many crates, including `Array1`/`Array2`/`Array3`, `zeros`, `from_shape_fn`, `from_vec`, `from_shape_vec`, `mapv`, shape checks, axis semantics, and Python `numpy` ownership conversion.
- [x] [patch] `cargo fmt --check` is clean after formatting the workspace.
- [x] [patch] `cargo clippy --all-targets --all-features -- -D warnings` is clean after fixing `mnemosyne-alloc` allocator use and public module docs.
- [x] [patch] `cargo test --all-features` is clean.
- [x] [patch] `CowStorage` is available for Leto arrays that borrow read-only Apollo/Coeus inputs and clone only when mutable access is requested. Evidence tier: value-semantic tests assert pointer identity on read-only borrowed storage, source preservation after mutation, and owned-detach output values.
- [ ] [patch] Full `cargo doc --no-deps` is blocked by a rustdoc internal compiler error in the `leto-python`/`numpy-0.23.0` documentation path. `cargo doc --no-deps -p leto -p leto-ops` passes.

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
- [ ] Add contiguous fast paths and strided fallback benchmarks for elementwise ops, reductions, and matmul.
- [ ] Verify Moirai scheduling uses bounded work partitioning without raw-pointer aliasing hazards.
- [ ] Integrate Hermes SIMD through sealed scalar/vector traits, not ad hoc per-operation dispatch.
- [ ] Keep Mnemosyne allocation optional and feature-gated; no downstream Apollo/Coeus crate should need allocator-specific types in public domain structs.

## Phase 5: Python and Interop [minor]
- [ ] Keep Python as a thin PyO3/NumPy boundary over Rust operations.
- [ ] Resolve or route around the `numpy-0.23.0` rustdoc ICE for `leto-python` without weakening Rust crate documentation gates.
- [x] Replace current Python result construction that clones through `Vec` after computation. Verification: `leto-python` now transfers owned `VecStorage` with `Array::into_vec()` and `PyArray1::from_vec`, then reshapes without the former `as_mut_slice().to_vec()` clone path.
- [x] Add Python boundary tests for shape validation, C-contiguous input, rejected non-contiguous inputs, and value parity with NumPy-visible outputs. Verification: `leto-python` unit tests cover `add`, `sum`, `matmul`, shape mismatch rejection, and a real NumPy transposed non-contiguous input.

## Phase 6: Coeus Backend Consolidation [arch]
Source: `gap_audit.md` §C. Coeus (the Atlas burn replacement) carries a duplicate non-differentiable array layer (`coeus-tensor`/`coeus-core` layout, storage, COW, traversal) over the same Mnemosyne/Moirai substrate as Leto. Structural-duplication rule: consolidate to Leto. Coeus keeps `ComputeBackend`, autodiff, NN kernels (conv/pool/attention), optimizers, sparse formats, and GPU backends.
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
Source: `gap_audit.md` §A. Apollo already exposes `forward_leto`/`inverse_leto` boundaries; these items unblock replacing ndarray inside the kernels.
- [x] [minor] Add contiguous-slice access on views: `as_slice`/`as_mut_slice` (now offset-independent C-dense) plus `as_slice_memory_order`/`as_mut_slice_memory_order` and `is_c_contiguous`/`is_f_contiguous`/`is_contiguous` queries (named Apollo FFT butterfly blocker). Value tests cover offset-contiguous subviews, F-order blocks, strided-gap rejection, and mutable offset-block writes.
- [x] [patch] Add `map_inplace` in-place unary mutation (Apollo 1/N normalization sites); memory-order fast path, zero-stride aliasing rejected.
- [x] [patch] Add 1D `dot` (contiguous fast path + strided fallback, native-precision accumulation).
- [x] [minor] Add scalar–array elementwise ops: `scalar_map`/`scalar_map_into` reusing `BinaryOp` markers.
- [ ] [arch] std::ops operator overloading on arrays/views: DEFERRED, see `docs/adr/0001-elementwise-operator-overloading.md` (orphan rule; revisit when a consumer driver exists; `scalar_map` covers the scalar case meanwhile).
- [x] [minor] Add 3+-operand zip traversal: `zip2_mut_with` (one mutable output + two read inputs), the `Zip::from(out).and(a).and(b)` analogue. Verification: fused multiply-add and strided-input value tests.

## Phase 8: nalgebra Successor Policy [minor]
Source: `gap_audit.md` §B. Apollo's nalgebra removal is complete; this phase is demand-driven.
- [x] [minor] Generalize `symmetric_eigen_jacobi`/`SymmetricEigenDecomposition` over `T: RealScalar`; runs in native precision with no hidden widening (the wider-accumulator path is intentionally not introduced — a consumer needing higher working precision converts first). f32 genericity test added; f64 path unchanged. `RealScalar` is a segregated transcendental extension of `Scalar` (ISP).
- [ ] Policy: LU/QR/Cholesky/SVD/solve/norms enter leto-ops only with a named consumer driver and a differential oracle as dev-dependency; no speculative linalg surface.

## Apollo Migration Gate [arch]
- [x] Add Leto as a Git workspace dependency in Apollo only after a pushed Leto revision passes all default and all-feature gates. Apollo pins Leto by Git rev with `["std", "ndarray-compat"]` and exposes `forward_leto`/`inverse_leto` API boundaries on FFT, CZT, DHT, NUFFT, SHT, Radon, and STFT.
- [x] [minor] Replace Apollo's nalgebra dependency: FrFT/GFT eigendecomposition migrated to `leto_ops::symmetric_eigen_jacobi`; GFT adjacency storage migrated to `leto::Array2<f64>`.
- [x] Add representative Leto-side Apollo and Coeus migration fixtures before direct consumer updates. Verification: fixtures cover Apollo FFT-like rank/complex/precision shapes and Coeus reduction/broadcast/matmul shapes.
- [ ] Migrate one low-risk Apollo crate first, preferably a verification-only or WGPU verification path, and keep differential tests against `ndarray`.
- [ ] Migrate public Apollo APIs only after compatibility/migration notes are in Apollo CHANGELOG because replacing `ndarray::Array*` public types is a breaking API change.
- [ ] Remove Apollo's workspace `ndarray` dependency only after all crate manifests and Python bindings no longer expose or construct `ndarray` arrays except under a temporary compatibility feature.
