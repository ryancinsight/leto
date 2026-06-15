# Leto Gap Audit: ndarray / nalgebra Replacement for Atlas

Audit date: 2026-06-12. Evidence tier: codebase scan of `leto` (0.19.6),
`D:/atlas/repos/apollo`, `D:/atlas/repos/coeus`, current docs.rs pages for
`ndarray 0.17` and `nalgebra`, and upstream Atlas crates. Counterparts:
`ndarray 0.17`, `nalgebra` (already removed from Apollo).

## Consumer Position

- **Apollo** (spectral transforms): partially migrated. Pins
  `leto rev=fd1d87b` with `["std", "ndarray-compat"]`; exposes
  `forward_leto`/`inverse_leto` boundaries on FFT, CZT, DHT, NUFFT, SHT,
  Radon, STFT; nalgebra removed (FrFT/GFT eigendecomposition now uses
  `leto_ops::symmetric_eigen_jacobi`, GFT adjacency uses `leto::Array2<f64>`).
  ndarray remains the internal CPU compute substrate and differential oracle.
- **Coeus** (tensor/autodiff, burn replacement): zero references to leto
  today. Coeus owns its own `coeus-core` storage traits (`Storage`,
  `StorageMut`, COW), a sealed `ComputeBackend` (associated
  `DeviceBuffer<T>`/`KernelDescriptor`/`DispatchFuture<T>`), dynamic-rank
  `Layout`, and CPU (Moirai) + wgpu + CUDA backends. Replacing its array
  backend with leto is an [arch] integration, not a drop-in.

## Layer Boundary Decision (proposed, [arch])

Leto owns the non-differentiable array substrate: layout/strides, storage,
views, slicing, broadcasting, elementwise binary/unary math, reductions,
matmul (incl. batched), shape ops (concat/pad/split), and dense linear
algebra. Coeus owns autodiff, NN kernels (conv, pool, attention), optimizer
fusion, sparse formats, and device (GPU) backends. Apollo owns transform
kernels. FFT stays in Apollo; Coeus already routes `fft_1d` there.

## A. Gaps vs ndarray 0.16 (Apollo-facing)

Present and verified: const-rank `Array/ArrayView/ArrayViewMut` (+ rank
aliases 1–3), C/F layouts, ndarray-style `SliceArg` slicing, transpose,
broadcast, axis iteration, `zeros`/`ones`/`from_elem`/`from_vec`/
`from_shape_vec`/`from_shape_fn`/`into_vec`, `map_into`/`mapv`/`map`,
`zip_mut_with`, sum/mean/min/max (all + keep-dim axis), argmin/argmax, 2D
matmul, CoW storage, Mnemosyne storage, ndarray-compat conversions.

| Gap | ndarray counterpart | Consumer driver | Class |
| --- | --- | --- | --- |
| Contiguous-slice access on views (`as_slice`, `as_slice_mut`, memory-order variant) | `as_slice_memory_order_mut`, `is_standard_layout` | Apollo FFT butterfly kernels require contiguous mutable slices (~20 call sites) | Closed |
| Multi-array zip (3+ operands) and `Zip::indexed` | `Zip::from(..).and(..).and(..)`, `Zip::indexed` | Apollo precision-downgrade, scaling, and position-aware paths | Closed (`zip2_mut_with`, `indexed_zip_mut_with`, `indexed_zip2_mut_with`) |
| `mapv_inplace` / in-place unary mutation | `mapv_inplace` | Apollo normalization (1/N scaling) (~5 sites) | Closed |
| Reshape / `into_shape` on contiguous arrays | `into_shape_with_order` | Apollo (low frequency), Coeus `reshape` (required) | Closed |
| Scalar–array elementwise ops (array + scalar, array * scalar) | `&a + 1.0`, `mapv` shortcuts | Apollo scaling, Coeus bias/scale paths | Closed |
| Broadcast-aware binary ops into caller-owned output | broadcasted elementwise ops | Coeus passes `a_layout`, `b_layout`, `c_layout`; Apollo validation and scale paths | Closed |
| std::ops operator impls on arrays/views (`Add`, `Sub`, `Mul`, `Div`, `Neg`) | operator overloads | Ergonomics for both consumers; std-trait integration mandate | Deferred by ADR 0001; current scalar/binary map APIs cover driven cases |
| `concat`/`stack` along axis | `ndarray::concatenate`, `stack` | Coeus `cat()`; Apollo validation builders | Closed (`concat`; `stack` via `InsertAxis` rank helper) |
| Dynamic-rank escape type at I/O boundaries | `IxDyn` | Apollo generic-over-dimension helpers (~30 sites use `Array<T, D>`); Coeus layout is dynamic-rank | Decided by ADR 0002: consumer-owned const-rank dispatch adapters; Leto stays const-rank |
| 1D dot / vector ops | `Array1::dot` | Apollo, Coeus | Closed |
| Elementwise unary math suite (`exp`, `ln`, `sin`, `cos`, `sqrt`, `abs`, `neg`, `powf`) as named ZST ops | `mapv` with std float fns | Coeus `UnaryOp` enum (17 math/activation variants build on these) | Closed |
| `cumsum` / prefix scans along axis | (ndarray lacks native; Coeus has) | Coeus `cumsum`, `suffix_sum` | Closed (`scan_axis`, `cumsum`, fwd/rev, CumSum/CumProd) |
| Random constructors (uniform/normal, seeded) | `ndarray-rand` | Coeus init (`Xorshift64`, Box-Muller); keep deterministic, seed-based | Closed (`uniform_with_seed`, `normal_with_seed`) |
| Pad / split along axis | (manual in ndarray) | Coeus shape ops | Closed (`pad`, `split`) |
| Batched (rank-3) matmul | (via einsum/manual) | Coeus batched contraction — boundary decision places it in leto | Closed (`batched_matmul`, batch broadcast) |

Non-goals confirmed: conv/pool/attention/optimizer kernels, sparse formats
(COO/CSR, SpMV/SpMM), autodiff — these stay in Coeus. GPU buffers stay
behind Coeus's `ComputeBackend`.

## B. Gaps vs nalgebra (linear algebra)

Apollo's nalgebra removal is complete; remaining gaps are forward-looking
for Coeus/consumer needs, not blocking any current consumer. nalgebra's
documented decomposition surface includes Schur/Hessenberg, symmetric
eigendecomposition, SVD, LU, QR, and Cholesky; Leto only admits the subset with
a named Atlas consumer driver.

| Gap | nalgebra counterpart | Status |
| --- | --- | --- |
| Symmetric eigensolver | `SymmetricEigen` | **Closed** — `symmetric_eigen_jacobi` (+ tolerance variant), generic over `T: RealScalar`, native precision, Jacobi rotations |
| Symmetric eigenvalues-only path | `SymmetricEigen::eigenvalues` / eigenvalue access | **Closed in 0.14.0** — `symmetric_eigenvalues_jacobi` (+ tolerance variant) shares the Jacobi diagonalization kernel through a monomorphized `RotationTarget`; the eigenvalues-only path uses a zero-sized no-vector target and avoids eigenvector storage |
| LU / solve / inverse / determinant | `LU`, `try_inverse` | **Closed** — `lu_decompose`, `solve`, `det`, and `inv`, generic over `T: RealScalar`; CFDrs dense solver driver |
| QR + least squares | `QR` | **Closed** — Householder `qr_decompose` and `solve_least_squares`; CFDrs least-squares driver |
| Cholesky | `Cholesky` | **Closed** — SPD `cholesky_decompose` and solve; CFDrs SPD driver |
| Thin full-rank SVD | `SVD` subset | **Closed through 0.14.1** — `svd_decompose` supports tall/square full-column-rank and wide full-row-rank matrices; full-vector decomposition rejects rank-deficient inputs explicitly |
| Rank-deficient singular values | `SVD::singular_values` subset | **Closed in 0.14.2** — `singular_values` computes the smaller Gram-matrix spectrum and returns zero singular values without constructing missing null-space vectors |
| Full rank-revealing SVD / pseudoinverse | `SVD`, pseudo-inverse helpers | Open — [major], requires ADR before implementation |
| Norms (L1/L2/Frobenius) | `norm`, `norm_squared` | **Closed** — `NormKind` ZSTs with `norm_l1`, `norm_l2`, and `norm_max` |
| Non-symmetric eigenvalues/eigenvectors | `eigenvalues`, `complex_eigenvalues`, Schur | Open only with a named consumer driver; current policy rejects speculative surface |
| Small fixed-size matrix/vector types | `Matrix3`, `Vector3` | Non-goal — const-rank `Array<T, S, 2>` covers the layout; no consumer driver |

Policy: linalg routines enter leto-ops only with a named consumer driver and
a differential oracle (ndarray-linalg/nalgebra as dev-dependency oracle, per
the existing ndarray-oracle pattern).

## C. Gaps vs Coeus backend integration ([arch])

Coeus's `ComputeBackend`/`Backend` traits and `coeus-tensor` duplicate
leto's layout/storage/traversal layer (both built on Mnemosyne + Moirai).
This is the structural-duplication trigger: shared logic in two repos
consolidates to the deepest common ancestor — leto.

Integration path (recorded as the plan of record in backlog Phase 6):
1. Leto provides the CPU array kernels Coeus's CPU backend dispatches to
   (unary math suite, broadcast-aware binary into caller-owned output,
   reductions incl. argmax/cumsum, matmul, concat/pad/split).
2. Coeus's `coeus-tensor` CPU storage/layout layer re-bases onto
   `leto::Layout`/`Storage` (or thin adapters), deleting the duplicate.
3. Coeus keeps `ComputeBackend` ownership, GPU backends, autodiff, NN
   kernels, sparse, optimizers.

Step 1 Leto-side capability gaps are closed: broadcast-aware binary writing
through an output layout, unary ZST op suite, concat/pad/split/stack, batched
matmul, seeded RNG fill, and indexed mutable zip traversal. Remaining work is
consumer-side Coeus re-base and Apollo migration verification.

## D. Residual Risk Register

Update 2026-06-13 (v0.19.7): §A indexed zip parity, the Stage A1
consumer-driven nalgebra surface, Stage C2 dense norm SIMD coverage, and
Stage C3 unary/binary/zip column-walk line micro-tiling are closed through
symmetric eigenvalues-only, LU, QR, Cholesky, norms, full-rank thin SVD,
rank-deficient singular values, Hermes-backed dense reductions, and
cache-line tiled strided elementwise traversal. The optional themis topology
dependency is wired through `leto_ops::CacheGeometry`; dense matmul now has a
measured fixed row-block kernel backed by Hermes fused multi-row AXPY;
reverse-last-axis whole-array reductions now borrow unit-stride physical row
slices. Current ndarray/nalgebra oracle tests cover LU solve/determinant/
inverse, symmetric eigenvalues, Cholesky lower factors, singular values, and
reverse-last-axis reductions. Criterion oracle comparison shows reverse
reductions faster than ndarray, but dense matmul is still slower than
ndarray/nalgebra: Leto 17.430 µs / 108.98 µs / 1.0631 ms for
64x64/128x128/256x256 vs ndarray 8.4923 µs / 66.527 µs / 495.95 µs and
nalgebra 8.7752 µs / 62.935 µs / 505.35 µs. Topology-adaptive tile sizing and
non-unit truly strided reductions remain open. See CHANGELOG and the two ADRs
in `docs/adr/`. Remaining work is cross-cutting: the Coeus re-base,
Apollo/Coeus consumer migration with differential coverage, and dense matmul
oracle performance parity; full rank-revealing SVD vectors and non-symmetric
eigen are demand-driven only.

- `stack` (rank `N -> N+1`): CLOSED ([minor]) — implemented via the `InsertAxis`
  rank helper (dual of `RemoveAxis`, ranks 0..=7). `concat`/`pad`/`split`/`stack`
  all closed.
- Dynamic-rank boundary: DECIDED ([major]) in
  `docs/adr/0002-coeus-rank-boundary.md` — const-generic dispatch shim at the
  Coeus boundary, shim owned by Coeus, Leto stays const-rank. Phase 6 leto-side
  capabilities authored const-rank.
- `symmetric_eigen_jacobi`: CLOSED ([minor]) — now generic over `T: RealScalar`,
  native precision, no hidden widening. Residual: no wider-accumulator variant;
  consumers needing higher working precision than storage convert first
  (explicit). f16/bf16 transcendentals use the documented f32 fallback.
- `symmetric_eigenvalues_jacobi`: CLOSED ([minor]) — sorted eigenvalues without
  eigenvector allocation, implemented by a zero-sized no-vector rotation target
  over the shared Jacobi diagonalization kernel. Evidence tier:
  value-semantic full-vs-values parity and strided-view tests.
- Contiguous-slice view access: CLOSED — `as_slice`/`as_mut_slice` are now
  offset-independent C-dense; `as_slice_memory_order`/`as_mut_slice_memory_order`
  expose F-order/offset blocks. Apollo hot-kernel migration still unproven end
  to end (boundaries exist; internal FFT compute still on ndarray), but the
  named blocker is removed.
- std::ops operator overloading: DEFERRED ([arch]) in
  `docs/adr/0001-elementwise-operator-overloading.md` (orphan rule). `scalar_map`
  covers array–scalar arithmetic; no consumer blocked.
- Indexed zip parity: CLOSED ([minor]) — `indexed_zip_mut_with` and
  `indexed_zip2_mut_with` provide `Zip::indexed`-style logical coordinates for
  one- and two-input mutable zip traversals.
- Stage C3 column-walk elementwise traversal: CLOSED for binary and unary
  `map_into` plus `zip_mut_with` ([patch]/[minor]) — all use shared cache-line
  `TileGeometry`.
  Evidence tier: value-semantic strided tests plus criterion differential
  timing before/after the optimization. Remaining cache-aware CPU kernel work
  is blocked matmul cache hierarchy selection and themis topology wiring.
- Stage C2 dense norm SIMD coverage: CLOSED for `norm_l1`/`norm_l2`/`norm_max`
  over dense memory-order slices. `norm_l1` and `norm_max` now route f32/f64
  through Hermes absolute-value reductions; reduced precision keeps the native
  scalar fallback. Evidence tier: value-semantic norm tests plus criterion
  in-run scalar-reference comparison. Remaining reduction work is truly
  non-dense strided reductions, which need per-lane partial accumulators.
- Coverage of new ops: value-semantic tests plus ndarray differential oracles
  now cover the unary math suite (`exp`/`sqrt`), `scalar_map`, `concat`,
  `stack`, `batched_matmul` (per-batch ndarray `dot`), and `cumsum` (reference
  accumulate), alongside the existing map/reduction/matmul differentials. RNG is
  validated against closed-form mean/variance (correct per policy, not ndarray).
  Remaining: differential coverage is leto-internal; consumer-side (Apollo/Coeus)
  migration tests are the next cross-repo step.
- `leto-python` rustdoc ICE via `numpy 0.23`: REOPENED after the FFI
  dependency alignment back to NumPy 0.23/PyO3 0.23. Full workspace docs fail
  in rustdoc intra-doc link resolution inside `numpy 0.23.0`; `cargo doc -p
  leto -p leto-ops --all-features --no-deps` passes.
- Differential coverage: ndarray oracle covers map/reductions/matmul, unary
  suite, concat/stack, batched matmul, and cumsum. RNG uses closed-form
  references. Indexed zip currently rests on value-semantic traversal tests.
- Oracle performance parity: reverse-last-axis `sum`/`norm_l2` is at parity or
  faster than ndarray on the recorded 256x256 benchmark. Dense matmul is not
  at parity with ndarray/nalgebra and blocks replacement-performance claims.
  The 0.19.7 Hermes fused multi-row AXPY path improved Leto medians from
  21.443 µs / 127.63 µs / 2.4357 ms to 17.430 µs / 108.98 µs / 1.0631 ms for
  64x64/128x128/256x256, but ndarray/nalgebra remain faster. The 0.19.2
  zero-skip branch-removal experiment was rejected after canonical dense
  256x256 instability/regression. The 0.19.3 packed-RHS dot and
  scalar-row-update experiments also regressed 128x128. The 0.19.4 Hermes
  `tiled_gemm` f64 dense path regressed 128x128, and small-matrix serial
  scheduling was slower than row-block parallelism. The 0.19.5
  `MATMUL_ROW_BLOCK=16` and first-shared-row output initialization experiments
  did not meet the release benchmark stability/performance gate. Post-0.19.7
  Hermes column-chunk `axpy_rows` regressed 64x64/128x128/256x256 and ended
  with `STATUS_ACCESS_VIOLATION`; `MATMUL_ROW_BLOCK=64` also regressed against
  the 32-row baseline. Row-block fused-branch/alpha-buffer hoisting produced
  no statistically significant 128x128 improvement and also ended with
  `STATUS_ACCESS_VIOLATION`. Generic 4x4 registered dense tiles regressed the
  oracle shapes and are not retained. Next work needs allocation-controlled
  reusable packing scratch or a verified external micro-kernel provider with
  profile evidence.
- Locked dependency resolution: `--locked` focused gates and
  `cargo generate-lockfile` are blocked by the current upstream Git dependency
  set: `mnemosyne-arena` requires `themis ^0.8.0`, while the resolved themis Git
  head reports `0.9.0`. Unlocked local verification passed for this patch; the
  upstream dependency pin mismatch remains a supply-chain gate item before a
  fully locked release.
- Evidence tier of this audit: codebase scan + existing test suites +
  criterion benchmark measurements. No machine-checked proof was performed.

## Leto rank-deficient singular-values parity [patch]
- Performed: split `leto-ops::singular_values` from the full-vector `svd_decompose` path. Singular-values-only now diagonalizes the smaller Gram matrix and maps near-zero eigenvalues to zero singular values for finite rank-deficient inputs.
- Architecture effect: Leto matches the common nalgebra singular-values surface for rank-deficient matrices without fabricating null-space singular vectors. `svd_decompose` keeps explicit rank-deficient rejection until a rank-revealing SVD contract exists.
- Evidence tier: value-semantic tall/wide rank-deficient singular-value tests. No machine-checked proof was performed.

## Leto wide thin SVD parity [patch]
- Performed: generalized `leto-ops::svd_decompose` and `singular_values` from tall/square full-column-rank inputs to all full-rank thin SVD shapes. Wide full-row-rank matrices now use `A A^T`, then derive right singular vectors via `V = A^T U Σ^-1`.
- Architecture effect: Leto closes the current wide-matrix SVD nalgebra-parity gap without a second API or downstream Apollo-specific adapter. Rank-deficient inputs remain explicit errors until a rank-revealing SVD contract is implemented.
- Evidence tier: value-semantic reconstruction and orthonormality tests. No machine-checked proof was performed.
