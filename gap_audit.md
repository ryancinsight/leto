# Leto Gap Audit: ndarray / nalgebra Replacement for Atlas

Audit date: 2026-06-10. Evidence tier: codebase scan of `leto` (rev fd1d87b),
`D:/atlas/repos/apollo`, `D:/atlas/repos/coeus`, and upstream Atlas crates.
Counterparts: `ndarray 0.16`, `nalgebra` (already removed from Apollo).

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
| Contiguous-slice access on views (`as_slice`, `as_slice_mut`, memory-order variant) | `as_slice_memory_order_mut`, `is_standard_layout` | Apollo FFT butterfly kernels require contiguous mutable slices (~20 call sites) | [minor] |
| Multi-array zip (3+ operands) and `Zip::indexed` | `Zip::from(..).and(..).and(..)` | Apollo precision-downgrade and scaling paths (~20 sites use 2-operand; some need 3) | [minor] |
| `mapv_inplace` / in-place unary mutation | `mapv_inplace` | Apollo normalization (1/N scaling) (~5 sites) | [patch] |
| Reshape / `into_shape` on contiguous arrays | `into_shape_with_order` | Apollo (low frequency), Coeus `reshape` (required) | [minor] |
| Scalar–array elementwise ops (array + scalar, array * scalar) | `&a + 1.0`, `mapv` shortcuts | Apollo scaling, Coeus bias/scale paths | [minor] |
| std::ops operator impls on arrays/views (`Add`, `Sub`, `Mul`, `Div`, `Neg`) | operator overloads | Ergonomics for both consumers; std-trait integration mandate | [minor] |
| `concat`/`stack` along axis | `ndarray::concatenate`, `stack` | Coeus `cat()`; Apollo validation builders | [minor] |
| Dynamic-rank escape type at I/O boundaries | `IxDyn` | Apollo generic-over-dimension helpers (~30 sites use `Array<T, D>`); Coeus layout is dynamic-rank | [major] decision: const-rank adapters vs a `DynArray` boundary type |
| 1D dot / vector ops | `Array1::dot` | Apollo, Coeus | [patch] |
| Elementwise unary math suite (`exp`, `ln`, `sin`, `cos`, `sqrt`, `abs`, `neg`, `powf`) as named ZST ops | `mapv` with std float fns | Coeus `UnaryOp` enum (17 math/activation variants build on these) | [minor] |
| `cumsum` / prefix scans along axis | (ndarray lacks native; Coeus has) | Coeus `cumsum`, `suffix_sum` | [minor] |
| Random constructors (uniform/normal, seeded) | `ndarray-rand` | Coeus init (`Xorshift64`, Box-Muller); keep deterministic, seed-based | [minor] |
| Pad / split along axis | (manual in ndarray) | Coeus shape ops | [minor] |
| Batched (rank-3) matmul | (via einsum/manual) | Coeus batched contraction — only if the boundary decision places it in leto | [minor] |

Non-goals confirmed: conv/pool/attention/optimizer kernels, sparse formats
(COO/CSR, SpMV/SpMM), autodiff — these stay in Coeus. GPU buffers stay
behind Coeus's `ComputeBackend`.

## B. Gaps vs nalgebra (linear algebra)

Apollo's nalgebra removal is complete; remaining gaps are forward-looking
for Coeus/consumer needs, not blocking any current consumer.

| Gap | nalgebra counterpart | Status |
| --- | --- | --- |
| Symmetric eigensolver | `SymmetricEigen` | **Closed** — `symmetric_eigen_jacobi` (+ tolerance variant), f64, Jacobi rotations |
| Generic eigensolver over `T: Scalar` | `SymmetricEigen<T>` | Open — current impl is f64-only and returns `Vec<f64>`; violates generic-first authorship; [minor] |
| LU / solve / inverse / determinant | `LU`, `try_inverse` | Open — no consumer driver yet; defer until a consumer files the requirement |
| QR / Cholesky / SVD | `QR`, `Cholesky`, `SVD` | Open — same deferral rule |
| Norms (L1/L2/Frobenius) | `norm`, `norm_squared` | Open — cheap; [patch] when needed |
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

Blocking sub-gaps for step 1: broadcast-aware binary writing through an
output layout (Coeus passes `a_layout`, `b_layout`, `c_layout` — leto's
`binary_map` currently requires shape-matched views), unary ZST op suite,
reshape/permute/to_contiguous, concat/pad/split, batched matmul, seeded RNG
fill.

## D. Residual Risk Register

Update 2026-06-10 (v0.3.0): several §A/§B gaps closed — see CHANGELOG and the
two ADRs in `docs/adr/`.

- Dynamic-rank boundary: DECIDED ([major]) in
  `docs/adr/0002-coeus-rank-boundary.md` — const-generic dispatch shim at the
  Coeus boundary, shim owned by Coeus, Leto stays const-rank. Phase 6 leto-side
  capabilities authored const-rank.
- `symmetric_eigen_jacobi`: CLOSED ([minor]) — now generic over `T: RealScalar`,
  native precision, no hidden widening. Residual: no wider-accumulator variant;
  consumers needing higher working precision than storage convert first
  (explicit). f16/bf16 transcendentals use the documented f32 fallback.
- Contiguous-slice view access: CLOSED — `as_slice`/`as_mut_slice` are now
  offset-independent C-dense; `as_slice_memory_order`/`as_mut_slice_memory_order`
  expose F-order/offset blocks. Apollo hot-kernel migration still unproven end
  to end (boundaries exist; internal FFT compute still on ndarray), but the
  named blocker is removed.
- std::ops operator overloading: DEFERRED ([arch]) in
  `docs/adr/0001-elementwise-operator-overloading.md` (orphan rule). `scalar_map`
  covers array–scalar arithmetic; no consumer blocked.
- Coverage of new ops: value-semantic tests added (unary math, scalar_map, dot,
  map_inplace, memory-order slices, f32 eigensolver). No new ndarray differential
  oracle yet for the unary math suite or scalar_map — add before Apollo/Coeus
  consumer dependency updates (tracked in checklist next-increments).
- `leto-python` rustdoc ICE via `numpy 0.23` still open (tracked in backlog).
- Differential coverage: ndarray oracle covers map/reductions/matmul; no
  oracle yet for future unary suite, concat/stack, RNG (use closed-form
  references for RNG, ndarray for the rest).
- Evidence tier of this audit: codebase scan + existing test suites; no new
  proofs or benchmarks performed in this audit.
