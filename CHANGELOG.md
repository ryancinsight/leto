# Changelog

All notable changes to Leto are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
SemVer 2.0.0. Pre-1.0 minor bumps may include additive API surface.

## [0.13.0] - 2026-06-11

Minor nalgebra-replacement increment for thin SVD.

### Added

- `leto-ops`: `SvdDecomposition`, `svd_decompose`,
  `svd_decompose_with_tolerance`, and `singular_values`.
- The initial SVD surface supports tall or square full-column-rank matrices.
  It derives the thin decomposition from `A^T A` plus the existing symmetric
  Jacobi eigensolver, returning `U`, descending singular values, and `V`.
  Wide and rank-deficient inputs are rejected explicitly.

### Tests

- Added value-semantic SVD coverage for reconstruction, closed-form diagonal
  singular values, strided input, f32 generic execution, and invalid input
  rejection.

## [0.12.0] - 2026-06-11

Minor nalgebra-replacement increment for SPD linear algebra.

### Added

- `leto-ops`: `CholeskyDecomposition::det`, `CholeskyDecomposition::inv`,
  `cholesky_solve`, `cholesky_det`, and `cholesky_inv`.
- Cholesky inverse reuses the same private forward/back substitution helper as
  `solve`, preserving one authoritative SPD solve path.

### Tests

- Added value-semantic Cholesky determinant and inverse coverage: determinant
  checks against the nalgebra oracle, convenience API parity, and `A·A⁻¹ = I`
  through Leto `matmul`.

## [0.11.3] - 2026-06-11

Patch performance increment for dense L2/Frobenius norms.

### Changed

- `leto-ops`: added a generic `Scalar::dot_slice` reduction hook. Native
  f32/f64 route through Hermes SIMD dot dispatch; reduced-precision f16/bf16
  keep the existing native scalar fallback.
- `leto-ops`: rank-1 `dot` and dense-slice `NormL2` now reuse the shared
  `dot_slice` contract, keeping one authoritative dot reduction path.

### Performance (criterion, recorded in benchmark_results.md)

- `reductions/norm_l2_64k`: 28.07 µs → 5.508 µs (**−80.0%, p < 0.05**).
- `reductions/norm_l2_transposed_256x256`: 28.67 µs → 5.550 µs
  (**−80.7%, p < 0.05**) when the transposed view exposes a dense memory
  slice. Negative-stride `norm_l2` remains on the row-walk fallback.

### Tests

- Existing norm and dot value-semantic tests cover the shared dot-slice path,
  including reduced-precision fallback and strided fallback coverage.

## [0.11.2] - 2026-06-11

Patch performance increment for strided whole-array reductions and norms.

### Changed

- `leto-ops`: strided whole-array `sum` and generic `norm` traversal now use
  `RowMajorTraversal`, computing each innermost row base once and walking by
  the last-axis stride. This removes the per-element row-major index
  decomposition from reverse-last-axis and transposed reduction fallbacks.
- `crates/leto-ops/benches/kernels.rs`: added criterion baselines for
  transposed and reverse-last-axis `sum`/`norm_l2` reductions.

### Performance (criterion, recorded in benchmark_results.md)

- First measured strided reduction baselines: transposed `sum` 40.73 µs,
  transposed `norm_l2` 28.67 µs, reverse-last-axis `sum` 30.55 µs, and
  reverse-last-axis `norm_l2` 30.21 µs.

### Tests

- Added negative last-axis stride value-semantic coverage for whole-array
  `sum` and norms.

## [0.11.1] - 2026-06-11

Patch performance increment for strided unary and binary map traversal.

### Changed

- `leto-ops`: strided `map_into`, `mapv`, and `binary_map` now traverse by
  innermost logical rows. Each row computes base offsets once and then advances
  by signed last-axis strides, reducing per-element index decomposition and
  offset multiplication on Apollo/Coeus strided views.
- Shared the row traversal shape/chunk calculation through
  `RowMajorTraversal` to avoid maintaining separate unary and binary copies of
  the same traversal policy.

### Performance (criterion, recorded in benchmark_results.md)

- `elementwise_add/transposed_256x256`: 1.206 ms → 49–51 µs (**−95.9%,
  23.7×, p < 0.05**); `contiguous_64k` statistically unchanged (p = 0.56);
  untouched kernels (matmul, reductions) verified unchanged.

### Tests

- Added negative last-axis stride differential tests against ndarray for unary
  `mapv` and binary `add`, covering the signed-stride row-walk path.

## [0.11.0] - 2026-06-11

Stage A2 ndarray-parity increment: indexed mutable zip traversal for
position-aware Apollo/Coeus migration paths.

### Added

- `leto-ops`: `indexed_zip_mut_with` — `zip_mut_with` plus logical
  row-major `[usize; N]` coordinates passed into the closure, matching the
  `ndarray::Zip::indexed` use case without allocation or runtime dispatch.
- `leto-ops`: `indexed_zip2_mut_with` — indexed three-operand mutable zip for
  caller-owned outputs with two read-only inputs.

### Tests

- Value-semantic indexed zip tests cover logical coordinate use on dense
  rank-2 arrays and transposed strided views, including three-operand indexed
  traversal.

## [0.10.0] - 2026-06-10

Stage A1 third increment: Householder QR + least squares, and Cholesky (SPD).

### Added

- `leto-ops`: `application/linalg/qr.rs` — `qr_decompose` (Householder, m ≥ n,
  compact packed form: R upper, reflector tails below the diagonal, heads/β
  alongside; `Q` is never materialized — solves apply reflectors directly,
  the fast and memory-lean form) and `QrDecomposition::solve_least_squares`
  (`Qᵀ·rhs` by reflector application + back-substitution; exact solve at
  m = n). Rejections: underdetermined shape, non-finite input, exactly-zero
  pivot-column norm (documented exact contract — near-deficiency is
  conditioning, not detected by unpivoted QR).
- `leto-ops`: `application/linalg/cholesky.rs` — `cholesky_decompose`
  (`A = L·Lᵀ`, reads only the lower triangle so symmetric storage works
  unchanged) and `CholeskyDecomposition::solve`. Positive-definiteness is
  verified constructively (non-positive pivot rejects). Driver: CFDrs
  `cfd-math` SPD paths.

### Tests

- QR: square solve cross-checked against LU, overdetermined least squares vs
  the independent nalgebra SVD oracle, residual-orthogonality optimality
  property (`Aᵀ(Ax−b) ≈ 0`), underdetermined/zero-column rejection.
- Cholesky: factor vs nalgebra `cholesky().l()`, solve vs LU, transposed
  strided-view symmetry invariance, indefinite/non-square rejection.
- Both: f32 genericity cross-check (QR vs Cholesky agreement).

## [0.9.0] - 2026-06-10

Stage A1 second increment: LU with partial pivoting, solve, determinant,
inverse. Driver: CFDrs `cfd-math` dense solver paths.

### Added

- `leto-ops`: `application/linalg/lu.rs` — `lu_decompose` (partial pivoting,
  packed unit-L/U factors, permutation + parity), `LuDecomposition<T>` with
  `solve` (forward/back substitution), `det` (parity × U diagonal), and `inv`
  (identity-column solves); plus `solve`/`det`/`inv` convenience entry points.
  Generic over `RealScalar`, native-precision elimination (no hidden
  widening). `det` of a singular matrix returns zero; `solve`/`inv` reject
  singular and non-finite inputs with distinct error reasons.

### Tests

- nalgebra differential oracle (`lu().solve`, `determinant`, `try_inverse`),
  pivot-parity coverage via zero leading pivot, `inv·A = I` value check,
  strided/transposed logical-value decomposition (`det(Aᵀ) = det(A)`),
  singular/non-square/non-finite rejection, and f32 genericity.

## [0.8.0] - 2026-06-10

Stage A1 (nalgebra replacement) first increment: norms, plus a vertical
`linalg` module consolidating dense linear algebra.

### Added

- `leto-ops`: `application/linalg/norms.rs` — `NormKind` ZST contract with
  `NormL1` (`Σ|x|`), `NormL2` (`sqrt(Σx²)`; Euclidean over rank-1, Frobenius
  over rank-2+ through one generic entry point), and `NormMax` (`max|x|`)
  markers; one generic `norm::<K, T, N>` traversal (memory-order contiguous
  fast path + strided logical fallback) with `norm_l1`/`norm_l2`/`norm_max`
  wrappers. Generic over `RealScalar`, native-precision accumulation.
- `leto-ops`: top-level re-export of `symmetric_eigen_jacobi_with_tolerance`.

### Changed

- `leto-ops`: the eigensolver moved into the new `application/linalg/` module
  (`linalg/eigen.rs`); all public re-export paths (`leto_ops::symmetric_eigen_jacobi`,
  `SymmetricEigenDecomposition`) are unchanged.

### Tests

- Norm differential oracle vs nalgebra (`DVector::norm`/`lp_norm`/`amax`,
  `DMatrix::norm` Frobenius), strided/transposed layout-independence,
  logical-selection-only strided slices, empty-view zero, and exact
  reduced-precision (3-4-5 in `f16`) coverage.

## [0.7.0] - 2026-06-10

### Added

- `leto`: `InsertAxis` rank helper (`domain/insert_axis.rs`) — the dual of
  `RemoveAxis`, mapping rank `N -> N + 1` at compile time on stable Rust through
  the shared `RankMarker` ZST, for ranks 0..=7 (output 1..=8).
- `leto`: `stack` (`application/structure/stack.rs`) — stacks equal-shaped
  rank-`N` views along a new axis (inserted at `0..=N`), producing rank
  `M = N + 1` C-contiguous output in logical row-major order. Output rank is
  resolved via `InsertAxis`; call as `stack::<T, N, M>(..)`.

### Tests

- ndarray differential oracles added for `unary_map` (exp/sqrt), `scalar_map`,
  `concat`, `stack`, `batched_matmul` (per-batch `dot`), and `cumsum`, raising
  the `leto-ops` differential suite to 57 tests.

### Notes

- Closes the `stack` item deferred in 0.6.0. The rank-preserving structural ops
  (`concat`/`pad`/`split`) remain unchanged.

## [0.6.0] - 2026-06-10

Phase 6/7 gap remediation: structural ops, batched contraction, scans, seeded
RNG, and multi-operand zip. All const-rank per ADR 0002.

### Added

- `leto`: structural array operations in a new `application/structure/` module —
  `concat` (along an existing axis), `pad` (per-axis before/after with a fill
  value), and `split` (zero-copy subviews along an axis). `concat`/`pad`
  allocate C-contiguous output and read inputs in logical (row-major) order, so
  strided/transposed inputs are handled correctly.
- `leto-ops`: `batched_matmul` for rank-3 `[B,M,K] x [B,K,N] -> [B,M,N]`, with
  batch broadcasting when either operand's batch dim is 1. Dispatches each batch
  to the rank-2 `matmul` kernel (one authoritative contraction).
- `leto-ops`: prefix/suffix scans in a new `application/scan.rs` — `ScanOp`
  trait with `CumSumOp`/`CumProdOp` markers, `ScanDirection` (Forward/Reverse),
  `scan_axis`/`scan_axis_into`, and `cumsum`/`cumsum_into` wrappers.
- `leto-ops`: deterministic seeded random constructors `uniform_with_seed` and
  `normal_with_seed` (Box-Muller), backed by a new `Xorshift64` PRNG domain
  type (`domain/rng.rs`). Validated against closed-form distribution statistics.
- `leto-ops`: `zip2_mut_with`, the three-operand analogue of `zip_mut_with`
  (`ndarray`'s `Zip::from(out).and(a).and(b)`).
- `leto-ops`: `RealScalar::from_f64` (construction-time conversion for sampling
  and constants).

### Notes

- `stack` (rank-increasing concat, rank `N -> N+1`) is deferred: stable Rust
  lacks const-generic rank arithmetic, so it needs an `InsertAxis` rank helper
  mirroring `RemoveAxis`. Tracked in backlog Phase 6.
- `batched_matmul` value tests cover explicit batches and batch broadcast; its
  per-batch path is the rank-2 `matmul` already differentially tested against
  `ndarray`.

## [0.5.0] - 2026-06-10

### Added

- `leto`: `Layout::reshape`, `Array::reshape`, `Array::into_shape`,
  `ArrayView::reshape`, and mutable reshape view variants for dense row-major
  layout reinterpretation without materializing storage.
- `leto`: `permute`/`permute_mut` named aliases over transpose semantics for
  arrays and views.
- `leto`: `to_contiguous` materialization for arrays and views, copying
  strided, transposed, or broadcasted layouts into canonical row-major storage.

## [0.4.0] - 2026-06-10

### Added

- `leto-ops`: `binary_map`/`add`/`sub`/`mul`/`div` now broadcast each input
  view to the caller-owned output layout when shapes are compatible, covering
  Coeus-style `[N, 1]` and `[1, C]` elementwise tensor paths without allocating
  broadcasted inputs.

## [0.3.0] - 2026-06-10

ndarray/nalgebra gap remediation toward Apollo hot-kernel and Coeus backend
readiness. See `gap_audit.md` for the full analysis and `docs/adr/` for the
two architectural decisions this release records.

### Added

- Offset-independent contiguity model on `Layout`: `is_contiguous` (dense in C
  or F order regardless of offset) and `is_c_dense`, alongside the existing
  canonical `is_c_contiguous`/`is_f_contiguous` (offset 0).
- Contiguity queries on `ArrayView`/`ArrayViewMut`: `is_c_contiguous`,
  `is_f_contiguous`, `is_contiguous`.
- Memory-order slice access on views: `as_slice_memory_order` (both views) and
  `as_mut_slice_memory_order` (mutable view), the `ndarray::as_slice_memory_order`
  analogue Apollo's in-place FFT butterfly kernels require. `as_slice`/
  `as_mut_slice` now expose dense C-order blocks at non-zero offsets (sliced and
  axis-iterated subviews).
- `leto-ops`: `RealScalar` trait (segregated transcendental extension of
  `Scalar`) implemented for `f32`/`f64` (native) and `f16`/`bf16` (documented
  `f32` fallback).
- `leto-ops`: named unary math operations as ZST/value-carrying markers
  (`ExpOp`, `LnOp`, `SinOp`, `CosOp`, `SqrtOp`, `AbsOp`, `NegOp`, `RecipOp`,
  `PowfOp`) routed through the shared traversal kernel via the `UnaryOp` trait
  and `unary_map`/`unary_map_into` entry points.
- `leto-ops`: `map_inplace` (the `ndarray::mapv_inplace` analogue).
- `leto-ops`: `scalar_map`/`scalar_map_into` reusing the `BinaryOp` markers for
  array–scalar broadcast arithmetic.
- `leto-ops`: rank-1 `dot` product (contiguous fast path + strided fallback,
  native-precision accumulation).

### Changed

- `symmetric_eigen_jacobi`/`symmetric_eigen_jacobi_with_tolerance` and
  `SymmetricEigenDecomposition` are now generic over `T: RealScalar` (previously
  `f64`-only with `Vec<f64>` eigenvalues), running in native precision with no
  hidden widening. The `f64` call sites are unchanged by inference.

### Notes

- std::ops operator overloading on `Array` is intentionally not added; the
  orphan rule prevents `leto-ops` from implementing foreign operator traits for
  the foreign `Array` type. See `docs/adr/0001-elementwise-operator-overloading.md`.
- The const-rank vs dynamic-rank boundary for Coeus integration is decided in
  `docs/adr/0002-coeus-rank-boundary.md` (const-generic dispatch shim at the
  Coeus boundary); implementation is tracked in backlog Phase 6.
