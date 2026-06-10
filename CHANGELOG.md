# Changelog

All notable changes to Leto are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
SemVer 2.0.0. Pre-1.0 minor bumps may include additive API surface.

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
