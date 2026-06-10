# Changelog

All notable changes to Leto are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
SemVer 2.0.0. Pre-1.0 minor bumps may include additive API surface.

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
