# ADR 0009: Automatic sparsity support (CSR, SpMV/SpMM, density dispatch)

- Status: Accepted
- Date: 2026-06-16
- Class: [minor] (additive surface: a new sparse subsystem and an
  automatic-dispatch matmul variant; no change to existing dense APIs)

## Context

The directive is to "automatically support sparsity to improve performance."
A matrix is *sparse* when most entries are zero; storing and operating only on the
`nnz` nonzeros turns dense matrix–vector work `Θ(m·n)` into `Θ(nnz)` and
sparse×dense product `Θ(m·n·k)` into `Θ(nnz·k)` — an order-of-magnitude win once
density `nnz/(m·n)` is small.

Leto had no sparse support. The dense `matmul` already skips exact-zero LHS
pivots (`if lhs_value == ZERO { continue }`), but that is per-call zero-checking,
not a sparse representation: the storage and the iteration are still dense
`Θ(m·n·k)`. Upstream, `hermes-simd` exports SIMD `spmv_csr`/`spmv_bcoo`/
`spmv_sellp` over a `CsrData` view, providing a potential accelerated backend.

## Decision

Add `leto-ops::application::sparse`, a deep vertical leaf hierarchy:

```
sparse/
  mod.rs    CsrMatrix<T: Scalar> + from_dense/from_parts/to_dense/nnz/density/
            as_parts + the CSR-correctness + O(nnz) complexity theorem & proof
  spmv.rs   y = A·x          (O(nnz), contiguous-x fast path)
  spmm.rs   C = A·B          (O(nnz·k), SIMD via Scalar::axpy_slice)
```

and `matrix::matmul_auto` — the automatic dispatch.

1. **Representation: CSR (Compressed Sparse Row).** CSR is the canonical
   row-oriented sparse format and the natural fit for row-major leto storage and
   for `y = A·x` / `C = A·B` (row-at-a-time). `CsrMatrix<T>` stores `values`,
   `col_indices`, `row_ptr` with validated invariants (monotone `row_ptr`,
   in-range strictly-increasing columns per row).

2. **Automatic compression: `from_dense`.** Scans a dense view once (`O(m·n)`,
   handling strided/negative-stride views) and retains only nonzeros. This is the
   "automatic" detection: compress once, then every kernel over the result is
   nonzero-proportional.

3. **Kernels are `*_into` (SSOT) + thin allocating wrappers (DRY).** The
   row-scale-accumulate in `spmm` is dispatched through `Scalar::axpy_slice`, so
   it inherits the existing SIMD path on contiguous rows — one authoritative
   contraction path shared with dense matmul.

4. **Automatic density dispatch: `matmul_auto`.** Scans the LHS density once and
   routes to `spmm` when density `≤ SPARSE_DENSITY_THRESHOLD = 0.1` (contiguous
   row-major output), else to dense `matmul`. The threshold is derived from the
   cost model: dense is `Θ(m·s·n)`, sparse is one `O(m·s)` compression plus
   `Θ(density·m·s·n)`; ignoring the sub-dominant compression, sparse beats dense
   by ≈ `1/density`, discounted by the CSR gather's larger per-flop constant. `0.1`
   is conservative — the sparse path strictly wins (measured ~17× at `0.05`,
   below) and the dense majority case pays only the `O(m·s)` scan, never
   regressing.

5. **Zero-copy backend seam: `as_parts`.** Borrows `(values, col_indices,
   row_ptr)` so a future `spmv_csr`-style SIMD backend (DIP, mirroring the dense
   `SimdStrategy`) can be slotted in without copying.

### Evidence

- **Correctness**: the module theorem proves CSR exactly represents `A` and that
  SpMV/SpMM are `Θ(nnz+m)`/`Θ(nnz·k+m·k)` vs dense `Θ(m·n)`/`Θ(m·n·k)`. Backed by
  differential tests against the dense reference (round-trip, strided views,
  shape rejection, `matmul_auto` ≡ dense on both branches).
- **Performance**: `sparse_compare` benchmark, f64, 256² LHS at 5% density × 32
  dense columns — dense `matmul` 343 µs vs sparse `spmm` 20.3 µs (~17×).

## Rejected alternatives

- **Sparse as a `Storage` backend** (`Array<T, SparseStorage, 2>`). The `Storage`
  trait is a flat indexable buffer; CSR's three-array structure does not fit
  element-by-linear-offset access. A dedicated `CsrMatrix` type is cleaner than
  forcing CSR through the dense layout abstraction.
- **Auto-detect inside dense `matmul`'s hot path.** Burdens every dense product
  with a density scan and complicates the hot kernel. Kept as the separate,
  opt-in `matmul_auto` so dense callers are unaffected.
- **Per-call routing to `hermes_simd::spmv_csr`.** Verified blocker: `CsrData`
  uses `&[i32]` indices, leto's CSR uses `usize`; a per-call `usize→i32`
  conversion is `O(nnz+m)`, comparable to the SpMV work, so it negates the SIMD
  gain. `spmv_csr` is already callable (the export is unconditional — no
  `hermes-simd` feature exists for it), but a *worthwhile* routing requires
  caching i32 indices natively in `CsrMatrix`, gated on a measured SpMV speedup —
  and SpMM already gets SIMD via `axpy_slice`, so only SpMV's gather would
  benefit. Deferred as a measured/coordinated change, not forced unverified.

## Consequences

- Callers gain transparent sparsity: `matmul_auto` exploits a sparse operand with
  no API change; explicit `CsrMatrix` + `spmv`/`spmm` serve the compress-once,
  reuse-many workflow.
- Follow-ups (tracked): native-i32 + SIMD `spmv_csr` backend behind the `as_parts`
  seam; additional formats (CSC for column ops, COO for assembly) only when a
  consumer needs them (YAGNI).
