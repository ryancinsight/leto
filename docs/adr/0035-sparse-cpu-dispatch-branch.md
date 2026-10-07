# ADR 0035: CPU dispatch branch for sparse forward ops

- Status: Proposed
- Date: 2026-10-07
- Refs: atlas ADR 0069 (S4); leto ADR 0002 (coeus rank boundary), ADR 0009
  (sparsity); coeus `CtcOps` CPU-impl precedent; leto `CsrView`/`CscView`.

## Context

`coeus-ops::sparse::{spmv, spmm}` are `B: Backend`-generic own-kernel loops:
raw-pointer `parallel_for` traversals over `CsrTensor` (i64 indices). They are
live: `coeus-autograd` sparse linalg calls `spmm`, and differential tests pin
both against a reference on the Sequential and Moirai backends.

`leto-ops` owns validated CSR/CSC kernels, now including borrowed
`CsrView`/`CscView` entries — zero-copy, no rank match needed (sparse shapes
are fixed rank 1/2). The `coeus-leto` bridge sparse module (`CsrDispatch` plus
copying dispatch) exists only to bridge borrowed-to-owned and has zero
production callers.

Bounds today (`CpuAddressableStorage(Mut)`, host storages only) already
exclude device backends: only Sequential (sequential) and Moirai
(thread-parallel) qualify. Leto kernels are sequential-SIMD, so the Moirai
path may trade thread parallelism for SIMD (benchmark question, below).

## Decision

1. Delete the bridge sparse module (`CsrDispatch`, `spmv_into`,
   `spmm_into`, doctests) — dead, superseded by `CsrView`.
2. Rewire `coeus-ops::sparse::{spmv, spmm}` bodies to the leto view kernels,
   keeping signatures, bounds shape, and the `alloc_on`/writes-every-output
   contract: shapes to leto views via the existing `to_leto_view(Mut)` plus
   `CsrView::from_slices`; errors via `map_leto_error` (the `ctc_error`
   precedent: map each variant, no catch-all swallowing).
3. Translate i64 indices to `&[usize]` by validated transmute, zero-copy: a
   static `size_of::<i64>() == size_of::<usize>()` assertion plus
   `CsrView::from_slices` validation *after* the cast. Soundness: the cast is
   bitwise over same-width integers, and any negative index becomes huge and
   is rejected by the column-range check. The proof obligation lives in a code
   comment at the cast site.
4. Keep the `B: Backend` + `CpuAddressableStorage(Mut)` bounds as-is — do
   NOT tighten to `CpuBackend`. The sparse entries are free functions, not
   `BackendOps` trait methods, so there is no per-backend impl selection to
   branch on; tightening would only ripple `CpuBackend` into
   `coeus-autograd`'s generic sparse nodes for an identical exclusion set (no
   device backend satisfies the storage bounds today — device exclusion stays
   exactly as strong as it is). Revisit via `BackendOps` trait-ification if a
   device sparse path ever exists. `T` gains `+ leto_ops::Scalar` (additive,
   the standard DIP cost; the autograd sparse bounds move in the same slice).
5. Backward (`spmm_backward_values/dense`) stays in coeus untouched:
   autodiff-owned, and leto has no backward concept.
6. Gate: the existing differential suites (sequential + moirai vs reference)
   must pass unmodified. Bitwise equality is expected — both sides accumulate
   in row-nonzero order; any ulp deviation is investigated, never loosened
   silently.
7. Accept sequential leto kernels on Moirai initially; record the benchmark
   follow-up: if wide-matrix Moirai regresses vs `parallel_for`, revisit
   row-chunk parallelism (needs kernel row-range support — explicitly out of
   scope here).

## Alternatives

- Keep the own kernels: rejected — duplicates validated provider kernels and
  keeps ~130 lines of unsafe pointer loops beside safe provider code.
- Convert indices per call (`Vec<usize>` allocs): rejected — reintroduces the
  O(nnz) copies S4(a) removed; kept as fallback if the transmute proof fails
  review.
- Migrate `CsrTensor` to usize indices: rejected for this slice — breaking
  storage change across tensor/autograd/serde; revisit if the transmute ever
  bites.
- Tighten to `B: CpuBackend` now: rejected — for free functions (no
  per-backend impls) it buys only a different compile error while rippling
  into autograd's generic bounds; the storage bounds already exclude devices.
- Generic `B: Backend` with a runtime branch: rejected — impossible without
  specialization, and unnecessary: the storage bounds are the compile-time
  branch. Device kernels get their own entries (plus trait-ification) when a
  device caller exists.
- Chunked parallel dispatch now: rejected — needs kernel row-range parameters;
  benchmark first per (7).

## Consequences

- Coeus sparse forward becomes CPU dispatch over the leto SSOT; the unsafe
  traversal loops and the dead bridge module are deleted (net negative lines).
- Autograd sparse linalg and the differential suites are the acceptance gate
  and must pass unmodified.
- A future device sparse path starts from new entries (plus `BackendOps`
  trait-ification) instead of aspirational genericity; the storage bounds
  remain the compile-time CPU/device branch until then.
