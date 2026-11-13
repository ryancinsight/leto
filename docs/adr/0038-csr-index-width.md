<a id="adr-0038"></a>

# ADR 0038: Sparse formats store indices as a width-parameterized unsigned integer

Status: Proposed

Board item `LETO-CSR-INDEX-WIDTH-2026-09-27`, delivered in slices; it needs
`LETO-SPARSE-FORMAT-CONSOLIDATION-2026-11-13` (Decision 4).

## Context

`CsrMatrix`, `CscMatrix` and `CooMatrix` in `leto-ops`
(`application/sparse/`) and `CsrArray`, `CscArray` and `CooArray` in `leto`
(`infrastructure/sparse/`) store every index as `usize`; `col_indices`,
`row_ptr`, `as_parts`, `CsrView` and `CsrRow` expose `&[usize]`. On the
64-bit targets Leto ships this is 8 bytes per index.

Narrowing the index removes matrix-stream bytes; whether that removes time
depends on what binds the SpMV kernel. **The 18 GB/s premise is stale.** The
2026-07-20 audit
(`docs/benchmarks.md#spmv-bounds-check-elision-krylov-kernel-2026-07-20`)
measured about 18 GB/s on the single-accumulator serial kernel. Since then
commit 42177cf7 unrolled the gather-dot over four accumulators (−8.1% at
n = 1M from instruction-level parallelism alone), and commit 8c2cb26b spread
SpMV over workers past 65,536 nonzeros under the default `parallel` feature
(`spmv/parallel.rs`, which calls the kernel gather-bound, not a pure
stream). `bench_spmv` at 1<<20 now runs the spread path.

**Derived rates** from the `parallel.rs` table (`f64` seven-point operator,
7,340,020 nonzeros, 24-worker host; matrix stream only, 16 B per nonzero =
117,440,320 B): serial 7.02 ms gives 16.7 GB/s, spread 2.19 ms gives
53.6 GB/s. The repository records no host DRAM bandwidth, so no
machine-balance claim is made, and the audit host is not recorded as the
table's host. **Hypotheses, unverified.** H1: the serial path is partly
latency-bound, so narrowing gains less than the ceiling below. H2: the
spread path at 1<<20 is near DRAM saturation (its speedup over serial falls
from 5.03× at 65,536 rows to 3.21× at 1,048,576), so it gains near the
ceiling. The verification plan settles both.

**Ceiling.** Assume time proportional to streamed bytes, `x` and `y` each
streamed once, `y` one write without a write-allocate read, 8-byte `usize`.
A row of `r` nonzeros streams `(sizeof(T) + s)(r + 1)` bytes of values,
column indices, one `row_ptr` word and one `y` element (`s` the index
width), plus `X ≥ 0` bytes of `x` traffic. Narrowing `s = 8` to `s = 4`
speeds it up by at most

`((T + 8)(r + 1) + X) / ((T + 4)(r + 1) + X) ≤ (T + 8) / (T + 4)`,

since the ratio decreases in `X`. Ceilings at `X = 0`: `f64` `16/12 ≈ 1.333×`
(25% fewer bytes), `f32` `12/8 = 1.5×` (33%), `F16`/`Bf16` `10/6 ≈ 1.667×`
(40%). The `bench_spmv` operator has `r = 7`, `X = 8` B/row: `f64` ceiling
`136/104 ≈ 1.308×`. The ceiling is derived, not measured; "halving index
traffic" is exact for the index stream and a quarter of the `f64` matrix
stream.

## Decision (proposed)

1. **Index width is a type parameter with a 32-bit default.** The sparse
   formats take `I: SparseIndex`, a sealed trait implemented for `u32` and
   `u64`, defaulted to `u32`: `CsrMatrix<T, I = u32>`. Width is real
   variation, so it is a parameter of one implementation, not a second type
   or a copied kernel. `usize` is not an `I`: a layout that follows the
   target width cannot be serialized or shared with a device.
2. **Checked entry, unchecked traversal.** A pure `fits::<I>(nnz, nrows,
   ncols)` is called by `from_parts`, `from_dense`, COO assembly and every
   conversion, and returns `LetoError::StorageError` naming the exceeded
   bound. `nnz ≤ I::MAX` is representational (`row_ptr` holds counts). CSR
   stores no row index and `ncols − 1 ≤ I::MAX` admits `ncols = 2³²`, so
   requiring `nrows` and `ncols ≤ I::MAX` is a uniform choice across the
   three formats, not a representational cap. Kernels widen `I → usize` only
   where they index, through the crate's one conversion owner;
   `nonzero_stream_bytes` (`spmv/parallel.rs`) takes `size_of::<I>()`.
3. **Accessors expose the stored width.** `row_ptr()`, `col_indices()`,
   `as_parts` and the view types return `&[I]`. `nrows()`, `ncols()`,
   `nnz()` and `shape()` stay `usize`: counts for indexing and allocation.
4. **One storage format per concept.** `leto::CsrArray` (ADR 0028)
   duplicates `leto_ops::CsrMatrix`. The width change lands on the format
   that survives `LETO-SPARSE-FORMAT-CONSOLIDATION-2026-11-13`, never both,
   and no adapter bridges two widths. This record revises ADR 0028's
   `Vec<usize>` structure for the surviving format.
5. **Hermes routing follows, gated on measurement, and revises ADR 0009.**
   `hermes_simd::spmv_csr` takes `&[i32]`; a per-call `usize → i32`
   conversion costs `O(nnz + m)`, as much as the SpMV. ADR 0009's follow-up
   caches `i32` indices natively; this ADR rejects `i32` storage and
   reinterprets a `u32` array with entries below `2³¹` as `i32` without a
   copy, through `eunomia::layout::cast_slice` (used in
   `application/layout/complex/tile.rs`; no new dependency). Hermes fixes its
   index element at `i32` (`I: AsRef<[i32]>`) and parameterizes only
   ownership, so it is no precedent for width parameters. Routing is decided
   by the item that adds it, on a measured gain. ADR 0009 takes a dated
   revision note when this ADR is Accepted.

## Rejected alternatives

- **Fixed `u32`, no parameter.** Smallest diff, same expected win, but a
  silent cap of `2³² − 1` nonzeros (51.5 GB at `f64`) with no escape on a
  1.0 contract.
- **`u32` as a second type (`CsrMatrix32`).** Clones every kernel and
  conversion; width marker in a type name.
- **`i32` indices to match `spmv_csr`.** Admits negative indices and halves
  the range for no gain; `u32 → i32` reinterpretation reaches the kernel.
- **Narrow `row_ptr` only.** Saves 4 of `(T + 8)(r + 1)` bytes per row (3%).
- **Delta-coded column indices.** A per-nonzero decode cost and a different
  invariant set: a separate representation, not a width.

## Consequences

- **Public break [major].** Every `&[usize]` accessor, the `from_parts`
  signatures and the view types change. Files naming a Leto sparse type in
  a consumer, counted 2026-11-13 at each fetched default by `git grep -l -E`
  over `*.rs` with the word-bounded alternation `CsrMatrix|CscMatrix|
  CooMatrix|CsrArray|CscArray|CooArray`: athena 51, CFDrs 28, hephaestus 14,
  kwavers 5, coeus 2, prometheus 1; only prometheus names an `*Array`
  format. Migration is per consumer and dependency-ordered; no shim keeps
  the old signatures.
- **A matrix beyond `2³² − 1` nonzeros requires `u64`**, chosen by the
  caller's type annotation; the default never widens silently. Memory drops
  by the byte reductions above; kernels stay one implementation per format.
- **The sparse LU, AMD and SpGEMM paths** migrate in the same slice. The
  sparse directory last changed on 2026-11-10 (`coo.rs`; `amd` 2026-11-09),
  so the slice starts from the then-current `origin/main`.

## Verification plan

Each delivery slice carries its own oracle; this record claims none.

- `fits::<I>` is tested at the count boundary for `u32` and `u64` without
  building the matrix (accepts `nnz = I::MAX`, rejects `I::MAX + 1`; the
  same per dimension; `nnz = 2³² − 1` needs about 51 GB), and each
  constructor is shown to call it. Dimension boundaries with `nnz = 0` are
  constructor tests.
- Every sparse product and solve returns bitwise-identical results for
  `u32` and `u64` indices on the existing sparse suites, since evaluation
  order does not depend on the index width.
- `bench_spmv` at 1<<20 and `bench_csc_spmv` (`crates/leto-ops/benches/
  kernels`), `f64`, before and after, on a quiet host or by instruction and
  cache counters (`docs/benchmarks.md`), with default features (spread) and
  `parallel` off (serial), plus a host STREAM-class bandwidth figure. This
  settles H1 and H2. Wall-clock results under compiler load are discarded.
- Predicate: the old/new median-time ratio `S` of `bench_spmv` has a 95%
  interval whose lower bound exceeds 1 (the noise bound) and does not lie
  wholly above `1.308`; an interval wholly above falsifies the byte model.

## Overturning evidence

A `bench_spmv` ratio interval at 1<<20 that includes 1 (the kernel is not
index-bound, for instance because the `x` gather dominates) overturns the
performance case on the default-feature path; a serial-only absence is H1
evidence. The Hermes case would stand alone and the width is reconsidered.
