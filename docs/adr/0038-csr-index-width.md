<a id="adr-0038"></a>

# ADR 0038: Sparse formats store indices as a width-parameterized unsigned integer

Status: Proposed

Board item `LETO-CSR-INDEX-WIDTH-2026-09-27`; delivered in slices, this
record is the first.

## Context

`CsrMatrix`, `CscMatrix` and `CooMatrix` in `leto-ops`
(`application/sparse/`) and `CsrArray`, `CscArray` and `CooArray` in `leto`
(`infrastructure/sparse/`) store every index as `usize`. `col_indices`,
`row_ptr`, `as_parts`, `CsrView` and `CsrRow` expose `&[usize]`. On the
64-bit targets Leto ships this is 8 bytes per index.

SpMV is bandwidth-bound once the matrix leaves the cache: per nonzero it
streams one value and one column index, gathers one word of `x`, and does
one fused multiply-add, so arithmetic intensity is `2 flop / (sizeof(T) +
sizeof(index))` bytes, far below the machine balance of any host in
`docs/benchmarks.md`. The 2026-07-20 audit
(`docs/benchmarks.md#spmv-bounds-check-elision-krylov-kernel-2026-07-20`)
measured the DRAM-bound CSR case at about 18 GB/s after bounds-check
elision, so streamed bytes, not instructions, set the time.

Streamed matrix bytes per nonzero, excluding `row_ptr` (one index per row,
negligible once rows average more than a few entries):

| `T` | `usize` index | `u32` index | reduction | ceiling on DRAM-bound SpMV speedup |
| --- | --- | --- | --- | --- |
| `f64` | 16 B | 12 B | 25% | `16/12 ≈ 1.33×` |
| `f32` | 12 B | 8 B | 33% | `12/8 = 1.5×` |
| `F16`, `Bf16` | 10 B | 6 B | 40% | `10/6 ≈ 1.67×` |

The ceiling is derived, not measured: the `x` gather and `y` write are
unchanged, so a measured gain sits below it. The board phrase "halving index
traffic" is exact for the index stream and about a quarter of the whole
stream for `f64`.

A second consumer points the same way. ADR 0009 records that
`hermes_simd::spmv_csr` takes `&[i32]` indices and that a per-call
`usize → i32` conversion costs `O(nnz + m)`, as much as the SpMV, so the
SIMD route was deferred until indices are held natively at 32 bits. Hermes
itself already parameterizes the index storage (`CsrMatrix<T, V, I>`,
`hermes-simd-core/src/sparse/types.rs`) and bounds gather offsets to `i32`
(`sparse/ops.rs`).

## Decision (proposed)

1. **Index width is a type parameter with a 32-bit default.** The sparse
   formats take `I: SparseIndex`, a sealed trait implemented for `u32` and
   `u64`, defaulted to `u32`: `CsrMatrix<T, I = u32>`. Index width is real
   variation (a 32-bit index caps `nnz` and the larger dimension at
   `2³² − 1`), so it is a parameter of one implementation, not a second type
   or a copied kernel. `usize` is not an `I`: its width follows the target,
   and a format whose byte layout changes with the target cannot be
   serialized or shared with a device.
2. **Checked entry, unchecked traversal.** `from_parts`, `from_dense`, COO
   assembly and every conversion validate once that `nnz`, `nrows` and
   `ncols` fit `I` and return `LetoError::StorageError` naming the exceeded
   bound. A kernel widens `I → usize` only where it indexes, through the one
   conversion owner for the crate; `u32 → usize` is lossless on every
   supported target.
3. **Accessors expose the stored width.** `row_ptr()`, `col_indices()`,
   `as_parts` and the view types return `&[I]`. `nrows()`, `ncols()`,
   `nnz()` and `shape()` stay `usize`: they are counts used for indexing
   and allocation, not stored columns.
4. **One storage format per concept.** `leto::CsrArray` duplicates
   `leto_ops::CsrMatrix`. The width change lands on the format that
   survives the consolidation filed as its own item; it is never applied to
   both, and no adapter bridges two widths.
5. **Hermes routing follows, gated on measurement.** A `u32` index array
   with every entry below `2³¹` is bit-identical to `i32` and
   reinterpretable without a copy (`bytemuck::cast_slice`, never a loop).
   Whether to route SpMV through `spmv_csr` is decided by the item that adds
   it, on a measured gain (ADR 0009).

## Rejected alternatives

- **Fixed `u32`, no parameter.** Smallest diff and the same measured win,
  but it caps every matrix at `2³² − 1` nonzeros (12 B each, 51.5 GB at
  `f64`) with no escape on a 1.0 contract. Meshes in the stack are far below
  it today; the cap is nevertheless a silent ceiling.
- **`usize` kept, `u32` offered as a second type (`CsrMatrix32`).** Clones
  every kernel and conversion and puts a width marker in a type name.
- **`i32` indices to match `spmv_csr` directly.** Signed storage admits
  negative values the format forbids, halves the range for no gain, and
  imports a foreign layout into Leto's contract; the `u32 → i32`
  reinterpretation reaches the same kernel without it.
- **Narrow `row_ptr` only.** `row_ptr` has `nrows + 1` words against `nnz`
  column words, so it removes nothing from the stream the table is about.
- **Delta-coded or compressed column indices.** A larger win on banded
  matrices at a per-nonzero decode cost, with a different invariant set
  (strictly increasing columns become differences); a separate
  representation, not a width.

## Consequences

- **Public break [major].** Every `&[usize]` accessor, the `from_parts`
  signatures and the view types change in `leto` and `leto-ops`. Files
  naming the sparse types in a stack consumer of Leto, counted with `grep -l`
  over `repos/` on 2026-11-13: athena 50, CFDrs 28, hephaestus 14, kwavers 5,
  coeus 2, prometheus 1. Migration is per consumer and dependency-ordered,
  `cargo-semver-checks` reports the break at the provider, and no shim keeps
  the old signatures.
- **A matrix beyond `2³² − 1` nonzeros requires `u64`**, chosen by the
  caller's type annotation; the default never widens silently.
- **Memory for the common case drops** by the table above, beside the
  bandwidth effect; `Clone`, conversion and assembly copy fewer bytes.
- **Kernels stay one implementation per format.** Instantiations per sparse
  kernel go from one to at most two (`u32`, `u64`); the `u64` instance is
  not exercised by default call sites.
- **The sparse LU, AMD and SpGEMM paths** (`lu_sparse`, `lu_numeric`,
  `amd`, `spgemm`) index through the same accessors and migrate in the same
  slice as the format; they changed on 2026-11-09 and 2026-11-10, so the
  migration slice starts from the then-current `origin/main`.

## Verification plan

Each delivery slice carries its own oracle; this record claims none.

- `from_parts` rejects `nnz`, `nrows` or `ncols` above `I::MAX` with the
  typed error and accepts `I::MAX` exactly (boundary partition), for `u32`
  and `u64`.
- Every sparse product and solve returns bitwise-identical results for
  `u32` and `u64` indices on the existing sparse suites, since evaluation
  order does not depend on the index width.
- `bench_spmv` and `bench_csc_spmv` (`crates/leto-ops/benches/kernels`) on a
  quiet host or with instruction and cache counters (`docs/benchmarks.md`):
  the `f64` DRAM-bound ratio is reported against the `1.33×` ceiling.
  Wall-clock results taken under concurrent compiler load are discarded.

## Overturning evidence

A re-measured `bench_spmv` at the DRAM-bound size showing no gain beyond the
noise bound (the stream is then not index-bound, for instance because the
`x` gather dominates) overturns the performance case. The Hermes routing
case of ADR 0009 would then stand alone, and the width is reconsidered on
that evidence only.
