<a id="adr-0039"></a>

# ADR 0039: Allocating kernels zero-initialize their output storage

Status: Proposed

Board item `LETO-VECSTORAGE-UNINIT` (class `[major]`, delivered by PR 422; its
predecessor `LETO-VECSTORAGE-UNINIT-2026-11-13` by PR 419). The number 0036
belongs to the typed `MaybeUninit` design on `refs/pull/360/head`.

Revision 2026-11-15: `uninit` is removed and each allocating kernel zero-fills its output, replacing PR 419's `unsafe` constructor, because its `Vec::set_len` call breached std's initialization precondition (`alloc/src/vec/mod.rs:2141`, Rust 1.97.0; PR 419 post-merge verdict G1); the zero-fill choice is a user veto point, reversed by `refs/pull/360/head`.

## Context

`VecStorage::<T>::uninit` was a safe `pub fn` that called `Vec::set_len`
over `with_capacity` for any `T`. Safe code could therefore read, drop or
compare uninitialized `T`. Miri (nightly 2026-08-26, `-Zmiri-disable-isolation`)
on `b738a4e2` reported UB for `VecStorage::<f64>::uninit(4)` followed by
`as_slice()[0]`, with no `unsafe` block in the test. For a `T` with drop
glue, `*slot = value` first drops the never-constructed old value (`pad` over
`String`, PR 360).

PR 419 made `uninit` an `unsafe fn` over `T: Copy` with a `SAFETY` proof at
each caller. That removed the safe-code reads and the drop defect, and it
left the `Vec::set_len` breach: the precondition concerns memory state at the
call, before any caller receives the storage, so no caller proof discharges
it. std checks only `new_len <= capacity()` (`check_library_ub`), so no
runtime check, Miri included, observes the initialization clause, and the
green Miri runs on PR 419 do not bear on it. `MaybeUninit`'s documentation
also calls uninitialized data in an integer-typed variable undefined behavior
(`core/src/mem/maybe_uninit.rs:42-46`).

## Decision

1. `VecStorage::uninit` is deleted, with its `#[expect(clippy::uninit_vec)]`.
2. The six former call sites (`embedding_gather`, `reduce_axis`, `topk_axis`
   (two buffers), `triangular`, `ray_line_integrals`; `pad` had already
   stopped using it) allocate zero-filled storage through
   `VecStorage::fill(size, T::ZERO)` (`0_u32` for indices, `0.0_f32` for
   `ray_line_integrals`) and then overwrite it. `fill` is `vec![value; len]`,
   which std specializes to `alloc_zeroed` for the primitive element types
   when the value is zero bits (`alloc/src/vec/spec_from_elem.rs:20-26`);
   `F16`, `Bf16` and complex elements take the element-wise fill.
3. No `set_len` runs over a `Vec<T>` whose new elements are unwritten.
   `concat` and `stack` keep their `Vec<MaybeUninit<T>>` assembly, where
   `set_len` is vacuous, and now carry `SAFETY` proofs for their writes.

## Alternatives rejected

- `VecStorage<MaybeUninit<T>>` with an unsafe consuming `assume_init`
  (`refs/pull/360/head`, 25 files, +743/-171): the only design that keeps the
  single-write path without a `set_len` breach, since no `&mut [T]` is formed
  over unwritten memory. Rejected for now: it needs an output-slot trait
  through every kernel, a separate sliced delivery, and the zero-fill cost is
  not shown to matter; it stays the upgrade path if it does.
- Keeping `uninit` with corrected prose, `unsafe` or not: every call still
  breaches the precondition, and documentation does not repair that.

## Consequences

`uninit` leaves the public API, a `[major]` source break (it was a safe `pub fn`
in the released 0.47.0). No other stack member at its fetched default head or
in its working tree calls it; external callers use `generate`, `fill` or `new`.

Each allocating kernel pays one store pass over its output before the
overwrite. Whether `alloc_zeroed` (decision 2) avoids the pass for zero-bit
primitives depends on the global allocator: `GlobalAlloc::alloc_zeroed`
defaults to `alloc` followed by `write_bytes`
(`core/src/alloc/global.rs:216-226`, Rust 1.97.0). Skipping the pass until
the pages are touched is an unmeasured hypothesis for allocators that
override it.

Measurement (`crates/leto-ops/benches/output_allocation.rs`, Criterion, 30
samples, release profile, Intel Core Ultra 9 285K, Windows 11, rustc 1.97.0).
A is origin/main `bd349c5e` (`uninit`, no fill; its kernels are the pre-PR 419
kernels, which `d16e3b2` introduced for `reduce_axis` alone), B is this
change. Interleaved A/B rounds over the five operations ran at a host CPU
load of 29-100% from peer agents. The spread inside one binary (`triangular`
A 0.96-2.17 ms) exceeds any effect, so the wall-clock instrument is invalid
for a cost claim under this load and its ratios are discarded: no regression
is established and none is excluded. The expected cost is
bounded by one store pass over the output (4 MiB for `triangular` and
`embedding_gather`); isolated-core confirmation is `LETO-OUTPUT-ZEROFILL-COST`.

Overturning evidence: a regression on a hot path that
`LETO-OUTPUT-ZEROFILL-COST` measures beyond its derived noise bound. The
remedy is the `MaybeUninit` design, not a return to `set_len`.
