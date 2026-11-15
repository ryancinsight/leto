<a id="adr-0039"></a>

# ADR 0039: Allocating kernels zero-initialize their output storage

Status: Proposed

Board item `LETO-VECSTORAGE-UNINIT` (class `[major]`, delivered by PR 421; its
predecessor `LETO-VECSTORAGE-UNINIT-2026-11-13` was delivered by PR 419). The
number 0036 is held by the typed `MaybeUninit` design on
`refs/pull/360/head`, the strongest rejected alternative below.

Revision 2026-11-15: the decision changes from "`uninit` is an `unsafe`
constructor over `Copy` elements" (PR 419) to "`uninit` is removed and each
allocating kernel zero-fills its output". Reason: `Vec::set_len` requires the
elements at `old_len..new_len` to be initialized (`alloc/src/vec/mod.rs:2141`,
Rust 1.97.0), and `uninit` called it over elements nothing had written, so
every call breached the precondition, `unsafe` or not. The PR 419 text, which
called the design sound once callers proved write-before-read, was wrong.
Evidence: the std source above, read at 1.97.0; the post-merge verdict on
PR 419 (finding G1). The zero-fill choice is a user veto point: the reversal
is `refs/pull/360/head`.

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
runtime check, Miri included, observes the initialization clause; the green
Miri runs on PR 419 do not bear on it. `MaybeUninit`'s documentation also
states that uninitialized data in an integer-typed variable is undefined
behavior (`core/src/mem/maybe_uninit.rs:42-46`).

Six call sites used `uninit` outside the constructor: `embedding_gather`,
`reduce_axis`, `topk_axis` (two), `triangular` and `ray_line_integrals`, all
instantiated at `T: Scalar` (integers, floats, `F16`/`Bf16`, complex pairs of
them), `u32` indices, or `f32`. `pad` (`T: Clone`) had already stopped using
it.

## Decision

1. `VecStorage::uninit` is deleted, with its `#[expect(clippy::uninit_vec)]`.
2. Each former caller allocates zero-filled storage through
   `VecStorage::fill(size, T::ZERO)` (`0_u32` for indices, `0.0_f32` for
   `ray_line_integrals`) and then overwrites it. `fill` is `vec![value; len]`,
   which std specializes to `alloc_zeroed` for the primitive element types
   when the value is zero bits (`alloc/src/vec/spec_from_elem.rs:20-26`);
   `F16`, `Bf16` and complex elements take the element-wise fill.
3. No `set_len` runs over a `Vec<T>` whose new elements are unwritten.
   `concat` and `stack` keep their `Vec<MaybeUninit<T>>` assembly, where
   `set_len` is vacuous.

## Alternatives rejected

- `VecStorage<MaybeUninit<T>>` with an unsafe consuming `assume_init`
  (`refs/pull/360/head`, 25 files, +743/-171): the only design that keeps the
  single-write path without a `set_len` breach, since no `&mut [T]` is formed
  over unwritten memory. It stays the upgrade path if the zero-fill cost below
  matters on a hot path; it needs an output-slot trait through every kernel and
  is a separate sliced delivery.
- Keeping `uninit` with corrected prose, `unsafe` or not: every call still
  breaches the precondition, and documentation does not repair that.

## Consequences

`uninit` leaves the public API: a source-breaking change classified `[major]`
(`uninit` was a safe `pub fn` in the released 0.47.0). A search of every other
stack member at its fetched default head, and of the working trees, finds no
call; external callers use `generate`, `fill` or `new`.

Each allocating kernel pays one store pass over its output before the
overwrite, except that zero-bit primitives are served by `alloc_zeroed` and
skip the pass until the pages are touched.

The cost is not yet measured: a wall-clock A/B run under peer load was
too noisy to establish or exclude a regression, and `LETO-OUTPUT-ZEROFILL-COST`
holds the isolated-core A/B against `bd349c5e`.

Overturning evidence: a measured regression on a hot path beyond the noise
bound recorded above. The remedy is the `MaybeUninit` design, not a return to
`set_len`.
