<a id="adr-0039"></a>

# ADR 0039: `VecStorage::uninit` is an unsafe constructor over `Copy` elements

Status: Proposed

Board item `LETO-VECSTORAGE-UNINIT-2026-11-13` (class `[major]`, delivered by
the PR that adds this record). The number 0036 is held by the typed
`MaybeUninit` design on `refs/pull/360/head`, the strongest rejected
alternative below.

## Context

`VecStorage::<T>::uninit` was a safe `pub fn` that called `Vec::set_len`
over `with_capacity` for any `T`. Safe code could therefore read, drop or
compare uninitialized `T`. Miri (nightly 2026-08-26, `-Zmiri-disable-isolation`)
on `b738a4e2` reported UB for `VecStorage::<f64>::uninit(4)` followed by
`as_slice()[0]`, with no `unsafe` block in the test. For a `T` with drop
glue the defect is wider: assigning `*slot = value` first drops the old,
never-constructed value, so `pad` over `String` destroyed unwritten strings
(Miri findings recorded on PR 360).

Seven call sites existed: six numeric allocating kernels (`embedding_gather`,
`reduce_axis`, `topk_axis`, `triangular`, `ray_line_integrals`, `pad`) and
the constructor itself. The six numeric kernels instantiate `T: Scalar`,
sealed to integers, floats, `F16`/`Bf16` and complex pairs of them, `u32`
indices, or `f32`: every one is valid for every bit pattern and has no drop
glue. `pad` is `T: Clone`, so it has neither property.

## Decision

1. `uninit` is `pub unsafe fn`, defined in `impl<T: Copy> VecStorage<T>`. The
   `Copy` bound removes drop glue from the contract, so overwriting or
   discarding an unwritten element destroys nothing. The `# Safety` contract
   requires: `T` valid for every bit pattern, every element written before it
   is read on every exit, and no read, clone or comparison of an unwritten
   element. A `compile_fail,E0133` doctest pins the unsafe boundary.
2. Each numeric caller carries a `// SAFETY:` proof covering every route of
   the kernel it calls (contiguous, strided, parallel, empty axis) and its
   early `Err` returns, which drop the storage unread.
3. `pad` stops using `uninit`. It builds the output by pushing one value per
   cell in row-major order, through the value-selection rule `pad_into` also
   uses, so the vector length is the live prefix on `Err` and unwind.

## Alternatives rejected

- `VecStorage<MaybeUninit<T>>` with an unsafe consuming `assume_init`
  (`refs/pull/360/head`, 25 files, +743/-171): the stricter design, since no
  `&mut [T]` is formed over unwritten memory, but it threads an output-slot
  trait through every numeric kernel. It stays the upgrade path if the
  hypothesis below is overturned; it is a separate slice, not a precondition.
- Zero-filling (`vec![zero; n]`): needs a zero for arbitrary `T` and adds
  the memory pass that PR 157 removed from `reduce_axis`.
- Keeping the safe signature with prose obligations: unsound by construction.

## Consequences

`uninit` leaves safe code: a source-breaking change classified `[major]`;
external callers write `unsafe { .. }` with their own proof or use
`generate`, `fill` or `new`. `pad` pays one extra element-wise clone path
instead of an uninitialized pass (a push per cell, no zero fill). Whether
forming `&mut [T]` over unwritten primitive memory is itself UB is open in
the Rust reference (hypothesis); the contract forbids reads, which is what
Miri observed, and the executed Miri job covers the reduction route.

Overturning evidence: a Miri or reference ruling that `&mut [T]` over
unwritten integers is UB would require the `MaybeUninit` design above.
