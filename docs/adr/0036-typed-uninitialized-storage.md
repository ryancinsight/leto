# ADR 0036: Typed uninitialized storage

## Status

Proposed (implementation and safety evidence submitted for review)

## Date

2026-10-14

## Context

`VecStorage::uninit` safely returns `VecStorage<T>` after extending a vector's
length over uninitialized elements. Safe reads and destructors then consume
uninitialized values. Miri rejects reads of `NonZeroU8` and `u32`, destruction
of `String`, and ordinary `pad` of strings. A prose caller obligation cannot
make a safe generic constructor sound. Completing writes also does not make
assignment safe for a non-Copy type: assignment first drops its old value.

## Decision

1. `VecStorage::<T>::uninit` returns `VecStorage<MaybeUninit<T>>`. Allocation
   establishes only valid uninitialized slots, with `T`'s size and alignment.
   An unsafe consuming `assume_init` converts the same allocation once every
   slot holds a valid, uniquely owned `T`. Its contract covers all slots,
   including resource ownership; partial initialization cannot justify it.
2. Numeric allocation paths share their existing kernels with caller-owned
   output paths through a private static output-slot trait. Stores to an
   uninitialized slot use `MaybeUninit::write`; initialized stores retain
   assignment/copy/fill. Reads in reduction and top-k explicitly require a
   preceding store. Kernel success initializes every dense output slot.
3. Numeric kernels require Copy scalars. Error and panic paths drop only
   `MaybeUninit` slots and their allocation, never nonexistent scalar values.
   Padding supports arbitrary Clone types, so its allocating path pushes each
   generated value into a normal vector. A shared value-selection helper
   keeps allocating and caller-owned padding semantics identical. Vector
   length tracks exactly the live prefix during error or unwind.
4. The pre-1.0 breaking return type advances the workspace to 0.46.0. External
   callers must write slots and explicitly establish the unsafe conversion
   contract, or use `generate`, `fill`, or `new` for safe construction.

## Alternatives rejected

- Making the old constructor unsafe still creates ordinary typed storage
  before initialization and allows destructors to observe unwritten slots.
- Zero filling unrestricted T is invalid for types such as NonZeroU8 and
  String. Adding a blanket numeric zero fill adds an unnecessary memory pass.
- Duplicating owned and borrowed numerical algorithms creates two independent
  contracts. Output-slot monomorphization preserves their single algorithm.

## Consequences

Safe callers can allocate, inspect and drop uninitialized slots without
reading a T. Unsafe conversion is the sole initialization boundary. Native
tests cover values, layout, tails, errors and unwind; Miri checks both borrow
models, including non-Copy padding and the original reduction scenarios.
Negative compile tests reject ordinary typed access and safe conversion.
Publication and merge remain held by the current task's release constraints.
