# ADR 0002: Const-rank vs dynamic-rank boundary for Coeus integration

- Status: Accepted
- Date: 2026-06-10
- Revised: 2026-07-31
- Class: [major, arch]

Revision 2026-07-31: CPU operation families over Leto views belong in Leto,
including convolution and scaled dot-product attention. Coeus retains autodiff,
NN orchestration, and backend selection; accelerator implementations belong in
Hephaestus. This replaces the earlier statement that Coeus owns attention
kernels. The const-rank boundary decision is unchanged.

## Context

Coeus (the Atlas `burn` replacement) carries its own non-differentiable array
layer (`coeus-tensor`/`coeus-core`: layout, storage, COW, traversal) built over
the same Mnemosyne + Moirai substrate as Leto. The structural-duplication rule
requires consolidating that layer into Leto. Leto owns CPU array-operation
families expressed over its views, including convolution and scaled dot-product
attention. Coeus retains `ComputeBackend`, autodiff graphs, NN orchestration,
optimizers, higher sparse formats, and runtime backend selection. Hephaestus
owns accelerator kernels. Leto may also own narrow CPU sparse kernels when they
are part of ndarray/nalgebra parity, such as CSR SpMV/SpMM.

The blocking mismatch: Coeus's `Layout` is **runtime-rank** (rank carried as a
`Vec`/dynamic dimension), while Leto's `Layout<const N: usize>` is
**const-rank** (rank in the type system, enabling compile-time shape checks,
monomorphized stride loops, and `[usize; N]` storage with no heap indirection).

## Options

1. Const-generic dispatch shim at the Coeus boundary: Coeus's dynamic rank is
   matched to a Leto `const N` via a bounded `match rank { 1 => …<1>, 2 => …<2>,
   … }` dispatch at the FFI/tensor-op entry, calling monomorphized Leto kernels.
   Leto stays purely const-rank.
2. Introduce a `DynArray`/`IxDyn` escape type in Leto carrying runtime rank,
   used at the Coeus boundary and converted to const-rank internally.
3. Make Leto dual-rank (both const and dynamic layouts as first-class).

## Decision

Adopt option 1: a const-generic dispatch shim at the Coeus boundary. Leto
remains const-rank end to end.

Rationale:

- Preserves Leto's core invariant — rank in the type system — which is the
  source of its compile-time shape safety and monomorphized, allocation-free
  stride traversal. A `DynArray` would fork the layout model and reintroduce
  runtime shape checks and their error variants that Leto deliberately deleted.
- The dispatch is bounded: Atlas tensor ranks are small (Coeus activations,
  Apollo transforms ≤ rank 4–6 in practice). A `match` over a closed rank set
  monomorphizes each arm and adds one branch at the boundary, off the hot inner
  loops.
- Keeps the duplication consolidation one-directional: Coeus depends on Leto
  for CPU operations and Hephaestus for accelerator operations; Leto gains
  nothing Coeus-specific.

## Consequences

- Phase 6 leto-side capabilities (broadcast-aware binary into output layouts,
  unary math suite, reshape/permute/to_contiguous, concat/pad/split, batched
  matmul, convolution, scaled dot-product attention, cumsum, seeded RNG) are
  authored const-rank.
- The shim itself lives in Coeus (the consumer), not in Leto, per the upstream-
  ownership rule: Leto owns the const-rank kernels; Coeus owns the adaptation of
  its dynamic rank onto them. A Coeus-side backlog item names Leto as provider.
- A rank cap (the largest `const N` the shim dispatches) must be stated
  explicitly in Coeus; ranks beyond it are a logged error, not silent
  truncation.
- Leto attention accepts borrowed rank-3 views, caller-owned outputs, and
  additive optional gradient targets. Validation is typed and completes before
  mutation; masks broadcast as views and are never materialized.
- Coeus must propagate Leto and Hephaestus failures through a fallible attention
  contract. Unsupported layouts, devices, scalars, compilation, and launches
  never change execution provider.
