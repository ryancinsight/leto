# ADR 0031: Runtime-rank broadcast and injectivity contract

## Status

Accepted (retroactive: recorded from the code as it stands, not from a
proposal that preceded it)

## Date

2026-09-09

## Context

Hephaestus's provider-owned fusion seam accepts runtime-rank views, so a
consumer submits an expression over an arbitrary number of tensor inputs
without converting to a fixed-rank or contiguous representation. `LayoutDyn`
already carried shape, element strides and offset, but its broadcast and
output-injectivity laws lived only on fixed-rank `Layout<N>` or were
duplicated in the WGPU provider.

Duplicating either law in the provider would give CPU and accelerator paths
different answers for the same view, and would make the provider responsible
for array-layout semantics. The `layout::kernels` module is the rank-agnostic
arithmetic single source of truth.

Both operations were implemented and merged without a decision record. This
ADR records the contract as built; nothing here proposes a change.

## Decision

1. `LayoutDyn::broadcast(&[usize])` delegates to
   `kernels::broadcast_strides`, which applies the same trailing-axis rules as
   fixed-rank `Layout<N>`: axes align at the trailing end, an equal extent
   retains its stride, a source extent of one takes stride zero, and prepended
   axes are zero-stride. Any other extent mismatch is
   `LetoError::IncompatibleBroadcast` carrying both shapes.
2. `LayoutDyn::is_injective()` and `Layout<N>::is_injective()` both delegate to
   `kernels::is_injective`, which runs the separated-stride proof first and
   falls back to `exact_injectivity`'s bounded integer-difference search for
   ambiguous layouts. A provider therefore rejects overlapping writable views
   without a provider-local approximation.
3. Both operations are layout-only. They allocate metadata when a dynamic
   broadcast or an exact proof requires it, never copy backing elements, and
   do not depend on an accelerator crate.

Buffer length and device address-width validation remain the provider's,
because those invariants require the buffer contract rather than the layout.

## Alternatives rejected

- **Keep the WGPU injectivity heuristic.** It rejects valid interleaved
  layouts and duplicates a CPU law at the wrong ownership layer.
- **Convert runtime-rank views to fixed-rank layouts in the provider.** The
  fusion seam must admit runtime rank and an arbitrary input count; a rank cap
  belongs to the provider's shader ABI, not to Leto's layout model.
- **Materialize broadcast operands.** Broadcast changes metadata, not values,
  so materialization adds memory traffic and breaks the zero-copy view
  contract.

## Consequences

Fixed-rank and dynamic layouts share one broadcast implementation and one
injectivity implementation, so a provider cannot observe them disagreeing.
The public surface is additive.

`broadcast_strides` writes only strides and takes the target shape from the
caller, so a caller that needs the broadcast *shape* supplies it — the kernel
does not derive it. Callers that must validate the target shape do so before
calling.

## Provenance

The contract was drafted on `feat/leto-fusion-seam` (PR #159) alongside an
independent implementation. `main` reached the same design first, so that
branch's code is superseded and its ADR — the only part `main` lacked — is
recorded here against the merged names. The branch's number 0029 collided with
`0029-provider-source-identity-during-coevolution.md`; this record takes 0031.
