# ADR 0029: Runtime-rank fusion layout contract

## Status

Accepted

## Date

2026-09-02

## Board item

`LETO-DYNAMIC-LAYOUT-PROVIDER-SEAM-2026-09-02`

## Context

Hephaestus's provider-owned fusion seam accepts runtime-rank views so a
consumer can submit an expression with an arbitrary number of tensor inputs
without converting the data to a fixed-rank or contiguous representation.
`LayoutDyn` already carries the required shape, element strides, and offset,
but its broadcast and output-injectivity laws were present only on fixed-rank
`Layout<N>` or were duplicated in the WGPU provider.

Duplicating either law in Hephaestus would give CPU and accelerator paths
different answers for the same view and would make the provider responsible
for array-layout semantics. The existing layout `kernels` module is the
rank-agnostic arithmetic single source of truth.

## Decision

1. Add `LayoutDyn::broadcast(&[usize])`, using the same trailing-axis rules as
   fixed-rank `Layout<N>`: equal extents retain their strides, source extent
   one becomes a zero stride, and prepended axes are zero-stride broadcasts.
2. Add `LayoutDyn::is_injective()` and move the fixed-rank injectivity proof to
   the shared layout kernel. The fast separated-stride proof is followed by
   the existing exact bounded integer-difference search for ambiguous layouts.
   A provider can therefore reject overlapping writable views without a
   provider-local approximation.
3. Keep these APIs layout-only. They allocate metadata only when a dynamic
   broadcast or exact proof requires it; they never copy the backing elements
   and do not depend on an accelerator crate.

The provider remains responsible for buffer length and device address-width
validation because those invariants require the buffer contract.

## Alternatives rejected

- **Keep the WGPU injectivity heuristic.** Rejected: it rejects valid
  interleaved layouts and duplicates a CPU law at the wrong ownership layer.
- **Convert runtime-rank views to fixed-rank layouts in Hephaestus.** Rejected:
  the fusion seam must admit runtime rank and arbitrary expression input count;
  a rank cap belongs to the provider's shader ABI, not to Leto's layout model.
- **Materialize broadcast operands.** Rejected: broadcast changes metadata,
  not values, so materialization would add memory traffic and violate the
  zero-copy view contract.

## Consequences

The public Leto surface gains additive runtime-rank layout operations. Fixed
and dynamic layouts share the same broadcast and injectivity implementation.
Hephaestus consumes the proof through `LayoutDyn::is_injective()` and no longer
needs a second output-overlap algorithm. The exact difference search remains a
cold boundary operation; the provider's hot shader loop receives already
validated metadata.

## Verification

Dynamic tests compare broadcast metadata with fixed-rank results and compare
injectivity across exhaustive small shapes and strides. Provider contracts use
the dynamic methods before WGPU allocation or submission and exercise both
valid non-contiguous views and rejected overlapping output views.
