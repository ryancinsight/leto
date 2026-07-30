# ADR 0019: Own CPU convolution in leto-ops

- **Status:** Accepted
- **Date:** 2026-07-30
- **Backlog:** `LETO-CONVOLUTION-PROVIDER-1`
- **Cross-repository driver:** Coeus ADR 0046
- **Revision 2026-07-30:** promoted parameter type identity into `leto` after
  Hephaestus dependency analysis; cargo-semver-checks classifies the canonical
  identity move as major even though `leto-ops` retains curated re-exports.

## Context

Coeus must select its execution provider at the backend boundary. CPU
convolution cannot route directly through Leto while regular forward,
backward, and transposed convolution remain consumer-owned host loops.
Retaining those loops would leave two authorities for shape validation,
numeric semantics, and failure behavior.

The public contract must preserve native scalar execution for `f32`, `f64`,
`F16`, and `Bf16`; arbitrary valid Leto layouts; and caller-owned output
storage. Transposed convolution follows the PyTorch shape and layout contract:
weights use `[input_channels, output_channels, kernel...]`, and output padding
changes the derived output shape without adding values to the output.

## Decision

`leto-ops::application::convolution` owns one const-generic operation family:

- `convolution_forward_into` computes regular N-dimensional cross-correlation.
- `convolution_backward_accumulate` adds input, weight, and bias gradients.
- `convolution_transposed_forward_into` scatters an N-dimensional transposed
  convolution.
- `convolution_transposed_backward_accumulate` adds selected input, weight,
  and bias gradients without consumer-owned host logic.

The lightweight `leto` domain owns validated parameter vocabulary carrying
stride, padding, dilation, and, for the transposed contract, output padding.
This lets CPU operations and accelerator planners share one contract without
an infrastructure-to-operations dependency. A preflight plan validates tensor
rank, shape, storage reachability, writable-layout aliasing, and checked
dimension arithmetic before any output mutation. Kernels borrow input views,
mutate caller-provided views, and use fixed-size coordinate arrays. Scalar and
spatial rank parameters monomorphize the loop bodies; no dynamic dispatch or
per-element allocation is present.

Regular and transposed contracts remain separate leaves because their weight
layouts, output-shape equations, and iteration directions differ. Shared
coordinate decoding remains in their deepest common convolution ancestor.

## Alternatives

- **Keep Coeus host defaults:** rejected because the consumer would remain the
  CPU algorithm owner and backend selection could not be provider-direct.
- **Convert tensors through owned staging buffers:** rejected because it adds
  allocation and copies at every operation boundary.
- **Duplicate rank-specific kernels:** rejected because rank is a const-generic
  variation dimension and one implementation can monomorphize for each rank.
- **Use a dynamic convolution trait:** rejected because the implementor set and
  rank are known at compilation and vtable dispatch adds no contract value.

## Failure modes

Invalid ranks, shapes, zero stride or dilation, unreachable storage, aliased
writable layouts, and arithmetic overflow return typed `LetoError` values.
Validation precedes mutation, so failure leaves every supplied output
unchanged. Floating-point reduction order is deterministic for a fixed rank
and layout; no cross-backend bitwise-equivalence claim is made.

## Migration

Consumers that need the parameter vocabulary without CPU operations import
`leto::{ConvolutionParameters, TransposedConvolutionParameters}`. Existing
`leto_ops` imports remain curated exports, but code that records canonical
item identity in generated API metadata must update it from `leto-ops` to
`leto`. No compatibility wrapper or duplicate parameter type is retained.

## Verification

The generic conformance suite instantiates all supported scalar types.
Analytical tests cover regular forward/backward, 1-D/2-D/3-D transposed
forward/backward convolution, padding, dilation, output-padding gradient
behavior, typed errors, strided views, and failure atomicity. Package check,
configured Nextest, doctests, Rustdoc, warning-denied Clippy, and SemVer
classification form the delivery gate.

## References

- [PyTorch ConvTranspose2d shape and output-padding contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d)
