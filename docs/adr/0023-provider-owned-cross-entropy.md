# ADR 0023: Own CPU cross-entropy in leto-ops

- **Status:** Accepted
- **Date:** 2026-08-04
- **Backlog:** `LETO-CROSS-ENTROPY-PROVIDER-1`
- **Cross-repository driver:** Coeus ADR 0052

## Context

A downstream tensor consumer currently downloads non-CPU logits, computes
stable row-wise softmax and mean cross-entropy in consumer code, and retains an
owned probability vector for backward. CPU execution therefore has no
provider-owned classification-loss contract and the consumer duplicates
arithmetic, validation, and storage policy.

The required operation accepts logits with logical shape `[batch, classes]`,
one target class per batch row, mean reduction, and an upstream scalar for
backward. Forward must retain probabilities in caller-owned storage because
backward consumes them without recomputation. Native scalar arithmetic,
arbitrary reachable layouts, checked indexing, and validation-before-mutation
are part of the contract.

## Decision

`leto_ops::{cross_entropy_forward_into,
cross_entropy_backward_accumulate}` exposes one scalar-generic family owned by
`application::loss`:

- forward writes normalized probabilities and one mean loss;
- backward additively writes the logit gradient from saved probabilities,
  targets, and the upstream scalar.

Inputs are borrowed views and outputs are mutable borrowed views. A shared
preflight validates rank, nonzero batch and class extents, target count and
range, storage reachability, writable-layout uniqueness, and checked products
before either output changes. The stable log-sum-exp shift uses the row maximum
and all arithmetic remains in `T`. Scalar and rank are compile-time dimensions;
no dynamic dispatch, intermediate collection, or consumer-shaped adapter is
introduced.

## Alternatives

- Keep the host formula in Coeus: rejected because the consumer remains the CPU
  algorithm owner.
- Compose public softmax and gather calls downstream: rejected because it
  materializes an operation graph and duplicates validation rather than
  providing one failure-atomic provider contract.
- Return owned arrays: rejected because forward probabilities already have a
  graph-defined lifetime and caller-owned storage avoids allocation and copy.

## Failure modes

Invalid rank, empty dimensions, target length or range, unreachable storage,
overlapping writable layouts, insufficient probability-sum resolution, and
non-finite inputs or arithmetic return typed `CrossEntropyError` values before
mutation. The operation rejects NaN and infinity explicitly. Its saved-row
validation tolerance composes the scalar division error with the standard
gamma bound for sequential summation and rejects class widths once that bound
can no longer distinguish a valid normalized row.

## Verification

Generic tests instantiate supported real scalar types and cover exact or
analytically bounded forward/backward values, contiguous and strided layouts,
invalid targets and shapes, special values, and failure atomicity. Focused
Nextest, warning-denied Clippy, doctests, Rustdoc, SemVer classification,
independent review, and exact-head CI gate the merge.

## References

- [PyTorch cross_entropy contract](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.cross_entropy.html)
