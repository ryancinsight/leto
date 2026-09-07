# ADR 0030: Own temporal label alignment in Leto

- Status: Accepted
- Date: 2026-09-07
- Item: [LETO-CTC-LOSS](../../backlog.md#leto-ctc-loss)
- Driver: Coeus's tracked connectionist temporal classification loss.

## Contract

CTC assigns each collapsed target the sum of its alignment-path probabilities
([Graves et al., 2006, equations 2–3 and 12](https://www.cs.toronto.edu/~graves/icml_2006.pdf)).
Inputs are borrowed `[frames, batch, classes]` log-probability views and
concatenated nonblank target indices. Valid frame counts select prefixes;
padding has no effect. Empty targets use the all-blank path. Empty input and
target have probability one; impossible alignments have probability zero and
infinite loss. Mean reduction divides each sample by its target length clamped
to one, then by batch size, matching the
[PyTorch reduction definition](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossCTC.cpp).

Lengths exceeding the input extent are rejected, never clamped. Index ranges,
target totals, layout reachability, and all state-size arithmetic are validated
before state allocation. Active log weights must be nonpositive and not NaN;
negative infinity represents zero probability. The caller supplies normalized
log-probabilities; the same path-sum definition also applies to independent
nonpositive log weights.

## Ownership and representation

`leto_ops::ctc::CtcState<T>` owns one generic recurrence in the existing loss
family. `T: RealScalar` supplies arithmetic throughout, including
normalization and stored state. No fixed-precision bridge or hidden accumulator
changes the selected precision. Float wrappers retain their scalar-provider
arithmetic semantics.

Each log message stores a large offset and its rounding residual, both in `T`.
Compensated addition uses [Knuth's TwoSum as stated by Ogita et al., slide 7](https://ogilab.w.waseda.jp/ogita/math/presen/Dag2005_Ogita.pdf).
This assumes round-to-nearest and gradual underflow, with overflow rejected;
residual combinations and elementary functions still round. No arbitrary-length
exactness or universal doubled-precision guarantee is claimed. Eunomia's F16
and Bf16 operators round back into the selected scalar after every operation.
Searches of Leto, Eunomia and Hermes found no existing compensated-sum provider.

A single raw scalar message loses `ln(2)` beside a large negative offset and
can therefore return posterior one where two equally weighted paths require
one half. Common per-frame centering alone also loses corrections in states far
below that frame's maximum. Retaining each message's residual until centered
posterior exponentiation closes both cases. Dividing the loss and seeded
occupancy by each divisor in sequence avoids an underflowed reciprocal product.

Forward state stores alpha, suffix beta, extended labels and sample ranges.
Alpha includes the current emission; beta excludes it. Thus their sum minus the
sample log-likelihood is posterior log occupancy, and backward needs no saved
input log-probability copy. Space is linear in the checked sum of each sample's
valid frames times extended target length; allocation failures are typed.

The consumer owns autograd nodes and adapts views. It consumes the loss/state
through its operation seam rather than importing the recurrence into its graph.

## Derivative and failure semantics

Backward adds `-upstream * posterior / (batch * max(target_length,1))` to a
borrowed mutable gradient view. This differentiates independent log inputs;
composition with log-softmax yields `probability - posterior`. Graves equation
16 differentiates logits, not independent log inputs. PyTorch exposes the
logit-form derivative at its normalized log-input boundary; agreement through
log-softmax does not imply equality for leaf log inputs.

Impossible alignment derivatives are undefined, so backward returns a typed
error naming the sample before any destination write. The same guarantee holds
for invalid mutable layouts, nonfinite seeds, and unrepresentable additions.
A class-sized scratch row aggregates posterior states. One complete read-only
pass checks updates; the identical second pass applies them. Padded gradient
coordinates remain unchanged. State is private so callers cannot invalidate
the cached recurrence between passes.

## Alternatives and verification

Retaining Coeus's fixed-f64 recurrence is rejected: it violates native scalar
arithmetic and leaves the CPU provider without ownership. Reusing cross-entropy
is mathematically insufficient: temporal path collapsing requires its own
recurrence. Retaining input probabilities for backward is unnecessary under the
suffix-beta convention. No accelerator implementation is claimed by this item.

Tests enumerate short paths independently of the recurrence across f32, f64,
F16 and Bf16, alongside explicit one/two-frame loss and gradient oracles,
repeated labels, mixed empty targets, impossible paths, padding, seeded additive
gradients and failure-atomic boundaries. Focused nextest and release tests use
the committed budgets; doctests and strict lint/doc checks cover the public API.
No performance claim follows from these correctness gates.
