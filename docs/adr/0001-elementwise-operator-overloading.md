# ADR 0001: Elementwise operator overloading on `Array`

- Status: Superseded by ADR 0004 (2026-06-15)
- Date: 2026-06-10
- Class: [arch]

## Context

The std-trait integration mandate calls for domain types to implement the std
operator traits their semantics support (`Add`, `Sub`, `Mul`, `Div`, `Neg`).
Apollo and Coeus would benefit from `a + b` / `a * scalar` ergonomics on Leto
arrays.

`Array<T, S, N>` is defined in the `leto` core crate. The elementwise math
kernels (`binary_map`, SIMD/parallel dispatch) live in `leto-ops`. `leto` core
is deliberately independent of `hermes`/`moirai` so layout and storage compile
without the compute stack.

The orphan rule blocks the obvious placement: `leto-ops` can implement neither
`core::ops::Add` (foreign trait) for `leto::Array` (foreign type). Only the
`leto` crate may implement std operators for `Array`, but the addition kernel
it would need lives downstream in `leto-ops`.

## Options

1. Implement operators in `leto` core using a scalar-only fallback kernel
   (duplicating the elementwise loop in core; SIMD/parallel paths stay in
   leto-ops and are unreachable from operators).
2. Move the scalar fallback elementwise kernels into `leto` core; `leto-ops`
   keeps only the SIMD/parallel-accelerated paths and reduction/matmul. Core
   gains a minimal arithmetic surface with no `hermes`/`moirai` dependency.
3. Do not add operator overloading; keep the explicit `add(&a, &b, &mut out)` /
   `scalar_map::<AddOp>` API as the single arithmetic entry point.

## Decision

Defer. Adopt option 3 for now and revisit option 2 when a consumer has a
concrete ergonomic driver. Reasons:

- Option 1 splits one operation family across two crates with divergent
  performance characteristics (operators slow, free functions fast) — a
  correctness-of-expectation hazard and an SSOT violation.
- Option 2 is the architecturally sound target but is an [arch] move of the
  elementwise seam into core, affecting the crate-boundary contract. It should
  be done deliberately with its own change, not bundled into gap remediation.
- The current explicit API is complete and zero-cost; no consumer is blocked.

## Consequences

- `scalar_map`/`scalar_map_into` (added in 0.3.0) cover array–scalar arithmetic
  without operators.
- When option 2 is taken, it is a [minor] additive change (operators are new
  surface) but relocating kernels to core is the [arch] part requiring this ADR
  to be superseded.
