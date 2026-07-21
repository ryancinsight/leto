# ADR 0016: Own the typed Cartesian Laplacian stencil

- Status: Accepted
- Date: 2026-07-20
- Class: [minor] [arch]

## Context

Hephaestus owns a WGPU five-point Laplacian, while CFDrs still implemented the
CPU form directly and carried a second test-only reference formula. Boundary
codes, spacing validation, and operator polarity therefore had multiple owners.
The CPU solver applied `-nabla^2`, while its GPU counterpart applied `nabla^2`.

## Decision

- Leto owns `Laplacian2D<T>`, `BoundaryCondition`, `LaplacianPolarity`, and
  typed validation of Aequitas `Length<T>` spacing.
- Leto Ops owns the native-precision, allocation-free CPU evaluation into a
  caller-provided array view.
- Hephaestus derives its POD dispatch parameters from the Leto contract and
  retains only device dispatch and WGSL execution.
- Consumers select polarity explicitly at the operation boundary. No consumer
  keeps a copied stencil or numeric boundary code.

## Rejected alternatives

- A CFDrs-local shared helper keeps the provider contract consumer-owned.
- A Hephaestus-only contract gives CPU execution no canonical owner.
- Converting the generic CPU operation through `f32` would make the scalar
  parameter false and change `f64` results.

## Consequences

- `leto` gains an Aequitas dependency for the public dimensional boundary.
- `leto-ops::laplacian_2d_into` evaluates `f32` and `f64` in native precision.
- CPU and GPU implementations share grid, boundary, spacing, and polarity
  semantics without a compatibility adapter.

## Verification

- a generic `f32`/`f64` regression checks `-nabla^2(x^2 + 3y^2) = -8` on every
  point of an anisotropic Neumann grid;
- package checks and warning-denied Clippy cover all targets and features;
- configured Nextest covers the new operation through the committed budgets.
