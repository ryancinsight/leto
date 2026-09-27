<a id="adr-0034"></a>

# ADR 0034: Atlas array/linalg layer boundary

Status: Accepted (retroactive — as-built)

Date recorded: 2026-09-27 (decision predates this record; carried forward
from `gap_audit.md`'s "Layer Boundary Decision" entry, undated in-repo since
at least 2026-06-15).

## Context

Leto, Coeus, Hephaestus, and Apollo each touch array-shaped data and needed
one non-overlapping ownership line so operation families are not duplicated
or misplaced across crates (architecture_scoping: canonical component homes).
No ADR previously recorded this boundary; it existed only as a `gap_audit.md`
entry, which is the wrong home for a standing cross-cutting architectural
decision (gap_audit.md holds open risks, not accepted boundaries).

## Decision

- **Leto** owns the non-differentiable array substrate: layout/strides,
  storage, views, slicing, broadcasting, elementwise binary/unary math,
  reductions, matmul (including batched), shape ops (concat/pad/split),
  dense linear algebra, scaled dot-product attention on CPU, and narrow CPU
  CSR sparse-dense parity kernels.
- **Coeus** owns autodiff, NN orchestration, optimizer fusion, and backend
  selection.
- **Hephaestus** owns accelerator attention.
- **Apollo** owns transform kernels; FFT stays in Apollo (Coeus routes
  `fft_1d` there rather than duplicating it).

## Consequences

- A capability request that crosses this boundary is routed to its owning
  crate rather than implemented locally (upstream ownership,
  architecture_scoping).
- Leto's linalg surface accepts new routines only with a named consumer
  driver and a differential oracle, superseded for full parity work by
  `docs/completeness/PLAN.md`'s literal-parity scope (which re-opens some
  routines this boundary alone would exclude).

## Evidence

Verified as-built against `crates/leto`, `crates/leto-ops` (array/dense-linalg
surface, no autodiff/accelerator/FFT code), and the Coeus/Hephaestus/Apollo
repository boundaries at the time of this record.
