# ADR 0020: Leto release scope for 1.0

- Status: Accepted
- Date: 2026-11-10
- Class: `[arch]` `[minor]`

## Context

Leto (workspace 0.48.0) provides N-dimensional strided array layouts, views,
storage primitives, dense and sparse operations, and linear algebra for the
Atlas stack. The backlog contains open correctness, performance, and feature
items, and no milestone ladder exists to 1.0.

## Decision

Adopt a three-milestone ladder to 1.0 recorded in `backlog/milestones.md`:

- **M-array-core** (0.49.0): users construct arrays, views, slicing, and
  layouts with documented core contracts.
- **M-ops-linalg** (0.50.0): users perform elementwise operations, reductions,
  dense/sparse linear algebra, and solver primitives with verified behavior.
- **M-1.0** (1.0.0): stable array and operations surface (authority-gated)
  with documented contracts and release gates satisfied.

## Alternatives

- Plan 1.0 as a single release: rejected because open correctness and
  verification work remains across core and ops crates.
- Keep roadmap implicit in backlog items: rejected because stack policy
  requires a runnable capability-oriented milestone ladder.

## Consequences

- Open items carry `milestone:` assignments and feature items name the exit
  check they serve.
- Milestones close only when exit checks pass on fetched default commits; 1.0
  release/publication remains authority-gated.
- This ADR is rewritten in place as scope evidence changes.
