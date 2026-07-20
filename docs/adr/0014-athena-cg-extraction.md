# ADR 0014: Move conjugate gradient orchestration to Athena

- Status: Accepted
- Date: 2026-07-19
- Class: [major] [arch]

## Context

`leto-ops` exposed `cg` and `CgResult` beside CSR storage and SpMV. Atlas now
requires one PCG recurrence to execute over both Leto CPU arrays and
Hephaestus WGPU buffers. Keeping the recurrence in Leto would make the host
array provider own cross-backend solver policy.

No Atlas consumer outside Leto calls the public CG surface at this revision.
Athena's replacement passes a generic Leto CPU suite for `f32` and `f64`, a
zero-allocation post-initialization measurement, and a real Hephaestus WGPU
manufactured-system test.

## Decision

- Remove `leto_ops::cg` and `leto_ops::CgResult` without a compatibility
  re-export or forwarding wrapper.
- Keep `CsrMatrix`, `spmv_into`, `dot`, arrays, views, reductions, and
  decompositions in Leto.
- Let Athena's Leto backend map its GAT views directly to `ArrayView1` and
  `ArrayViewMut1`.
- Track restarted GMRES as a separate complete replacement increment, recorded
  by ADR 0015, so each public breaking removal has its own evidence.
- Move the slice-based SpMV leaf used by GMRES into the canonical SpMV module
  rather than retaining it in a deleted CG module.

## Migration

Callers replace the allocating zero-initial-guess helper with explicit Athena
policy and caller-owned storage:

```text
leto_ops::cg(matrix, rhs, max_iterations, tolerance)

becomes

athena_core::Cg::<athena_leto::LetoBackend<T>>::solve_into(
    backend,
    operator,
    preconditioner,
    right_hand_side,
    solution,
    workspace,
    convergence_policy,
)
```

The new contract makes the initial guess, preconditioner, workspace lifetime,
absolute/relative tolerance, and numerical termination explicit.

## Consequences

- This is a public breaking removal and requires the next release boundary.
- Leto returns to one bounded context for sparse storage and kernels.
- Athena can add a backend without cloning the recurrence.
- Restarted GMRES was removed in the subsequent ADR 0015 increment.

## Verification

- residue scans contain no `CgResult`, `leto_ops::cg`, or CG module export;
- `leto-ops` formatting, warning-denied Clippy, nextest, doctest, and rustdoc
  gates pass;
- Athena CPU and WGPU conformance passed before this removal.
