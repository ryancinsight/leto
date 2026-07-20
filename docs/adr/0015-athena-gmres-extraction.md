# ADR 0015: Move restarted GMRES orchestration to Athena

- Status: Accepted
- Date: 2026-07-19
- Class: [major] [arch]

## Context

After the PCG extraction, `leto-ops` still exposed `gmres` and `GmresResult`.
That recurrence owned Arnoldi basis construction, orthogonalization,
least-squares updates, convergence policy, and solution updates beside Leto's
array and sparse-kernel substrate. Athena now provides one restarted,
right-preconditioned GMRES recurrence over both Leto CPU arrays and Hephaestus
WGPU buffers.

No Atlas consumer outside Leto calls the public GMRES surface at this
revision. Athena's replacement passes generic Leto CPU conformance for `f32`
and `f64`, forced multi-cycle restart, termination and dimension error cases,
post-initialization allocation measurement, and a real Hephaestus WGPU
nonsymmetric-system test.

## Decision

- Remove `leto_ops::gmres` and `leto_ops::GmresResult` without a compatibility
  re-export or forwarding wrapper.
- Retain `CsrMatrix`, `spmv_into`, arrays, views, reductions, and
  decompositions in Leto.
- Let Athena own the single backend-neutral recurrence and its
  `Gmres<B, const RESTART: usize>` zero-sized policy.
- Let caller-owned `GmresWorkspace<B, RESTART>` retain the `RESTART + 1`
  orthonormal basis vectors, `RESTART` preconditioned vectors, Hessenberg
  storage, and scalar rotations across solves.
- Keep matrix storage and multiplication in provider backends: Athena's Leto
  backend borrows `ArrayView1`/`ArrayViewMut1`; its Hephaestus backend operates
  on resident WGPU buffers through prepared kernels.

## Migration

Callers replace the allocating helper with explicit Athena policy and
caller-owned state:

```text
leto_ops::gmres(matrix, rhs, restart, max_iterations, tolerance)

becomes

athena_core::Gmres::<athena_leto::LetoBackend<T>, RESTART>::solve_into(
    backend,
    operator,
    preconditioner,
    right_hand_side,
    solution,
    workspace,
    convergence_policy,
)
```

The replacement makes the initial guess, right preconditioner, workspace
lifetime, restart width, absolute and relative tolerances, observation
schedule, and numerical termination explicit.

## Consequences

- This is a public breaking removal and requires the next release boundary.
- Leto owns one bounded context for arrays, sparse storage, and kernels.
- Athena owns one recurrence that monomorphizes for each backend and restart
  width; adding a backend does not clone the algorithm.
- Restart storage is bounded by a structural const generic, while all
  per-solve vector storage is allocated only when the workspace is constructed.

## Verification

- residue scans contain no `GmresResult`, `leto_ops::gmres`, or solver module;
- `leto-ops` formatting, warning-denied Clippy, nextest, doctest, and rustdoc
  gates pass;
- Athena CPU and WGPU conformance pass before this removal;
- Athena's GMRES contract and reference are recorded in
  [ADR 0002](https://github.com/ryancinsight/athena/blob/main/docs/adr/0002-restarted-gmres-contract.md).
