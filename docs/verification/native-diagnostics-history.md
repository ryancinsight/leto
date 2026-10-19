# Historical native diagnostic correction

<a id="LETO-NOPARALLEL-CLIPPY-2026-10-15"></a>

## LETO-NOPARALLEL-CLIPPY-2026-10-15 — Restore strict diagnostics without parallel [patch] — fixed

- priority: verification; scope: `application/linalg/lu_batch.rs:112`.
- Evidence: strict `leto-ops --all-targets --no-default-features` Clippy rejects
  the unchanged explicit `match` on `lu_decompose` with `clippy::question_mark`.
  Default-feature strict Clippy passes; both sparse feature modes pass tests.
- Closed: the implementation propagates with `?`, preserves the computed
  decomposition assignment, and passes real batched-LU tests and strict
  diagnostics in both feature modes.


Original CSR audit source record:

one recurrence. Final source `b7670636` passes 1202 native cases, both strict
