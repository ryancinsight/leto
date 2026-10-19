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


## Original CSR audit source record

The following records the original CSR source `b7670636`. Its memory and
performance evidence remains historical and does not certify the current
Moirai 0.10 provider cohort.

Checked signed/stored-order views, separate native arithmetic, complete row
writes and synchronous dispatch replace duplicated consumer forward math.
Checked fixed metadata and fused row validation preserve error priority and
one recurrence. Final source `b7670636` passes 1202 native cases, both strict
feature modes, docs/API, twenty memory cases in each strict/isolation model
and ten ARM32 cases per model. The [original CSR evidence](csr-forward-contract.md)
replaces superseded proposal-only and intermediate acceptance claims at that
source revision.
