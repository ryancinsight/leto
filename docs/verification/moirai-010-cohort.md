# Moirai 0.10 provider continuation

This change continues Leto #362 and preserves its reserved 0.47.0 release,
approved borrowed CSR implementation and original owner commits. Current main
`d3feb7162a51e2e44d2243919e96df1624ab9116` is merged, retaining the separately
published Python 0.46.1 repair and its existing workflows and budgets.

The canonical runtime requirement advances to Moirai 0.10.0. Normal Cargo
resolution selects the current default
`68466225913301d30281b134d42b17c604baf78e` for every Moirai package, with no
parallel 0.9 identity, manifest revision pin or path override. The other
canonical providers resolve to their current defaults; Hermes advances to
`ac60146a` and Themis to `5595fc0a` without changing the scalar cohort.

On the ARM64 Mac with Rust 1.97.0:

- All 1,202 core/ops native tests pass with zero skips.
- All 1,190 applicable feature-disabled core/ops tests pass with zero skips.
- Both feature modes pass strict all-target Clippy. Format and the canonical
  standalone locked-source check pass.
- All 38 core/ops doctests pass, retaining one existing ignored example;
  warning-denied documentation passes.

The existing tests check computed sparse/dense values, signed index rejection,
initialization, native stored-order arithmetic, caller-owned scheduling and
completion during unwinding. No numerical kernel or assertion changes belong
to this dependency migration. The original CSR memory/performance receipts
remain scoped to their recorded revisions; they are not new Moirai 0.10 memory
or universal performance certification.

The exact-source API comparison, final hosted receipts and ordered provider /
consumer integration remain release gates. No merge, release, credential or
service action is implied by these local checks. Coeus Miri remains deferred
and unverified in the shared downstream migration.
