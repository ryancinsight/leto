# ADR 0006: Non-symmetric eigensolver track (Hessenberg → Francis QR)

- Status: Accepted (phased)
- Date: 2026-06-15
- Class: [major]

## Context

leto has a symmetric eigensolver (Jacobi). The non-symmetric eigenvalue problem
(nalgebra `Schur`, `eigenvalues`, `complex_eigenvalues`) is the remaining large
linalg gap. Real matrices can have complex-conjugate eigenvalue pairs, so the
target is the **real Schur form** `A = Q T Qᵀ` with `Q` orthogonal and `T`
quasi-upper-triangular (1×1 blocks for real eigenvalues, 2×2 for complex pairs).

The standard algorithm is two phases:
1. **Hessenberg reduction** `A = Q H Qᵀ` (Householder) — makes the subsequent
   iteration `O(n²)`/step instead of `O(n³)` and is preserved by it.
2. **Francis double-shift QR iteration** on `H` — bulge-chasing with implicit
   shifts, deflation, and 2×2 block standardization, converging to `T`.

## Decision

Deliver the track in phases, each correct and verified on its own, rather than
landing a single large, hard-to-verify drop:

- **Phase 1 (this ADR, done):** Hessenberg reduction. A self-contained,
  classically-proven Householder algorithm with a clean contract
  (reconstruction, orthogonality, structure) — verifiable independently of the
  iteration. Leaf hierarchy `linalg/hessenberg/{mod, householder, reduce}.rs`.
- **Phase 2 (next):** Francis double-shift QR producing the real Schur form and
  the eigenvalue list (real + complex). It consumes Phase 1's `H`/`Q`. This is
  the intricate part (deflation criteria, exceptional shifts, 2×2 block
  eigenvalue extraction) and gets its own focused implementation + adversarial
  differential tests against nalgebra `complex_eigenvalues`.

Rationale for phasing: the Francis QR is notoriously easy to get subtly wrong;
shipping Hessenberg first (correct, foundational, independently tested) is the
meticulous decomposition of the work and de-risks Phase 2, which builds on a
verified base. This honors correctness-over-breadth rather than rushing a
possibly-buggy full eigensolver.

## Consequences

- New surface: `hessenberg` / `HessenbergDecomposition` /
  `MatrixDecompose::hessenberg`. Verified on the convention-independent contract
  (`H` is unique only up to reflector signs) plus orthogonal-similarity
  invariants (trace, Frobenius) and a nalgebra Frobenius-norm tie.
- The Householder reflector primitive is currently local to
  `linalg/hessenberg/householder.rs`; QR's packed-reflector scheme differs, so a
  shared core is deferred (recorded in `gap_audit.md`) rather than forced.
- Phase 2 will add `schur` / `eigenvalues` / `complex_eigenvalues` and close the
  last major nalgebra dense-LA gap.
