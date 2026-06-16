# ADR 0005: Rank-revealing SVD via one-sided Jacobi

- Status: Accepted
- Date: 2026-06-15
- Class: [major] (new decomposition contract; additive surface)
- Update 2026-06-16: `svd_decompose` and `singular_values` are no longer
  Gram-backed. The full-rank-only default path now uses the bidiagonal-QR
  implementation (`svd/bidiagonal_qr.rs`) for accuracy and performance, while
  this ADR's one-sided Jacobi path remains the rank-revealing implementation for
  rank-deficient SVD and pseudoinverse.

## Context

`svd_decompose` (the Gram-matrix path, ADR-era Stage A1) forms `AᵀA` (or `AAᵀ`)
and diagonalizes it with the symmetric Jacobi eigensolver, deriving the missing
factor as `U = A V Σ⁻¹`. This is accurate for full-rank inputs but **rejects
rank-deficient matrices**: a zero singular value makes `Σ⁻¹` undefined, so the
null-space singular vectors cannot be recovered. That leaves two parity gaps vs
nalgebra: a rank-revealing SVD and a rank-deficient Moore-Penrose pseudoinverse.

Forming the Gram matrix also squares the condition number (`κ(AᵀA) = κ(A)²`),
which loses ~half the significant digits on ill-conditioned inputs.

## Options

1. **Golub-Kahan bidiagonalization + implicit-shift QR** — the LAPACK `gesvd`
   route. Most general and fastest asymptotically, but a large, intricate
   implementation (Householder bidiagonalization, Wilkinson shifts, deflation,
   careful handling of tiny diagonal/superdiagonal entries).
2. **One-sided Jacobi SVD** — orthogonalize the columns of `A` by a sweep of
   Jacobi rotations; the converged column norms are the singular values, the
   normalized columns are `U`, and the accumulated rotation matrix is `V`.
3. Keep Gram-only; leave rank-deficient SVD/pinv unimplemented.

## Decision

Adopt option 2, one-sided Jacobi, as the rank-revealing path. At the time of
this ADR it lived alongside the existing Gram path; as of 2026-06-16 that
full-rank-only path is replaced by bidiagonal QR without changing this ADR's
rank-revealing decision:

- It is **rank-revealing by construction**: rank-deficient columns converge to
  zero norm, surfacing `σ = 0` honestly; `V` stays fully orthonormal (it is a
  product of rotations), so no fabricated null-space vectors are needed.
- It is **more accurate** than Gram — it never forms `AᵀA`, so it works in the
  native precision of `A` without squaring the condition number.
- It is **far simpler and more auditable** than Golub-Kahan, with a clean
  monotone-convergence proof, which suits a from-scratch, theorem-documented
  implementation. (Golub-Kahan can be revisited later purely as a perf
  optimization for large matrices, behind the same `SvdDecomposition` contract.)

Wide inputs (`m < n`) are handled by decomposing `Aᵀ` and swapping `U ↔ V`
(`A = (Aᵀ)ᵀ`), so one code path covers all shapes (DRY).

The Moore-Penrose pseudoinverse is **unified** onto this path: `pinv` becomes
rank-revealing (`A⁺ = Σ_{σᵢ>τ} σᵢ⁻¹ vᵢ uᵢᵀ`), matching nalgebra's single
`pseudo_inverse`. The full-rank `svd_decompose` keeps its explicit
rank-deficiency rejection, but its implementation is now bidiagonal QR rather
than Gram.

## Structure (deep vertical hierarchy)

`linalg/svd.rs` is refactored into a leaf-module tree (SRP/SoC):

```text
linalg/svd/
  mod.rs           SvdDecomposition struct, shared validation/tolerance, re-exports
  bidiagonal_qr.rs full-rank/default thin SVD and singular values
  jacobi.rs        one-sided Jacobi rank-revealing SVD (svd_rank_revealing)
  pseudoinverse.rs Moore-Penrose pinv (rank-revealing, via jacobi)
```

The module path `linalg::svd` and all public names are unchanged.

## Consequences

- New public surface: `svd_rank_revealing` / `svd_rank_revealing_with_tolerance`,
  and `MatrixDecompose::svd_rank_revealing`. `pinv` now handles rank-deficient
  input (strictly more capable; full-rank behavior unchanged).
- Verified against nalgebra `SVD`/`pseudo_inverse` plus oracle-independent
  invariants: reconstruction `A = U Σ Vᵀ`, orthonormality `VᵀV = I`, and the
  Moore-Penrose conditions `A A⁺ A = A`, `A⁺ A A⁺ = A⁺`.
- Convergence is capped at a fixed sweep count with a relative off-orthogonality
  threshold; the cap is a safety bound, not a tuned tolerance.
- Evidence tier: monotone-convergence proof sketch in rustdoc + differential and
  property tests. No machine-checked proof.
