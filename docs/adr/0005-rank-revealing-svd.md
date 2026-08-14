# ADR 0005: Rank-revealing SVD

- Status: Accepted
- Date: 2026-06-15
- Class: [major] (decomposition contract; rank deficiency is data, not an error)

## Revisions

- **2026-08-14 — evidence gap closed; zero-diagonal chase added.** The decision
  below stands, but the evidence table that justified it was gathered at `f64` on
  *near*-deficient matrices only, and that is a different case from exact rank
  deficiency. Exact deficiency can leave an **exact zero on the diagonal** of the
  bidiagonal factor, which a shifted QR step cannot deflate: the implicit `BᵀB`
  is singular, the Wilkinson shift takes its nonzero eigenvalue, and the sweep
  converges to a fixed point at `d = 0`, `e ≠ 0` that `scale + |e| == scale`
  never accepts. Whether the zero is exact depends on the rounding of the
  bidiagonalization and therefore on the precision, so the gap was specifically
  **exact rank deficiency at `f32`**: `[[1,2],[2,4],[3,6]]` bidiagonalizes to
  `d = [−3.7416573, 0]`, `e = [7.4833145]` at `f32` and spun to the
  4000-iteration cap, surfacing in `hephaestus-cuda` as
  `SVD failed: … QR failed to converge`. Leto's own suite missed it because it
  tested rank deficiency at `f64` and `f32` genericity only on full-rank input —
  never together. Fixed structurally, not by widening the deflation tolerance
  (which would have traded every singular value's accuracy for a structural
  gap): a negligible diagonal inside the active block is now chased out with
  Givens rotations — left rotations along the row for an interior zero, right
  rotations up the trailing column for a zero at the block's bottom — after which
  the block splits and normal deflation proceeds. Tests now run exact
  rank-deficiency at **both** precisions across tall, wide, square, rank-1,
  rank-2-of-3 and rank-2-of-4, asserting analytic σ, reconstruction and
  orthonormality of `U` *and* `V` against a derived `8·max(m,n)·ε·‖A‖₂` bound,
  plus unit tests driving each chase branch directly from a constructed
  bidiagonal. The orthonormality assertions are deliberate: they pin the property
  that retiring Jacobi was chosen to protect.
- **2026-08-13 — decision re-derived; one-sided Jacobi retired.** The original
  decision selected one-sided Jacobi *against the Gram-matrix path*, on the
  grounds that Gram could not produce null-space vectors. The Gram path was
  deleted on 2026-06-16 and replaced by bidiagonal QR, which removed that
  premise. The record was patched with a note rather than re-derived, leaving
  this ADR justifying a second implementation by comparison with a path that no
  longer existed. Re-derived below against the surviving alternative: bidiagonal
  QR is itself rank-revealing, so `svd/jacobi.rs` is deleted, `pinv` moves onto
  the bidiagonal path, and the full-rank rejection policy is removed. The
  decomposition surface collapses to a single entry point (`svd_decompose`);
  `svd_rank_revealing`, `svd_via_bidiagonal`, `svd_decompose_with_tolerance` and
  `MatrixDecompose::svd_rank_revealing` are removed with it. Superseded content
  is in git history; this file states only the decision now in force.

## Context

Leto is the linear-algebra SSOT for the Atlas stack, so a duplicated
decomposition is duplicated in every consumer. The SVD surface carried two
implementations behind one `SvdDecomposition` contract and one validator:
`svd/bidiagonal_qr.rs` (Golub–Reinsch implicit-shift) and `svd/jacobi.rs`
(one-sided Jacobi, ~200 lines).

They were separated by policy, not capability. The bidiagonal path *chose* to
reject rank-deficient input while its own documentation stated that it "handles
rank-deficient input (zero singular values emerge)". Rank deficiency is a
property of the data that the decomposition measures; refusing to return it
forces callers to a second entry point for an answer the first one already
computed.

## Options

1. **Keep both paths.** Justified only if Jacobi is more capable or more
   accurate on some class the bidiagonal path handles poorly.
2. **Single bidiagonal-QR path, rank revealed in `Σ`.** Delete Jacobi, move
   `pinv` across, drop the rejection policy.
3. Keep both implementations but unify the names behind one entry point — a
   forwarding shim, prohibited: it preserves the duplication and adds an alias.

## Decision

Adopt option 2. Option 1 was tested before deletion rather than assumed, since
one-sided Jacobi has a genuine published accuracy advantage (Demmel & Veselić,
*Jacobi's method is more accurate than QR*, SIMAX 13(4), 1992) on column-scaled
matrices, and a real advantage would have been a capability difference.

Measured, `f64`, both paths on identical input:

| Class | bidiagonal QR | one-sided Jacobi |
| --- | --- | --- |
| Exact rank deficiency (tall/wide/square, rank-1 and 5×4 rank-2) | σ correct; recon ≤ 3.1e-15; `U`,`V` orthonormal to 8.9e-16 | σ correct; **`UᵀU − I` = 1.0** (zero column); wide input puts the defect in `V` |
| σ = 1e-14, below the 1e-12 tolerance | recon 1.6e-30 | **recon 1.0e-14** — the whole singular value dropped |
| Graded triangular, κ ≈ 1e16 | σ = [1.732051, 8.164966e-9, 7.071068e-17] | identical to 7 digits |
| Column-scaled `B·D`, κ ≈ 1e17 (Demmel–Veselić class) | σ₃ = 5.4433105395182064e-17 | σ₃ = 5.443310539518174e-17 (6e-15 relative) |
| Column-scaled `B·D`, κ ≈ 1e21 | σ₃ = 5.443310539518211e-21; ∏σ vs det 7.4e-15 | σ₃ = 5.4433105395181706e-21; ∏σ vs det 7.0e-16 |

The published Jacobi advantage does not materialize here: on the column-scaled
class the two paths agree to 15 significant digits on the smallest singular
value, and both match the determinant oracle to a few ulps. There is no class
where the bidiagonal path fails and Jacobi succeeds.

The comparison instead runs the other way. Jacobi normalizes converged columns
only where `σⱼ > τ`, leaving the rest of `U` zero, which costs it two contract
properties the bidiagonal path holds unconditionally:

- **Orthonormal factors at deficient rank.** Bidiagonal `U` and `V` are
  accumulated products of Householder reflectors and Givens rotations;
  orthogonality of a product of orthogonal factors does not depend on the
  singular values being nonzero, so `UᵀU = VᵀV = I` at every rank. Jacobi
  returns a zero column instead of a null-space basis vector.
- **Reconstruction of small singular values.** Any `σ` at or below the absolute
  tolerance loses its `U` column entirely, so `A = U Σ Vᵀ` fails by exactly the
  dropped `σ` — 1e-14 in the measured case, versus 1.6e-30.

The second point also falsified this ADR's own prior claim that on the Jacobi
path "`V` is always fully orthonormal": for wide input the kernel transposes and
swaps the factors, so the zero column lands in `V`.

Consequently the surviving path is strictly more capable, and the rejection
policy has nothing left to protect. `svd_decompose` accepts every finite
non-empty matrix and reports rank in `Σ`; a caller needing a full-rank guarantee
tests `singular_values.last()` against its own noise floor, which the library
cannot know. `pinv` keeps its relative cutoff (`τ·σ_max`, `τ = 1e-12`) and is
now sound for a stronger reason: every retained direction has a genuine
orthonormal `U` column behind it.

Wide inputs (`m < n`) are handled by decomposing `Aᵀ` and swapping `U ↔ V`, so
one code path covers all shapes.

## Structure

```text
linalg/svd/
  mod.rs           SvdDecomposition contract, shared validation, re-exports
  bidiagonal_qr.rs the SVD: svd_decompose (thin, any rank) and singular_values
  pseudoinverse.rs Moore-Penrose pinv on svd_decompose
```

`singular_values` stays a separate entry point: it is the same algorithm under
`qr_iterate::<_, false>`, a zero-cost const-generic specialization that skips
`U`/`V` accumulation. That is one implementation with a compile-time switch, not
a second implementation.

## Consequences

- Public surface shrinks to `svd_decompose`, `singular_values`, `pinv` and
  `MatrixDecompose::{svd, singular_values}`.
- Breaking for downstream consumers of the removed names. `svd_rank_revealing`
  callers migrate to `svd_decompose` with no behavioral loss — the replacement
  is strictly more capable. Callers that relied on `svd_decompose` returning
  `Err` for rank-deficient input must test `Σ` explicitly; a silent change from
  `Err` to a zero singular value is the migration risk to flag. Known consumers:
  `hephaestus-{cuda,metal,rocm}`, which re-export `svd_rank_revealing` and
  delegate to it.
- Verified by oracle-independent invariants: reconstruction `A = U Σ Vᵀ`,
  orthonormality `UᵀU = VᵀV = I` (now asserted at deficient rank, which the
  Jacobi path could not satisfy), sub-tolerance σ reconstruction, the
  Moore-Penrose conditions, and cross-path σ agreement between the values-only
  and accumulating instantiations.
- Evidence tier: implicit-shift QR convergence proof sketch in rustdoc, plus
  differential and property tests. No machine-checked proof.
