# ADR 0010: Blocked-reflector (compact-WY) vectorization for eig/SVD

- Status: Accepted (phased; Phases 0–1 done, Phase 2 measured-valueless/reverted,
  Phase 3 SVD + eig done — both disparities resolved without the full multishift
  GEMM rewrite, which is now an optional very-large-`n` lever only)
- Date: 2026-06-16
- Class: [major] (post-1.0 the trailing-update path changes observable timing and
  the internal reduction/iteration kernels are restructured; the public surface
  and numerics are preserved, verified by the existing differential batteries)

## Context

After the apply-window confinement (ADR-adjacent, `gap_audit.md`) and the
`axpy_slice` vectorization of the contiguous reflector sweeps, the dense
non-symmetric eigensolver and the SVD retain a residual disparity to nalgebra
(LAPACK): `eig` ≈ 4.4×, full `svd` ≈ 3.4×, `singular_values` ≈ 2.3× at 64×64.

The residual is **structural, not a tuning gap**. Both algorithms spend their
dominant time applying *single* Householder reflectors of width 2–3:

- **eig** — the Francis bulge chase calls `apply_right` (`H ← H P`) row by row;
  each row touches only `len ∈ {2,3}` contiguous columns. The long dimension is
  the row index, which is strided by `n`, so no contiguous SIMD span exists. The
  `apply_left` direction was vectorized (contiguous column span), leaving the
  right-apply as the bottleneck.
- **SVD** — the Golub–Kahan bidiagonalization applies a sequence of single
  reflectors to the trailing matrix; the implicit-shift bidiagonal QR sweep
  applies a sequence of *single* `2×2` Givens rotations down strided columns. A
  single rotation has no width to vectorize.

A single narrow reflector/rotation is intrinsically memory-latency-bound: `O(n)`
work touching `O(n)` strided cache lines with `O(1)` arithmetic intensity. The
roofline ceiling is bandwidth, and SIMD cannot raise it for a width-3 operator.

LAPACK does **not** apply reflectors one at a time on these paths. It aggregates
`nb` consecutive reflectors into a **block reflector** and applies the block as a
matrix–matrix product (GEMM), which is compute-bound and routes through the tuned
`tiled_gemm` SIMD micro-kernel leto already owns. This raises the arithmetic
intensity from `O(1)` to `O(nb)` and is the standard route to BLAS-3 performance.

## Theorem (compact-WY representation, Schreiber–Van Loan 1989)

Let `Q = H₁ H₂ … H_r` be a product of `r` Householder reflectors,
`H_i = I − β_i v_i v_iᵀ`, with the `v_i` stored as the columns of `V ∈ ℝ^{m×r}`
(`v_i` zero above its pivot). Then there is an upper-triangular `T ∈ ℝ^{r×r}` with

    Q = I − V T Vᵀ.

`T` is built columnwise by the recurrence `T_{11} = β_1` and, for `i = 2…r`,

    T_{1:i-1, i} = −β_i · T_{1:i-1,1:i-1} · (V_{:,1:i-1}ᵀ v_i),   T_{ii} = β_i.

*Proof (induction on `r`).* For `r = 1`, `Q = I − β_1 v_1 v_1ᵀ = I − V T Vᵀ` with
`V = v_1`, `T = β_1`. Assume `Q_{r-1} = H_1…H_{r-1} = I − V_{r-1} T_{r-1} V_{r-1}ᵀ`.
Then `Q_r = Q_{r-1} H_r = (I − V_{r-1}T_{r-1}V_{r-1}ᵀ)(I − β_r v_r v_rᵀ)`. Expanding,

    Q_r = I − V_{r-1}T_{r-1}V_{r-1}ᵀ − β_r v_r v_rᵀ
            + β_r V_{r-1}T_{r-1}(V_{r-1}ᵀ v_r) v_rᵀ.

Set `V_r = [V_{r-1} | v_r]` and
`T_r = [[T_{r-1}, −β_r T_{r-1}(V_{r-1}ᵀ v_r)], [0, β_r]]`. Block-multiplying
`V_r T_r V_rᵀ` reproduces exactly the four terms above, so
`Q_r = I − V_r T_r V_rᵀ`, and `T_r` is upper-triangular because `T_{r-1}` is. ∎

### Corollary (BLAS-3 application, the performance lever)
Applying the block to a trailing matrix `C` is three GEMMs and is numerically a
sequence of orthogonal similarities (spectrum/​singular-value preserving):

    Qᵀ C = C − V (Tᵀ (Vᵀ C)),        C Q = C − ((C V) T) Vᵀ.

`Vᵀ C` (or `C V`) is `r×(cols)` — a GEMM; the small `T` product is `r×r·…`; the
rank-`r` correction is the final GEMM. Each routes through `Scalar::tiled_gemm`.
The forward error is the standard blocked-orthogonal bound `O(r·ε‖C‖)`, within the
backward-error tolerance the eig/SVD batteries already assert (see ADR note on the
`8·√(ε‖A‖)` defective-eigenvalue bound).

## Theorem (Phase 2 — blocked two-sided panel reduction, `dlabrd`/`dlahr2`)

Let a panel of `nb` left reflectors form `Q_L = I − V T_L Vᵀ` and `nb` right
reflectors form `Q_R = I − W S Wᵀ` (each compact-WY by the theorem above). The
two-sided trailing update `A ← Q_Lᵀ A Q_R` on the sub-block below/right of the
panel equals applying the `2·nb` single reflectors in their original interleaved
order, and is realised by two rank-`nb` GEMM corrections
`A ← A − V Yᵀ − X Wᵀ`, where `Y = Aᵀ V·(…)` and `X = A W·(…)` are the accumulator
panels built columnwise during the reduction.

*Proof.* Orthogonal-similarity associativity: `Q_Lᵀ A Q_R = (H_{L,nb}…H_{L,1}) A
(H_{R,1}…H_{R,nb})`, and the WY identity collapses each one-sided product to
`I − VT_LVᵀ` / `I − WSWᵀ` (proved above), so the grouped form equals the
sequential interleaving up to the floating-point reorder bound `O(nb·ε‖A‖)`. The
*non-trivial* content — and why a naïve “defer the trailing columns” scheme is
incorrect — is the **look-ahead**: the right reflector at panel column `i` mixes
column `i+1` (still inside the panel) with the *entire* trailing column range, so
column `i+1` must already carry the trailing contribution before its own reflector
is formed. The `X`, `Y` accumulators encode exactly that partial trailing action,
so each in-panel column is reduced as if the full trailing update had run, while
the bulk trailing GEMM is deferred to the panel boundary. The reduction is
therefore mathematically identical to the unblocked sweep (the implementation
oracle), not an approximation of it. ∎

*Correctness gate:* differential reconstruction `A = U B Vᵀ` (bidiagonal) /
`A = Q H Qᵀ` (Hessenberg), `U`/`V`/`Q` orthogonality, and bidiagonal/Hessenberg
structure against the unblocked path, plus the existing SVD/eig batteries.

## Theorem (Phase 3 — small-bulge multishift QR, `dlaqr5`)

A multishift Francis sweep introduces `m/2` `3×3` bulges simultaneously near the
top of the active Hessenberg block and chases them down the band together. The
result equals `m` single-shift Wilkinson steps (one per shift), and the chase of a
*tight chain* of bulges across a window of `nb` rows is a sequence of small
reflector applications confined to that window — accumulated into a compact-WY
block and applied to the off-window portions of the band (and to the Schur-vector
matrix) as GEMMs.

*Proof sketch.* By the implicit-Q theorem (ADR 0006), a Francis sweep with shift
polynomial `p(H) = Π(H − μⱼI)` is determined up to reflector signs by its first
column `p(H)e₁`; chasing the resulting bulge restores Hessenberg form and realises
`Qᵀ H Q` for the `Q` of `p(H) = QR`. Splitting `p` into `m/2` conjugate-pair
quadratics and introducing the corresponding bulges in a packed chain yields the
same cumulative `Q` (same first column `p(H)e₁`), so the multishift sweep equals
the composition of the single steps. Within the chase window the reflectors touch
only `≤ nb` rows/columns; aggregating them by compact-WY turns the
off-window band update and the `Q`-accumulation into GEMMs (BLAS-3), which is the
performance lever for the dominant iteration cost. The numerical reorder stays
within the backward-error bound already adopted for the spectrum. ∎

*Correctness gate:* the hardened 8×8/16×16/defective eigenvalue battery under the
`8·√(ε‖A‖)` backward-error tolerance, plus `A = Q T Qᵀ` Schur reconstruction.

## Decision

Introduce a compact-WY block-reflector seam and route the three reduction/iteration
kernels through it, phased so each phase is independently verified against the
current (correct) unblocked path before the next begins. No public API or numeric
contract changes; the differential batteries are the gate.

### Deep vertical file hierarchy (new leaf module)

```
linalg/
  reflector_block/
    mod.rs       apply_block_left (Phase 1) / apply_block_right (Phase 2) via
                 tiled_gemm + the compact-WY theorem/proof
    accumulate.rs  build_t columnwise from a panel of single reflectors (the
                   Schreiber–Van Loan recurrence) — SSOT for T construction
    panel.rs       (Phase 2) Cow column-panel view: borrow a contiguous panel,
                   materialize a strided one
```

Design-pattern application, with the Phase-1 reality and the deliberate
deferrals (a deferral here is an engineering decision recorded against the
mechanical trigger, not an omission):

- **DIP / SSOT (Phase 1, done):** `apply_block_left` depends only on
  `Scalar::tiled_gemm` (the backend seam), never a concrete SIMD type;
  `accumulate::build_t` is the single `T`-construction site.
- **Monomorphization / zero-cost (Phase 1, done; const-`NB` rejected):** the block
  apply is generic `<T: RealScalar>` and monomorphizes per scalar. A const-generic
  `NB` was **deliberately not used**: panels are variable width (the last panel is
  `cols mod NB`, and the `BLOCK_MIN_ROWS` gate makes the *effective* width runtime),
  so a fixed `NB` would force padding or a separate tail path — runtime `r ≤ NB`
  is the correct, non-cargo-culted choice (const generics buy nothing when the
  dimension is genuinely runtime; over-specializing is the documented anti-goal of
  performance_engineering's instantiation-count rule).
- **ZST / typestate `Side` (Phase 2):** a `Left`/`Right` ZST selecting the apply
  form at compile time is introduced **with** `apply_block_right`, whose only
  consumer is a two-sided reduction (Phase 2). Adding `apply_block_right` + `Side`
  in Phase 1 — with no caller — would be dead code (`-D warnings`) and speculative
  generality; they land when their consumer does.
- **Zero-copy / CoW `panel.rs` (Phase 2):** Phase 1's sole consumer (QR) stores
  reflectors as *strided* columns of a row-major matrix, so its panel is always
  materialized — the borrow arm of a `Cow` panel would be unexercised (slop) in
  Phase 1. `panel.rs` lands with Phase 2, whose contiguous-panel reductions
  exercise the borrow path; until then QR's direct materialization is the honest
  minimal form.

### Phases

- **Phase 0 (done):** contiguous single-reflector sweeps vectorized via
  `axpy_slice` (Householder apply + Francis left-apply); eig 5.9×→4.4×, svd
  4.1×→3.4×. Establishes the SIMD-apply baseline and the backward-error test
  tolerance that admits blocked reorderings.
- **Phase 1 (done):** `reflector_block/{mod,accumulate}` — the compact-WY
  accumulation (`build_t`) and `tiled_gemm`-based `apply_block_left`. Verified in
  isolation: differential equality (within `O(r·ε‖C‖)`) of the block apply against
  `r` sequential single-reflector applies, plus column-norm (orthogonality)
  preservation. First consumer: **panel-blocked `qr_decompose`** (`dgeqrf`
  structure — the one-sided, no-look-ahead case, lower-risk than Hessenberg).
  Gated on `BLOCK_MIN_ROWS = 256` (measured A/B crossover ≈ 200: 256² QR 1.51 →
  1.29 ms blocked, but 128² regresses 175 → 223 µs), so below it the factorization
  is byte-for-byte the unblocked sweep (64² unchanged). Verified by a 256² solve
  recovering a known `x` to 1e-9. (Blocked Hessenberg is folded into Phase 2 with
  the two-sided reductions.)
- **Phase 2 — blocked U/V factor formation: implemented, verified, REVERTED as
  valueless (measured).** The SVD's `U = L₀…L_{n-1}` / `V = R₀…R_{n-2}` formation
  was rewritten to store the panel reflectors and apply them to the identity by
  blocked compact-WY `apply_block_right` (one-sided — no `dlabrd` look-ahead),
  verified correct by a 256² `A = U B Vᵀ` reconstruction + orthogonality +
  singular-value contract. **But A/B showed no benefit: 256² full SVD 164 ms
  blocked vs 163 ms unblocked.** The entire cost is the sequential Givens
  bidiagonal-QR sweep (Phase 3); factor formation is < 1 ms. Blocking it is
  cargo-cult and was removed (`apply_block_right` with it). *Blocking the two-sided
  reduction itself (`dlabrd`) is likewise low-leverage: the per-reflector applies
  are already SIMD (Phase 0), and the sweep dominates.* The genuine SVD lever is
  Phase 3.
- **Phase 3 — accelerate the Givens bidiagonal-QR sweep (`dbdsqr`) / small-bulge
  multishift QR for eig (`dlaqr5`):** the dominant, sequential iteration cost.
  Inherently serial (each rotation/bulge feeds the next), so the lever is
  restructuring to chase `nb`-wide bulge chains whose off-window updates batch into
  GEMMs — highest value, highest risk; gated on the hardened 8×8/16×16/defective
  batteries with the backward-error tolerance. This, not factor formation
  (Phase 2), is where the residual eig/SVD disparity actually lives.

## Rejected alternatives

- **Strided SIMD of the single reflector apply.** A width-3 right-apply has no
  contiguous span; gather/scatter SIMD over the strided row dimension does not beat
  the scalar loop (verified: `axpy_rows` with `cols = 3` is bandwidth-bound). Does
  not change the roofline ceiling.
- **Transpose `H` for the right-apply.** Per-step transposition is `O(n²)` copy
  against `O(n²)` apply work — no asymptotic gain and double the memory traffic.
- **Forcing the existing within-block apply narrowing without blocking.** Closes a
  sub-2× constant only, and (ADR #20, machine-checked) perturbs defective
  eigenvalues; superseded by blocking, which is both correct and BLAS-3.

## Consequences

- A reusable compact-WY seam (`reflector_block/`) that every two-sided/one-sided
  reduction and bulge-chasing iteration shares — SSOT for blocked orthogonal
  updates, the same role `tiled_gemm` plays for products.
- Phases 1–3 are large and index-intensive (the historical #1 bug source in QR
  iterations); they are sequenced and individually differential-verified rather
  than landed as one drop, honoring correctness-over-breadth. Each phase is a
  separate WIP-limited item with the unblocked path as its oracle.
- Expected outcome: the trailing updates become compute-bound GEMMs, closing the
  bulk of the residual eig/SVD disparity; the Givens bidiagonal-QR sweep (Phase 3,
  SVD) is the last latency-bound holdout and may retain a constant factor.
