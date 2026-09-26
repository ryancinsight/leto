<a id="adr-0033"></a>

# 0033. Scale-safe kernels with a minimal-move matrix gate for dense factorizations

Status: Accepted

Revision 2026-09-25: Francis follows `dlahqr`/`dlanv2` in full, Golub–Kahan
`dbdsqr` with `dlasv2` 2×2 blocks and its zero-shift sweep; both gates
take LAPACK's driver range;
an unfittable floor is `Overflow`, not a clamp; tests certify every format
a posteriori. Evidence: PR #237 reviews (gate-edge wrong `Ok`, skew-symmetric
non-convergence, the `6k²·safmin` mutant, vacuous F16/Bf16 bounds).

## Context

The dense factorizations formed products of entries that overflow or
underflow long before the input leaves the scalar's range (f64 SVD failed
beyond `2^±256`; f32 `RealSchur::eigenvalues` returned `{3, 3}` for `{2, 4}`
at `2⁻⁸⁶`; f64 `col_piv_qr` returned rank 0 beyond `2^±512`; F16/Bf16
Francis stalled). Whole-matrix balancing alone does not fix it: local
products follow local magnitudes, and a fixed landing underflows small
entries.

## Decision

**Kernel tier.** Local products are formed scale-safely. A window
`(dₗ, dᵤ, f)` forms a product unscaled while the local magnitude `m` lies in
`[safmin^(1/dₗ), (Ω·2⁻ᶠ)^(1/dᵤ)]`, else first divides the operands by the
power of two bringing `m` into `[1, 2)`.

| Kernel | Product | Form | LAPACK |
|---|---|---|---|
| `bidiagonal_qr::givens` | `a² + b² ≤ 2m²` | window `(2, 2, 1)` | `dlartg` |
| `bidiagonal_qr::qr_step` shift | `δ² + t₁₂² ≤ 2m⁴` | window `(4, 4, 1)`¹ | `dbdsqr`, `dlas2` |
| `bidiagonal::colmajor::larfg` | `α² + ‖x‖² ≤ rows·m²` | window `(2, 2, ⌈log₂ rows⌉)` | `dlarfg` |
| `francis::stack_reflector` | `‖x‖² ≤ 3m²`; `v₀ = 1`, `\|vᵢ\| ≤ 1`, `τ ≤ 2` | window `(2, 2, 2)` | `dlarfg` |
| Francis shifts, first column | ratios of entries | by construction | `dlahqr` |
| `standard_block` | ratios, square roots | by construction² | `dlanv2` |
| `svd::triangular_pair` | ratios, sums of squares by `dlapy2`³ | by construction | `dlasv2` |
| `scaling::norm_ratio_log2` | `Σ(aᵢⱼ/‖A‖_max)²` pairwise⁴ | exact-exponent bound | — |

¹ Lower degree 4: an underflowed `t₁₂²` degrades the Wilkinson shift to the
Rayleigh shift, which stagnates on a near-equal pair. ² Compares `z/scale`,
not `dlanv2`'s dimensional `z`; rotations applied outside the block as
`dlahqr`'s `DROT`. ³ `dlasv2`'s `√(t² + m²)` needs `1/u²` representable,
false in F16. ⁴ With a running binary exponent: recursive sums stagnate in
F16 (`2048 + 1 = 2048`) and `256²` unit entries overflow it; the upper
bound charges a sum inside its rounding band below a power of two the next
one, the lower bound `2^l` divides by `1 − 2c` first.

**Deflation.** The absolute floor only has to stop a sweep on entries that
can no longer be resolved, so it is `safmin`. At most `k` distinct entries
are zeroed at or below it, jointly `≤ √k·safmin` in Frobenius norm, which
the backward-error premise holds to `ε·‖A‖_F`; with `2^l ≤ ‖A‖_F/‖A‖_max`
each gate raises its lower end to `2^(⌈⌈log₂ k⌉/2⌉ − l)·smlnum`
(`thresholds::deflation_count_log2`).

- Golub–Kahan: `dbdsqr`'s relative tests (`tol = tolmul·ε`,
  `tolmul = max(10, min(100, ε^(−1/8)))`, split at
  `|eᵢ| ≤ max(tol·σ̃_min, floor)`, bottom and forward-recurrence tests);
  2×2 blocks by `dlasv2`, since shifted steps cycle on one whose smaller
  singular value is subnormal; `dbdsqr`'s zero-shift sweep (loop 120) when a
  shift would ruin relative accuracy or is negligible, since the shifted
  step flips signs with period 2 on tiny F16 singular values.
- Francis: `dlahqr`'s test — `|h_{k,k−1}| ≤ floor`, or `≤ ulp·tst` (with the
  neighbour fallback when `tst = 0`) and Ahues–Tisseur (LAWN 122). Floor:
  the larger of `safmin` and `dlahqr`'s `SMLNUM = safmin·n/ulp`
  in units of the matrix, `2^e·min(2^⌈log₂ n⌉·safmin/ε, ε)`,
  `2^e ≤ ‖H‖_max/2^⌈log₂ n⌉ ≤ ‖A‖_max`. On a zero diagonal (skew-symmetric
  iterates) Ahues–Tisseur leaves only `SMLNUM`, and with an absolute one
  the bulge's `h₁₀/S` underflowed before the subdiagonal reached it
  (entries near `10⁵⁸`); the `ε` cap keeps F16 (`safmin/ε = 2⁻⁴`) inside the
  backward error.
- Francis step: `dlahqr`'s loop 50 starts the bulge at two consecutive
  small subdiagonals (writing `h_{m,m−1}·(1 − τ)`); the WANTT form sets the
  apply ranges; the normalized reflector keeps its applications degree 1.

**Matrix tier.** What the kernels cannot rescale locally is gated on
`‖A‖_max ∈ [max(smlnum^(1/d), 2^g·smlnum), (Ω·2⁻ᶠ)^(1/d)]`
(`thresholds::homogeneous_safe_range`, `smlnum = safmin/ε`, `2^g·safmin` the
floor). The bound factor enters by exponent arithmetic only and never
divides `smlnum`. An empty range — `2^f > Ω/smlnum`, or the floor end past
the upper end — is `LetoError::Overflow` naming which (`EmptyRange`).
Inside the range the input is factored unscaled; outside, by the minimal
power of two back inside, results restored by `scaling::restore`.
`2^r ≥ ‖A‖_F/‖A‖_max` from `norm_ratio_log2`.

| Routine | Derived intermediate bound | `d` | `f` | `g` |
|---|---|---|---|---|
| Jacobi | `2a_pq`, `a_qq − a_pp`, diagonal partial sums `≤ 2‖A‖₂` | 1 | `1 + r` | 0 |
| Symmetric QL | chase correction `e_{l+1}·eₗ ≤ ‖A‖₂²` | 2 | `2r` | 0 |
| Column-pivoted QR | `tail_norm_sq ≤ rows·‖A‖_max²` | 2 | `⌈log₂ rows⌉` | 0 |
| Francis | Hessenberg dots `‖v‖₂·‖A‖_F`, `‖v‖₂ ≤ 4√n`; driver range | 2 | `2r` | `⌈⌈log₂ n⌉/2⌉ − l` |
| Golub–Kahan | reflector dots `‖v‖₂·‖A‖_F`, `‖v‖₂ ≤ 4√M`; driver range | 2 | `2 + ⌈⌈log₂ M⌉/2⌉ + r` | `⌈⌈log₂ k⌉/2⌉ − l` |

Both need only degree 1 for their bounds; they take the `dgees`/`dgesvd`
form `[√safmin/ε, ε/√safmin]`, whose lower end leaves converging entries
headroom above `safmin`: at the degree-1 lower end Bf16 skew-symmetric
tridiagonals near `2⁻¹¹⁵` fail even with the zero-shift sweep
(`svd::bf16_skew_tridiagonals_near_safmin_converge`). In F16 the floor end
passes the upper end only past order `2¹⁴` (SVD) and `2²²` (Francis), by
the exponent count; `large_orders.rs` factors structured families to 640.

**Tests.** `tests/ops/backward_error.rs` composes Higham's `γ_k` (standard
model, Lemmas 3.1, 3.3, 19.7–19.8, §3.1) over each routine's enumerated
operations at its iteration cap (`γ_{8m+29}` per reflector, `γ_{17m+42}` per
rank-2 update, derived there). These are asserted only where below `1`:
f64; f32 except the Schur residual from order 5 and Francis at order 8;
never the iterative routines in F16/Bf16, where one length-3 reflector
costs `γ₅₃ ≈ 0.027`/`0.26`. Every format is certified a posteriori
(`tests/ops/a_posteriori.rs`): the measured residual and orthogonality of
the returned factors give, through the polar factor, a perturbation `E`
whose spectrum the returned values are, bounded by Weyl or Bauer–Fike (`κ`
from f64 eigenvectors). In F16/Bf16 this binds values to factors; no
derived bound certifies the factors themselves.

## Rejected alternatives

- Unconditional scaling to a fixed landing: loses entries far below the
  largest (`diag(1e300, 1e-300)`) when no scaling was needed.
- Recentring inputs to `[1, 4)`: it worked around unsafe kernel products
  and turned 82 finite F16 base `Ok` results into `Err` (review of 4959200).
- One generic LAPACK range for every routine: each gate's degree and
  factor follow from its own intermediates; `[√safmin/ε, ε/√safmin]` would
  move Jacobi's `diag(1e300, 1e-300)` and underflow the small entry, and in
  F16 `√safmin/ε = 8 > ε/√safmin = ⅛`: `dgeev`'s range is inverted, so every
  F16 input would be scaled.
- `dbdsqr`'s `MAXITR·(N·(N·UNFL))` — its iteration cap times `unfl` — and
  `dlahqr`'s absolute `safmin·n/ulp`: they presume unit scale, `n²·unfl ≪
  ulp` and `ulp² ≫ safmin`, false in F16 (`6k²·safmin = 0.21 = 216ε` at
  `k = 24`). The joint-norm derivation needs only `safmin`; `2^⌈log₂ k⌉·safmin`
  (the previous revision) was chosen, not derived, and refused F16 SVDs
  from order 384 that factor.
- Clamping the raised lower end to the upper end (floor deflations then
  exceed `ε·‖A‖_F`), or a condition on `ε·‖A‖_max` (refused F16 all-ones
  at orders 64–128).
- A power-of-two window around the old first column and 2×2
  standardization: it tested the block's largest entry, not the
  rotation-norm operands, which underflowed in window.
- An empirical `n²·ε` envelope, or a residual constant `c·n·ε` fitted to
  measured F16/Bf16 residuals: tolerances must be derived.

## Consequences

Tests, all four formats: `scale_range.rs` (two matrices at every exponent,
subnormal to near-overflow), `graded_scan.rs` (graded, clustered,
rank-deficient, skew-symmetric, order 2–8, every normal exponent; subnormal
blocks), `large_orders.rs` (F16 structured families to order 640) — every
routine converges within the certificates and the informative bounds;
`schur.rs` pins the gate edge and the deflation decisions; `eigen.rs`
Jacobi exact on 295 diagonals per format. In-range results are not
bit-identical to the pre-change tree; `pinv` reports `Overflow` for a
non-finite reciprocal. Cost: `dbdsqr`'s split tolerance `tolmul·ε ≈ 90ε`
raises the f64 `svd_decompose` residual on ordinary inputs to up to
`90ε‖A‖_F` (review measurement), against `≤ 10ε‖A‖_F` before (since 9054d80).

## Driving evidence

`backlog.md#LETO-DENSE-SCALE-RANGE-2026-09-24` (PR #237); the reviews on
PRs #233 and #237; LAPACK `dlahqr` (OpenBLAS) converging on the
skew-symmetric inputs that failed; `tests/ops/{backward_error, a_posteriori,
scale_range, graded_scan, large_orders, schur, eigen}.rs`.
