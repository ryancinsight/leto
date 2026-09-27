<a id="adr-0033"></a>

# ADR 0033: Scale-safe kernels with a minimal-move matrix gate for dense factorizations

Status: Accepted

Revision 2026-09-25: Francis follows `dlahqr`/`dlanv2`, Golub–Kahan
`dbdsqr` with `dlasv2` and its zero-shift sweep; both gates take LAPACK's
driver range; an unfittable floor is `Overflow`; tests certify every format
a posteriori (PR #237 reviews).
Revision 2026-09-27: a stalled Francis block deflates by the
neighbourhood test (Bf16 skew ties, `LETO-BF16-SKEW-FRANCIS-STALL-2026-09-25`).

## Context

Dense factorizations formed products that over- or underflow long before
the input leaves the range (f64 SVD beyond `2^±256`; f32 eigenvalues
`{3, 3}` for `{2, 4}` at `2⁻⁸⁶`; F16/Bf16 Francis stalled). Whole-matrix
balancing alone does not fix it: local products follow local magnitudes.

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
  backward error. A block not deflated through both exceptional shifts
  (`2·KEXSH` iterations, `dlahqr.f` ll. 369–387) switches to a leto-own
  test, `|h_{k,k−1}| ≤ ulp·(|h_{k−1,k−1}| + |h_{k,k}| + |h_{k−1,k−2}| +
  |h_{k+1,k}|)`: EISPACK `hqr`'s normwise form (`hqr.f` ll. 88–92) with the
  adjacent subdiagonals in the scale, charged `4ε‖H‖₂`; the cap is
  `dlahqr`'s `ITMAX = 30·max(10, n)` (l. 293). Bf16 skew iterates tie two clusters
  `±i(a ± δ/2)` over a residue diagonal, where Ahues–Tisseur keeps `δ` and
  rounding undoes every exceptional shift; ties past one ulp stay a typed
  `ConvergenceError` (gap_audit).
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

**Tests.** `backward_error.rs` composes Higham's `γ_k` (Lemmas 3.1, 3.3,
19.7–19.8) over each routine's operations at its iteration cap, asserted
where below `1` (f64; most f32; never the iterative routines in F16/Bf16,
`γ₅₃ ≈ 0.027`/`0.26`). `a_posteriori.rs` certifies every format: the
returned factors' residual and orthogonality give a perturbation `E` whose
spectrum the values are (Weyl, Bauer–Fike); in F16/Bf16 this binds values
to factors, not the factors themselves.

## Rejected alternatives

- Unconditional scaling to a fixed landing: loses `diag(1e300, 1e-300)`.
- Recentring inputs to `[1, 4)`: it worked around unsafe kernel products
  and turned 82 finite F16 base `Ok` results into `Err` (review of 4959200).
- One generic LAPACK range `[√safmin/ε, ε/√safmin]` for every routine: it
  underflows Jacobi's `diag(1e300, 1e-300)`, and is inverted in F16
  (`8 > ⅛`), scaling every input.
- `dbdsqr`'s `MAXITR·(N·(N·UNFL))` and `dlahqr`'s absolute `safmin·n/ulp`:
  they presume `n²·unfl ≪ ulp`, `ulp² ≫ safmin`, false in F16
  (`6k²·safmin = 216ε` at `k = 24`); `2^⌈log₂ k⌉·safmin` was chosen, not
  derived, and refused F16 SVDs from order 384 that factor.
- Clamping the raised lower end to the upper end (floor deflations exceed
  `ε·‖A‖_F`), or gating on `ε·‖A‖_max` (refused F16 all-ones, 64–128).
- A power-of-two window on the block's largest entry for the first column
  and 2×2 standardization: the rotation-norm operands underflowed in it.
- Fitted `n²·ε` or `c·n·ε` residual envelopes: tolerances must be derived.
- For a stalled block, a larger multiple of `ulp` or the whole-matrix
  norm (fitted to the ties; charges up to `n·‖H‖_F`), or waiting out
  `ITMAX` first (Bf16 Schur vectors lost orthogonality at order 67).

## Consequences

Every routine converges within the certificates and informative bounds in
all four formats (`scale_range.rs`, `graded_scan.rs`, `large_orders.rs`,
`schur.rs`). Cost: `dbdsqr`'s `tolmul·ε ≈ 90ε`
raises the f64 `svd_decompose` residual to up to `90ε‖A‖_F`, from
`≤ 10ε‖A‖_F` (since 9054d80).

## Driving evidence

PR #237 (`LETO-DENSE-SCALE-RANGE-2026-09-24`) and the reviews on PRs #233
and #237; LAPACK `dlahqr` (OpenBLAS) converging on the skew-symmetric
inputs that failed; the Bf16 skew-tie trace and `tests/ops/schur.rs`.
