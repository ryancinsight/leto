<a id="adr-0033"></a>

# 0033. Scale-safe kernels with a minimal-move matrix gate for dense factorizations

Status: Accepted

Revision 2026-09-25: Francis follows `dlahqr` and `dlanv2`, Golub–Kahan
`dbdsqr` with `dlasv2` 2×2 blocks; the SVD gate takes `dgesvd`'s degree-2
range; a deflation floor that cannot fit is a typed error, not a clamp;
tests assert a-priori bounds only where informative and certify every
format a posteriori. Evidence: the reviews' `schur` wrong `Ok` at the gate
edge, skew-symmetric non-convergence (`f64`, 18 of 320), surviving
`6k²·safmin` mutant and vacuous `F16`/`Bf16` bounds; the skew-symmetric
scan family's stalls, checked against LAPACK `dlahqr`.

## Context

The dense factorizations (`symmetric_eigen_qr`, `symmetric_eigen_jacobi`,
`schur`, `eigenvalues`, `singular_values`/`svd_decompose`/`pinv`,
`col_piv_qr`) form products of entries that overflow or underflow long
before the input leaves the range of the scalar: f64 SVD failed to converge
beyond `2^±256`, `RealSchur::eigenvalues` returned `{3, 3}` for `{2, 4}` at
`2⁻⁸⁶` in f32, f64 `col_piv_qr` returned rank 0 beyond `2^±512`, and F16/Bf16
Francis stalled on ordinary nonsymmetric input.

Balancing the whole matrix alone does not fix it: a local product such as a
Givens norm depends on the local magnitudes during the iteration, not on
`‖A‖_max`, and a fixed landing scales entries far below the largest into
underflow (`diag(1e300, 1e-300)`).

## Decision

**Kernel tier (`linalg::scaling`).** Local products are formed scale-safely.

| Kernel | Product | Form | LAPACK reference |
|---|---|---|---|
| `bidiagonal_qr::givens` | `a² + b² ≤ 2m²` | window `(2, 2, 1)` | `dlartg` (`rtmin`, `rtmax`) |
| `bidiagonal_qr::qr_step` shift + first column | `δ² + t₁₂² ≤ 2m⁴` | window `(4, 4, 1)` | `dbdsqr`, `dlas2` |
| `bidiagonal::colmajor::larfg` | `α² + ‖x‖² ≤ len·m²`, `len ≤ rows` | window `(2, 2, ⌈log₂ rows⌉)` | `dlarfg` via `dnrm2` |
| `francis::stack_reflector` | `‖x‖² ≤ 3m²`; `v` normalized, `\|vᵢ\| ≤ 1`, `τ ≤ 2` | window `(2, 2, 2)` | `dlarfg` |
| `francis` shift pair and first column | ratios of entries | by construction | `dlahqr` (shifts from the `s`-scaled 2×2; `V` divided by `S`, then by `‖V‖₁`) |
| `standard_block` (2×2 standardization and eigenvalues) | ratios and square roots | by construction | `dlanv2` |
| `svd::triangular_pair` (2×2 bidiagonal blocks) | ratios, sums of squares by `dlapy2` | by construction | `dlasv2` (`√(t² + m²)` needs `1/u²` representable, false in F16) |
| `scaling::norm_ratio_log2` | `Σ(aᵢⱼ/‖A‖_max)²`, pairwise | a sum strictly within `c = O(log₂ len·u)` below a power of two is charged the next | recursive sums stagnate in F16 (`2048 + 1 = 2048`) |

A window `(dₗ, dᵤ, f)` forms the product unscaled while the local magnitude
`m` lies in `[safmin^(1/dₗ), (Ω·2⁻ᶠ)^(1/dᵤ)]` and otherwise divides the
operands by the power of two bringing `m` into `[1, 2)`. The shift window's
lower degree is 4: an underflowed `t₁₂²` degrades the Wilkinson shift to the
Rayleigh shift, which stagnates on a nearly equal pair. `dlanv2`'s
real-versus-complex decision compares `z/scale` rather than `dlanv2`'s `z`,
which carries the block's units. The standardization applies its rotation
outside the block (`dlahqr`'s `DROT` calls) and writes the block from
`dlanv2`. The previous quadratic cancelled to `O(√ε)` on double
eigenvalues; the previous standardization's rotation norm underflowed at
the gate's lower end.

**Deflation.**

- Golub–Kahan (`svd/bidiagonal_qr.rs`): `dbdsqr`'s relative-accuracy tests —
  `tol = tolmul·ε`, `tolmul = max(10, min(100, ε^(−1/8)))`; split where
  `|eᵢ| ≤ thresh = max(tol·σ̃_min, floor)` (`σ̃_min` its `SMINOA` estimate);
  the bottom test `|e_{q−1}| ≤ tol·|d_q|` and the forward recurrence
  `|eᵢ| ≤ tol·μ`; a 2×2 block is diagonalized by `dlasv2`, since shifted
  steps cycle on it when its smaller singular value is subnormal. The floor
  is `2^⌈log₂ k⌉·safmin`. A deflated entry below it costs at most `ε‖A‖`
  backward error only if `floor ≤ ε·‖A‖_max` for every admitted input, so
  the gate raises its lower end to `floor/ε`. `dbdsqr`'s `6k²·safmin` is
  `0.21 = 216·ε` in F16 at `k = 24`: it needs `‖A‖_max ≥ 216`, past the
  upper end `(Ω·2⁻⁵)^½ ≈ 45` of every order-24 matrix
  (`deflation_floor_keeps_unit_scale_superdiagonals_in_f16`).
- Francis (`schur/francis.rs`): `dlahqr`'s test — `|h_{k,k−1}| ≤ floor`, or
  the pre-check `|h_{k,k−1}| ≤ ulp·tst` (with `dlahqr`'s neighbour fallback
  when `tst = 0`, `zero_diagonal_pair_deflates_against_its_neighbours`) and
  the Ahues–Tisseur test (LAWN 122). The floor is the larger of
  `2^⌈log₂ n⌉·safmin` (gate-raised, as for the SVD) and `dlahqr`'s
  `SMLNUM = safmin·n/ulp` taken in units of the matrix,
  `2^e·min(2^⌈log₂ n⌉·safmin/ε, ε)` with `2^e ≤ ‖H‖_max/2^⌈log₂ n⌉ ≤ ‖A‖_max`.
  An absolute `SMLNUM` froze skew-symmetric iterates with entries near
  `10⁵⁸`: a zero diagonal leaves only `SMLNUM` in the Ahues–Tisseur test,
  and the bulge's `h₁₀/S` underflows before the subdiagonal reaches it. The
  cap at `ε` keeps F16 (`safmin/ε = 2⁻⁴`) within the backward error.
- Francis step: `dlahqr`'s loop 50 starts the bulge at the lowest row with
  two consecutive small subdiagonals (writing `h_{m,m−1}·(1 − τ)`), and the
  stack reflector is normalized as `dlarfg`'s, so its applications are
  degree-1 sums; with an unnormalized `v` (`vᵀh` degree 2) `f64`
  skew-symmetric tridiagonals at the gate's lower end stalled. LAPACK's own
  `dlahqr` (OpenBLAS build) converges on every tridiagonal that failed.

**Matrix tier.** What the kernels cannot rescale locally is gated on
`‖A‖_max ∈ [max(smlnum^(1/d), 2^g·smlnum), (Ω·2⁻ᶠ)^(1/d)]`
(`thresholds::homogeneous_safe_range`; `smlnum = safmin/ε`, `Ω` the
overflow threshold, `2^g·safmin` the routine's deflation floor). The bound
factor enters by exponent arithmetic only and never divides `smlnum`. When
even `smlnum^(1/d)` exceeds the upper end — the bound factor exceeds
`Ω/smlnum`, about `2²⁰` in F16 — or the floor end does, the routine
returns `LetoError::Overflow` naming which (`EmptyRange`), never a false
value or a floor outside the backward error.
Inside the range the input is factored unscaled; outside it, by the minimal
power of two back inside, results multiplied back by `scaling::restore`.

| Routine | Remaining intermediate | `d` | `f` | `g` |
|---|---|---|---|---|
| Jacobi | `2a_pq`, `a_qq − a_pp`, diagonal partial sums `≤ 2‖A‖₂` | 1 | `1 + r` | 0 |
| Symmetric QL | chase correction `e_{l+1}·eₗ ≤ ‖A‖₂²` (`ql.rs`) | 2 | `2r` | 0 |
| Column-pivoted QR | `tail_norm_sq ≤ rows·‖A‖_max²` (`decompose.rs`) | 2 | `⌈log₂ rows⌉` | 0 |
| Francis (`schur`, `eigenvalues`) | Hessenberg dots `‖v‖₂·‖A‖_F`, `‖v‖₂ ≤ 4√n`; `dgees` range | 2 | `2r` | `⌈log₂ n⌉` |
| Golub–Kahan (SVD family) | reflector dots `‖v‖₂·‖A‖_F`, `‖v‖₂ ≤ 4√M`; `dgesvd` range | 2 | `2 + ⌈⌈log₂ M⌉/2⌉ + r` | `⌈log₂ k⌉` |

Francis and Golub–Kahan need only degree 1 for their bounds; they take
LAPACK's driver form, `[√safmin/ε, ε/√safmin]`, whose lower end leaves the
small singular values and converging subdiagonals headroom above
`safmin` (at the degree-1 lower end the SVD cycled in `Bf16`).

`2^r ≥ ‖A‖_F/‖A‖_max` (`scaling::norm_ratio_log2`; a computed sum exactly
on a power of two is taken as exact). Scaling up is exact;
scaling down can underflow an entry far below the largest, which the upper
ends — the overflow threshold itself — confine to inputs within `2^f` of
overflowing.

**Tests assert derived bounds.** `tests/ops/backward_error.rs` composes
Higham's `γ_k` bounds (standard model, Lemmas 3.1 and 3.3, §3.1 inner
products, Lemmas 19.7–19.8 for rotations) over the enumerated reflectors,
rotations, shifts and deflations of each routine at its iteration cap; the
per-reflector constant `γ_{8m+29}`, the rank-2 update's `γ_{17m+42}` and the
QL sweep's entry counts are derived in its documentation. They are
asserted only where below `1` (`backward_error::informative`): `f64`; `f32`
for the SVD, the Schur residual to order 4 and the Francis eigenvalues to
order 7; at the caps none is in `F16`/`Bf16`, where
one length-3 reflector costs `γ₅₃ ≈ 0.026`/`0.21`. Every format is certified
a posteriori (`tests/ops/a_posteriori.rs`): the measured residual and
orthogonality of the returned factors give, through the polar factor, a
perturbation `E` of the input whose spectrum the returned values are; Weyl
and Bauer–Fike (`κ` from `f64` eigenvectors) bound the values. The
certificates bind the values to the factors; the factors' own accuracy is
asserted only where the a-priori residual bound is informative.

## Rejected alternatives

- Recentring inputs to `[1, 4)`: worked around unsafe kernel products.
- A power-of-two window around the old first column and the old 2×2
  standardization (the previous revision). Rejected: the window tested the
  block's largest entry, not the operands of the rotation-norm product,
  which underflowed while the block was in window; `dlahqr`/`dlanv2` are
  scale-safe by construction.
- `dbdsqr`'s and `dlahqr`'s floor constants verbatim. Rejected: both assume
  `ulp² ≫ safmin` or `n²·unfl ≪ ulp`, false in F16.
- An empirical `n²·ε` tolerance envelope in the tests. Rejected by review:
  tolerances must be derived.
- Clamping the raised lower end to the upper end when the deflation floor
  cannot fit (the previous revision). Rejected: each floor deflation then
  costs more than `ε·‖A‖_max`, outside the backward error the tests count.
- A residual threshold `c·n·ε(T)` for `F16`/`Bf16`. Rejected: no `c`
  follows from the per-operation bounds (they exceed `1` at these
  precisions); one fitted to the measured residuals would be tuned.

## Consequences

What the tests establish, for `f64`, `f32`, `F16` and `Bf16`:

- `tests/ops/scale_range.rs`: `SIMILAR` and `GENERAL` at every binary
  exponent from the smallest subnormal to three below the largest converge
  in `schur`, `eigenvalues`, both SVD entry points and pivoted QR within the
  certificates and, where informative, the derived bounds; no exponent is
  exempt.
- `tests/ops/graded_scan.rs`: seeded graded, clustered, rank-deficient and
  skew-symmetric (tridiagonal and dense) matrices of order 2 to 8 at every
  exponent from the smallest normal to three below the largest converge in
  all four routines, within the a-posteriori certificates in every format
  and the derived a-priori bounds where informative; F16 order 48 and Bf16
  order 64 factor; subnormal blocks deflate at the floors; F16 orders past
  the deflation floor are `Overflow` (`tests/ops/scale_range.rs`).
- `tests/ops/schur.rs`: the clustered family at the Francis gate's edge
  (`n = 8`, `2⁻⁶⁰⁰ … 2⁵⁰⁰`) is within the derived Weyl bound.
- `tests/ops/eigen.rs`: Jacobi returns 295 diagonals per format exactly, and
  a near-overflow `M·(J₄ − 2I)` bit-for-bit `4×` its unscaled quarter.

In-range results are **not** bit-identical to the pre-change tree: the
`dlahqr` shifts, the Ahues–Tisseur deflation, `dlanv2`, `dbdsqr`'s
deflation and the kernel windows (at both edges — the SVD shift window
starts rescaling at `(Ω/2)^¼ ≈ 2²⁵⁶` in `f64`) change the rounding path.
`pinv` reports `LetoError::Overflow` when a retained singular value's
reciprocal is not finite.

## Driving evidence

`backlog.md#LETO-DENSE-SCALE-RANGE-2026-09-24` (delivered by PR #237), the
independent reviews on PRs #233 and #237, `scaling.rs`/`thresholds.rs`
module documentation, `tests/ops/backward_error.rs`, `tests/ops/scale_range.rs`,
`tests/ops/graded_scan.rs`, `tests/ops/schur.rs` and `tests/ops/eigen.rs`.
