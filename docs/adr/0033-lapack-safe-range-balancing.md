<a id="adr-0033"></a>

# 0033. Scale-safe kernels with a minimal-move matrix gate for dense factorizations

Status: Accepted

Revision 2026-09-25: Francis follows LAPACK `dlahqr` (shifts,
exceptional shifts, first column, Ahues–Tisseur deflation) and `dlanv2`
(2×2 standardization and eigenvalues); the Golub–Kahan iteration follows
`dbdsqr`'s relative-accuracy deflation; an empty gate range is a typed
error. Evidence: the fifth review's `schur` wrong `Ok` at the gate edge
(`1.0163` for `1.002`, a non-orthogonal 2×2 rotation whose norm
underflowed) and its extended scan (Bf16 and F16 non-convergence on
clustered and rank-deficient input).

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
| `francis::stack_reflector` | `vᵀv ≤ 12m²` | window `(2, 2, 4)` | `dlarfg` |
| `francis` shift pair and first column | ratios of entries | by construction | `dlahqr` (shifts from the `s`-scaled 2×2; `V` divided by `S`, then by `‖V‖₁`) |
| `standard_block` (2×2 standardization and eigenvalues) | ratios and square roots | by construction | `dlanv2` |

A window `(dₗ, dᵤ, f)` forms the product unscaled while the local magnitude
`m` lies in `[safmin^(1/dₗ), (Ω·2⁻ᶠ)^(1/dᵤ)]` and otherwise divides the
operands by the power of two bringing `m` into `[1, 2)`. The shift window's
lower degree is 4: an underflowed `t₁₂²` degrades the Wilkinson shift to the
Rayleigh shift, which stagnates on a nearly equal pair. `dlanv2`'s
real-versus-complex decision compares `z/scale` rather than `dlanv2`'s `z`,
which carries the block's units. The standardization applies its rotation
outside the block (`dlahqr`'s `DROT` calls) and writes the block from
`dlanv2`, so rounding residue left and below the block never enters its
subdiagonal. The previous quadratic `tr ± √(tr² − 4·det)` cancelled to
`O(√ε)` on double eigenvalues, and the previous standardization formed its
rotation's norm `√(ex² + ey²)` from unscaled products that underflowed at
the gate's lower end.

**Deflation.**

- Golub–Kahan (`svd/bidiagonal_qr.rs`): `dbdsqr`'s relative-accuracy tests —
  `tol = tolmul·ε`, `tolmul = max(10, min(100, ε^(−1/8)))`; split where
  `|eᵢ| ≤ thresh = max(tol·σ̃_min, floor)` (`σ̃_min` its `SMINOA` estimate);
  the bottom test `|e_{q−1}| ≤ tol·|d_q|` and the forward recurrence
  `|eᵢ| ≤ tol·μ`. The floor is `2^⌈log₂ k⌉·safmin`, not `dbdsqr`'s
  `maxitr·n²·unfl`, which assumes `n²·unfl ≪ ulp` and in F16 at `k = 24`
  (`≈ 0.2`) would split superdiagonals of unit-scale matrices.
- Francis (`schur/francis.rs`): `dlahqr`'s test — `|h_{k,k−1}| ≤ floor`, or
  the pre-check `|h_{k,k−1}| ≤ ulp·tst` and the Ahues–Tisseur test
  (LAWN 122); `dlahqr`'s neighbour fallback for `tst = 0` is not taken (no
  test input distinguishes it). The floor is
  `2^⌈log₂ n⌉·safmin` in place of `dlahqr`'s `safmin·n/ulp`, which assumes
  `ulp² ≫ safmin` (false in F16: `ε² = 2⁻²⁰ < safmin = 2⁻¹⁴`).

**Matrix tier.** What the kernels cannot rescale locally is gated on
`‖A‖_max ∈ [max(smlnum^(1/d), 2^g·smlnum), (Ω·2⁻ᶠ)^(1/d)]`
(`thresholds::homogeneous_safe_range`; `smlnum = safmin/ε`, `Ω` the
overflow threshold, `2^g·safmin` the routine's deflation floor). The bound
factor enters by exponent arithmetic only and never divides `smlnum`. When
the floor end would pass the upper end, the lower end stops at the upper end
(overflow is the hard constraint). When even `smlnum^(1/d)` exceeds the
upper end — the bound factor exceeds `Ω/smlnum`, about `2²⁰` in F16 — the
routine returns `LetoError::Overflow` naming the limit, never a false value.
Inside the range the input is factored unscaled; outside it, by the minimal
power of two back inside, results multiplied back by `scaling::restore`.

| Routine | Remaining intermediate | `d` | `f` | `g` |
|---|---|---|---|---|
| Jacobi | `2a_pq`, `a_qq − a_pp`, diagonal partial sums `≤ 2‖A‖₂` | 1 | `1 + r` | 0 |
| Symmetric QL | chase correction `e_{l+1}·eₗ ≤ ‖A‖₂²` (`ql.rs`) | 2 | `2r` | 0 |
| Column-pivoted QR | `tail_norm_sq ≤ rows·‖A‖_max²` (`decompose.rs`) | 2 | `⌈log₂ rows⌉` | 0 |
| Francis (`schur`, `eigenvalues`) | `vᵀH`, unscaled `‖v‖₂ < √Ω` times `‖H‖_F` | 2 | `2r` | `⌈log₂ n⌉` |
| Golub–Kahan (SVD family) | reflector dots `‖v‖₂·‖A‖_F`, `‖v‖₂ ≤ 4√M` | 1 | `2 + ⌈⌈log₂ M⌉/2⌉ + r` | `⌈log₂ k⌉` |

`2^r ≥ ‖A‖_F/‖A‖_max` (`scaling::norm_ratio_log2`). Scaling up is exact;
scaling down can underflow an entry far below the largest, which the upper
ends — the overflow threshold itself — confine to inputs within `2^f` of
overflowing.

**Tests assert derived bounds.** `tests/ops/backward_error.rs` composes
Higham's `γ_k` bounds (standard model, Lemmas 3.1 and 3.3, §3.1 inner
products, Lemmas 19.7–19.8 for rotations) over the enumerated reflectors,
rotations, shifts and deflations of each routine at its iteration cap; the
per-reflector constant `γ_{8m+29}`, the rank-2 update's `γ_{17m+42}` and the
QL sweep's entry counts are derived in its documentation. Weyl (singular
values, symmetric eigenvalues) and Bauer–Fike (general eigenvalues, `κ` from
`f64` left and right eigenvectors) turn them into value bounds. At the caps
they are loose (about `10⁻¹⁰` relative for `f64`, vacuous for the 8- and
11-bit formats); measured errors are reported, not asserted.

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

## Consequences

What the tests establish, for `f64`, `f32`, `F16` and `Bf16`:

- `tests/ops/scale_range.rs`: `SIMILAR` and `GENERAL` at every binary
  exponent from the smallest subnormal to three below the largest converge
  in `schur`, `eigenvalues`, both SVD entry points and pivoted QR within the
  derived bounds; no exponent is exempt.
- `tests/ops/graded_scan.rs`: seeded graded, clustered and rank-deficient
  matrices of order 2 to 8 at every exponent from the smallest normal to
  three below the largest converge in all four routines within the derived
  Weyl/Bauer–Fike bounds, and `singular_values` agrees with `svd_decompose`;
  F16 order 48 and Bf16 order 64 factor; subnormal blocks deflate at the
  floors.
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
five independent reviews on PRs #233 and #237, `scaling.rs`/`thresholds.rs`
module documentation, `tests/ops/backward_error.rs`, `tests/ops/scale_range.rs`,
`tests/ops/graded_scan.rs`, `tests/ops/schur.rs` and `tests/ops/eigen.rs`.
