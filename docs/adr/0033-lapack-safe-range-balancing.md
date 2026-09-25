<a id="adr-0033"></a>

# 0033. Scale-safe kernels with a minimal-move matrix gate for dense factorizations

Status: Accepted

Revision 2026-09-25: Francis and Golub–Kahan kernels now form their products
scale-safely (LAPACK `dlartg`/`dlarfg`/`dlahqr`/`dlanv2` pattern), so the
`[1, 4)` recentring exception and every recorded F16/Bf16 non-convergence
are gone; gate upper ends are the overflow threshold, not `ε/safmin`.
Evidence: third independent review of PR #237 (variant tree measuring zero
failures across every exponent once the kernels were scale-safe).
Revision 2026-09-25 (later): deflation gains LAPACK's absolute underflow
threshold (`dbdsqr` `maxitr·n²·unfl`; `dlahqr`'s small-subdiagonal test) and
the gates a matching floor, after the fourth review's graded random scan
found non-convergence at normal magnitudes (Bf16 near `2⁻¹²⁶`, F16 at
`2⁻¹⁰..2⁻¹⁴`).

## Context

The dense factorizations (`symmetric_eigen_qr`, `symmetric_eigen_jacobi`,
`schur`, `eigenvalues`, `singular_values`/`svd_decompose`/`pinv`,
`col_piv_qr`) form products of entries that overflow or underflow long
before the input leaves the range of the scalar: f64 SVD failed to converge
beyond `2^±256`, `RealSchur::eigenvalues` returned `{3, 3}` for `{2, 4}` at
`2⁻⁸⁶` in f32, f64 `col_piv_qr` returned rank 0 beyond `2^±512`, and F16/Bf16
Francis stalled on ordinary nonsymmetric input.

Balancing the whole matrix by a power of two only moves the problem: a local
product such as a Givens norm `a² + b²` or a Wilkinson discriminant
`δ² + t₁₂²` depends on the local magnitudes during the iteration, not on
`‖A‖_max`, and a fixed landing either leaves them unsafe or scales entries far
below the largest into underflow (`diag(1e300, 1e-300)`).

## Decision

Two tiers (`linalg::scaling`).

**Kernel tier.** Every local product in the Francis and Golub–Kahan kernels is
formed unscaled while its local magnitude `m` lies in the kernel window
`[safmin^(1/dₗ), (Ω·2⁻ᶠ)^(1/dᵤ)]` (`thresholds::kernel_window`,
`scaling::KernelWindow`), and otherwise from operands divided by the power of
two bringing `m` into `[1, 2)`. Inside the window the arithmetic is
bit-for-bit the unscaled arithmetic.

| Kernel | Product | window `(dₗ, dᵤ, f)` | LAPACK reference |
|---|---|---|---|
| `bidiagonal_qr::givens` | `a² + b² ≤ 2m²` | (2, 2, 1) | `dlartg` (`rtmin`, `rtmax`) |
| `bidiagonal_qr::qr_step` shift + first column | `δ² + t₁₂² ≤ 2m⁴`, `y ≤ 4m²` | (4, 4, 1) | `dbdsqr`, `dlas2` |
| `bidiagonal::colmajor::larfg` | `α² + ‖x‖² ≤ len·m²`, `len ≤ rows` | (2, 2, ⌈log₂ rows⌉) | `dlarfg` via `dnrm2`/`dlapy2` |
| `francis::stack_reflector` | `vᵀv ≤ 12m²` | (2, 2, 4) | `dlarfg` |
| `francis_step` first column | `x ≤ 9m²`, `y ≤ 5m²` | (2, 2, 4) | `dlahqr` (`S`-normalized `V`) |
| `standardize` 2×2 | `disc ≤ 8m²`, `ex² + ey² < 2⁵m²` | (2, 2, 5) | `dlanv2` |
| `eigenvalues_from_quasi_triangular` | `tr² − 4det ≤ 12m²` | (2, 2, 4) | `dlanv2` |

The shift window's lower degree is 4, not 2: an underflowed `t₁₂²` degrades the
Wilkinson shift to the Rayleigh shift `t₂₂`, which stagnates on a nearly equal
trailing pair (probed: a Bf16 2×2 failed at every exponent `2⁻¹³⁰..2⁻³²`).

**Deflation.** A relative deflation test alone is met only by an exact zero
once the off-diagonals it compares reach the subnormals, and the sweep then
cycles. Both iterations also deflate below an absolute floor:

- Golub–Kahan (`svd/bidiagonal_qr.rs`): `|eᵢ|` below the rounding of its
  neighbouring diagonals (the relative split, in precision-exact form) or
  `|eᵢ| ≤ 2^⌈log₂(6k²)⌉·safmin` — LAPACK `dbdsqr`'s `maxitr·n²·unfl`
  (`maxitr = 6`, `k` the bidiagonal order), rounded up to a power of two.
- Francis (`schur/francis.rs`): LAPACK `dlahqr`'s small-subdiagonal test,
  `|h_{k,k−1}| ≤ max(ulp·(|h_{k−1,k−1}| + |h_{k,k}|), floor)`, `ulp = ε`. The
  earlier precision-exact form demanded `≲ ε/2` and an 8-bit Bf16 step could
  not always reach it (the block cycled with period two). The floor is
  `2^⌈log₂ n⌉·safmin`, not `dlahqr`'s `safmin·n/ulp`, which assumes
  `ulp² ≫ safmin` and in F16 (`ε² = 2⁻²⁰ < safmin = 2⁻¹⁴`) would deflate
  subdiagonals near `0.19` at unit scale.

Each floor is `2^g·safmin`, and the routine's gate raises its lower end to
`2^g·smlnum` (below), so a floor deflation perturbs `A` by at most
`ε·‖A‖_max`.

**Matrix tier.** What the kernels cannot rescale locally is gated on
`‖A‖_max ∈ [smlnum^(1/d), (Ω·2⁻ᶠ)^(1/d)]` (`thresholds::homogeneous_safe_range`;
`smlnum = safmin/ε`, `Ω` the overflow threshold), with the lower end raised
to `2^g·smlnum` for a routine whose deflation floor is `2^g·safmin`. The bound factor enters by
exponent arithmetic only (no `128n⁴` formed in F16) and never divides
`smlnum`. `2^r ≥ ‖A‖_F/‖A‖_max` (`scaling::norm_ratio_log2`) keeps the bound
tight. Inside the range the input is factored unscaled; outside it, by the
minimal power of two back inside (`scaling::balancing_exponent`), results
multiplied back by `scaling::restore` (typed `Overflow` if unrepresentable).

| Routine | Remaining intermediate | `d` | `f` |
|---|---|---|---|
| Jacobi | `2a_pq`, `a_qq − a_pp`, diagonal partial sums `≤ 2‖A‖₂` | 1 | `1 + r` |
| Symmetric QL | chase correction `e_{l+1}·eₗ ≤ ‖A‖₂²` (`ql.rs`) | 2 | `2r` |
| Column-pivoted QR | `tail_norm_sq ≤ rows·‖A‖_max²` (`decompose.rs`) | 2 | `⌈log₂ rows⌉` |
| Francis (`schur`, `eigenvalues`) | `vᵀH`, unscaled `‖v‖₂ < √Ω` times `‖H‖_F`; floor `g = ⌈log₂ n⌉` | 2 | `2r` |
| Golub–Kahan (SVD family) | reflector dots `‖v‖₂·‖A‖_F`, `‖v‖₂ ≤ 4√M`; floor `g = ⌈log₂(6k²)⌉` | 1 | `2 + ⌈⌈log₂ M⌉/2⌉ + r` |

Each call site carries its derivation. Scaling up is exact; scaling down can
underflow an entry far below the largest, which the upper ends — the
overflow threshold itself — confine to inputs within `2^f` of overflowing.
There the backward-error bound, not entrywise exactness, is the guarantee.

## Rejected alternatives

- Recentring Francis/Golub–Kahan inputs to `[1, 4)` (the previous revision).
  Rejected: it worked around unsafe kernel products instead of removing them,
  left F16/Bf16 non-convergences recorded as exemptions, and moved inputs
  further than needed.
- Divide-by-max normalization in the kernels (the reviewer's probe variant).
  Rejected for the delivered form: a non-power-of-two divisor changes the
  rounding of in-window inputs; the power-of-two window keeps them
  bit-for-bit unscaled.
- LAPACK's `ε/safmin` upper margin for the gate. Rejected: it scales
  `diag(1e300, 1e-300)` (Jacobi) and f32 `diag(1e38, 1e-38)` down and loses the
  small entry; the kernels' own products need only the overflow threshold.
- One uniform `(d, c)` for every routine. Rejected: Jacobi is degree 1, the QL
  correction and Francis application degree 2; one range over- or
  under-scales each.

## Consequences

What the tests establish, for `f64`, `f32`, `F16` and `Bf16`:

- `tests/ops/scale_range.rs`: `SIMILAR` and `GENERAL` at every binary
  exponent from the smallest subnormal to three below the largest converge
  in `schur`, `eigenvalues`, both SVD entry points and pivoted QR, within
  Weyl/Bauer–Fike bounds on the `n²·ε` backward-error envelope; no exponent
  is exempt.
- `tests/ops/graded_scan.rs`: seeded random `2×2`–`4×4` matrices with rows
  graded by `2⁻²ⁱ`, twelve per exponent at every exponent from the smallest
  normal to three below the largest, converge in all four routines, and
  `singular_values` agrees with `svd_decompose`; random subnormal blocks
  beside a unit entry deflate at the floors.
- `tests/ops/eigen.rs`: Jacobi returns 295 diagonals per format exactly, and
  a near-overflow `M·(J₄ − 2I)` correctly through the Frobenius-ratio bound.
- The `n²·ε` tolerance is an empirical envelope with a measured margin
  (`tests/ops/scale_range.rs`), not a derived bound.

Against the pre-change tree, in-gate results are bit-identical except where a
kernel window fires — at its lower edge on a subnormal product, or at its
upper edge (the SVD shift window's `(Ω/2)^¼ ≈ 2²⁵⁶` in `f64`: f64 SVD
results at `2²⁷⁰` differ) — where Francis now deflates between `ε/2` and
`ε` relative, or where a deflation-floor gate moves a small input up. The
differential probe measured every in-gate singular-value and
symmetric-eigenvalue error, differing or not, within `0.44·n²·ε·‖A‖_F`.
- `pinv` reports `LetoError::Overflow` when a retained singular value's
  reciprocal is not finite.

## Driving evidence

`backlog.md#LETO-DENSE-SCALE-RANGE-2026-09-24` (delivered by PR #237), the
three independent reviews on PRs #233 and #237, `scaling.rs`/`thresholds.rs`
module documentation, `tests/ops/scale_range.rs`, `tests/ops/schur.rs`
(Bauer–Fike bound with an independently computed eigenvector condition), and
`tests/ops/eigen.rs` (Jacobi diagonal exactness), and
`tests/ops/graded_scan.rs` (graded random scan, subnormal-block deflation).
