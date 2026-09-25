<a id="adr-0033"></a>

# 0033. Scale-safe kernels with a minimal-move matrix gate for dense factorizations

Status: Accepted

Revision 2026-09-25: Francis and Golub–Kahan kernels now form their products
scale-safely (LAPACK `dlartg`/`dlarfg`/`dlahqr`/`dlanv2` pattern), so the
`[1, 4)` recentring exception and every recorded F16/Bf16 non-convergence
are gone; gate upper ends are the overflow threshold, not `ε/safmin`.
Evidence: third independent review of PR #237 (variant tree measuring zero
failures across every exponent once the kernels were scale-safe).

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
| `bidiagonal::colmajor::larfg` | `α² + ‖x‖² ≤ len·m²` | (2, 2, ⌈log₂ m⌉) | `dlarfg` via `dnrm2`/`dlapy2` |
| `francis::stack_reflector` | `vᵀv ≤ 12m²` | (2, 2, 4) | `dlarfg` |
| `francis_step` first column | `x ≤ 9m²`, `y ≤ 5m²` | (2, 2, 4) | `dlahqr` (`S`-normalized `V`) |
| `standardize` 2×2 | `disc ≤ 8m²`, `ex² + ey² < 2⁵m²` | (2, 2, 5) | `dlanv2` |
| `eigenvalues_from_quasi_triangular` | `tr² − 4det ≤ 12m²` | (2, 2, 4) | `dlanv2` |

The shift window's lower degree is 4, not 2: an underflowed `t₁₂²` degrades the
Wilkinson shift to the Rayleigh shift `t₂₂`, which stagnates on a nearly equal
trailing pair (probed: a Bf16 2×2 failed at every exponent `2⁻¹³⁰..2⁻³²`).

**Matrix tier.** What the kernels cannot rescale locally is gated on
`‖A‖_max ∈ [smlnum^(1/d), (Ω·2⁻ᶠ)^(1/d)]` (`thresholds::homogeneous_safe_range`;
`smlnum = safmin/ε`, `Ω` the overflow threshold). The bound factor enters by
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
| Francis (`schur`, `eigenvalues`) | `vᵀH`, unscaled `‖v‖₂ < √Ω` times `‖H‖_F` | 2 | `2r` |
| Golub–Kahan (SVD family) | reflector dots `‖v‖₂·‖A‖_F`, `‖v‖₂ ≤ 4√M` | 1 | `2 + ⌈⌈log₂ M⌉/2⌉ + r` |

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

- No exponent of any format is exempt in `tests/ops/scale_range.rs`
  (`SIMILAR` and `GENERAL`, `schur`/`eigenvalues`/SVD/pivoted QR, all four
  scalars); Jacobi factors diagonals exactly across 294 pairs per format.
- In-range inputs are bit-for-bit the unscaled computation wherever the
  kernels' products are normal; where the pre-change arithmetic formed a
  subnormal product, the lower kernel window rescales it and the result
  differs (evidence: disabling only the lower windows removes every in-range
  difference against the pre-change tree).
- `pinv` reports `LetoError::Overflow` when a retained singular value's
  reciprocal is not finite.

## Driving evidence

`backlog.md#LETO-DENSE-SCALE-RANGE-2026-09-24` (delivered by PR #237), the
three independent reviews on PRs #233 and #237, `scaling.rs`/`thresholds.rs`
module documentation, `tests/ops/scale_range.rs`, `tests/ops/schur.rs`
(Bauer–Fike bound with an independently computed eigenvector condition), and
`tests/ops/eigen.rs` (Jacobi diagonal exactness).
