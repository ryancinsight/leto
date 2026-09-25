<a id="adr-0033"></a>

# 0033. Per-routine safe-range balancing for dense factorizations

Status: Accepted

## Context

The dense factorizations (`symmetric_eigen_qr`, `symmetric_eigen_jacobi`,
`schur`, `eigenvalues`, `singular_values`/`svd_decompose`, `col_piv_qr`) each
rely on some largest intermediate quantity that overflows or underflows long
before the input matrix itself leaves the range of the scalar (f64 SVD failed
to converge below `2⁻²⁵⁸` and above `2²⁵⁴`; `RealSchur::eigenvalues` returned
`{3, 3}` for `{2, 4}` at `2⁻⁸⁶` in f32).

The first fix (`LETO-DENSE-SCALE-RANGE-2026-09-24`) balanced every input
unconditionally: factor `2⁻ᵏ·A` instead of `A`, `k` the even exponent bringing
the largest entry into `[1, 4)`, and scale the results back. Independent
review found this masks a distinct problem: multiplying by a power of two is
exact only while every scaled entry stays representable. An entry far below
the matrix's largest one can underflow under a scale chosen for the largest
(`diag(1e300, 1e-300)` in f64), and — separately — the unconditional rescale
changed the exact rounding path even for inputs that needed no scaling at
all, turning a working `Ok` into a spurious non-convergence `Err` for some
moderately-scaled F16 matrices.

## Revision (2026-09-25)

A second independent review of the first fix to this ADR (the plain LAPACK
`dsyev`-style `[rmin, rmax] = [√(safmin/ε), 1/rmin]` gate applied uniformly,
minimal-move scaling everywhere) found two further defects, both now
corrected below:

1. **One safe range does not fit every routine.** `[√(safmin/ε), 1/rmin]` is
   derived for an intermediate that squares the input once (degree 2). Using
   it for Jacobi (whose rotation update is degree 1) needlessly scaled inputs
   like `diag(1e300, 1e-300)` that Jacobi's own arithmetic never needed
   scaled, losing the small entry; using it for Francis/bidiagonal-QR (whose
   shift/discriminant is degree 4) left them scaling too rarely, so the
   probe matrix from finding G flipped `Ok → Err` in F16. The range must be
   derived per routine from its actual formulas.
2. **Minimal-move scaling is not safe for every routine.** Once a norm is
   judged out of range, moving it the least distance back in (rather than
   recentring to a fixed target) preserves the most precision in every other
   entry — correct for Jacobi, the symmetric tridiagonal QL, and
   column-pivoted QR, all confirmed by exhaustive per-exponent sweeps. For
   Francis and bidiagonal QR it is not: probing `schur` on a fixed
   nonsymmetric matrix across every exponent found the minimal-move landing —
   at or near the derived boundary — non-convergent across a wide band (Bf16:
   nearly every exponent from `2⁻¹³³` to `2⁻³³`; f32: the smallest subnormal
   exponent). Recentring the same out-of-range inputs to `[1, 4)` converges
   throughout. These two algorithms' shift formulas are evidently more
   fragile near the derived boundary than the degree/dimension analysis alone
   predicts — a gap the range derivation does not close, tracked separately
   below.

## Decision

**Per-routine degree and dimension factor.** For a routine whose largest
relied-upon intermediate is bounded above by `c·‖A‖_max^d` (`d` the degree,
`c` a derived dimension factor — never a magic constant), the safe range for
`‖A‖_max` is `[(safmin/(ε·c))^(1/d), (ε/(safmin·c))^(1/d)]`
(`thresholds::homogeneous_safe_range`), generalizing LAPACK `dsyev`'s
`d = 2, c = 1` case. **Inside that range, the routine factors its input
completely unscaled.** Only outside it does balancing apply, and the
scale-carrying results are restored after. Derivations, each citing its
source lines:

| Routine | Formula | `d` | `c` |
|---|---|---|---|
| Symmetric tridiagonal QL (`symmetric_eigen_qr`) | Householder reflector self-normalizes (`householder::reflect_in_place`); QL sweep's shift is ratio/`hypot`-based (`ql.rs`) — degree 1 throughout, but the shift ratio's denominator (`off_diagonal[l]`) underflowing to exact `0` while the numerator is still finite produces `0·∞ = NaN` at extreme scale (probed: f64 `2⁵³⁷`) unless kept comfortably clear of `safmin`/`bignum` — the plain `d = 2` LAPACK margin, not the wider `d = 1` bound self-normalization alone would justify, is what the sweep confirms is actually safe | 2 | `n` (Gershgorin/Frobenius: eigenvalues bounded by `n·‖A‖_max`) |
| Jacobi rotation (`symmetric_eigen_jacobi`/`symmetric_eigenvalues_jacobi`) | `rotate()`'s `app' = c²·app − 2sc·apq + s²·aqq` is a convex-combination-bounded update (`c² + 2\|sc\| + s² ≤ 2`, since `c²+s²=1`) — degree 1, confirmed safe at every probed exponent from `2⁻¹⁰⁰⁰` to `2¹⁰⁰⁰` | 1 | `n` (same Gershgorin bound) |
| Column-pivoted QR pivoting (`col_piv_qr`) | `decompose.rs`'s `tail_norm_sq`, a raw `Σrᵢ²` over up to `max(m, n)` rows | 2 | `max(m, n)` |
| 2×2 quasi-triangular block quadratic (`RealSchur::eigenvalues`) | `tr = a+d`, `det = ad−bc` (each degree 2); discriminant `tr² − 4·det ≤ 12·block_max²` | 2 | `12` |
| Francis double-shift reflector (`schur`, `eigenvalues`) | Hessenberg entries `≤ n·‖A‖_max` (orthogonal similarity); `x, y, zz` (`francis.rs`'s double-shift step) sum four such degree-2 terms, `≤ 6n²·‖A‖_max²`; `stack_reflector`'s `x²+y²+zz²` squares that again | 4 | `128n⁴` (derived `108n⁴`, rounded up) |
| Bidiagonal-QR Wilkinson shift (SVD family) | Bidiagonal entries `≤ max(rows, cols)·‖A‖_max`; `wilkinson_shift`'s `delta² + t12²` | 4 | `8M⁴`, `M = max(rows, cols)` (derived `5M⁴`, rounded up) |

**Exponent policy.** Once out of range: Jacobi, the symmetric tridiagonal QL,
and column-pivoted QR scale by the *minimal* integer exponent restoring the
range (`scaling::balancing_exponent`) — proven, by the revision's exhaustive
sweeps, to preserve small entries `diag(1e300, 1e-300)`-style inputs would
otherwise lose. Francis and bidiagonal QR instead recentre to `[1, 4)`
(`scaling::balancing_exponent_recentered`) per the revision's evidence above.

Exactness is explicitly bounded, not asserted: scaling by a power of two can
still lose an entry far below the matrix's largest one. That loss is covered
by the factorization's backward-error guarantee, never claimed as
entrywise-exact. Inputs whose norm is already in range are, by construction,
bit-for-bit identical to the unscaled computation.

## Rejected alternatives

- Keep the unconditional bring-to-`[1, 4)` scaling and special-case
  underflowing entries. Rejected: still perturbs the exact rounding path for
  every already-well-scaled input, which caused the reported F16
  regressions; a per-entry guard does not address that.
- Use LAPACK's plain `d = 2, c = 1` range uniformly for every routine.
  Rejected by the revision above: it neither matches Jacobi's actual (lower)
  degree nor Francis/bidiagonal-QR's actual (higher) degree, so it both
  over-scales (Jacobi) and under-scales (Francis/SVD) relative to what each
  routine's own formulas need.
- Minimal-move scaling uniformly for every routine (the revision's own first
  attempt). Rejected by direct evidence: it leaves Francis and bidiagonal QR
  non-convergent across a wide exponent band for Bf16 and at f32's smallest
  subnormal exponent — a regression relative to the unconditional-scaling
  baseline, not an improvement.

## Consequences

- `linalg::scaling::balancing_exponent`/`balanced` (minimal move) serve the
  symmetric tridiagonal QL, Jacobi, and column-pivoted QR, each with its own
  `(degree, dimension_factor)`.
- `linalg::scaling::balancing_exponent_recentered`/`balanced_recentered`
  (recentre to `[1, 4)`) serve `schur`, `eigenvalues`, and the SVD family.
- `svd::pseudoinverse::pinv` additionally reports `LetoError::Overflow` when
  a retained singular value's reciprocal is not finite, rather than a wrong
  `Ok` of `±∞`/`NaN`.
- F16's pre-existing Francis stagnation on non-symmetric input
  (`LETO-F16-FRANCIS-2026-09-24`) recurs at unit scale too (confirmed by
  direct probe both before and after this revision), so it is a precision
  limit of the 11-bit format's double-shift arithmetic, not a scaling
  artifact; the same class was found, more rarely, for Bf16 (8-bit mantissa)
  and is now covered by the same recorded-failure exemption in
  `tests/ops/scale_range.rs`.
- `LETO-FRANCIS-QUARTIC-SCALE-2026-09-24` tracks the open gap this revision
  works around rather than closes: Francis's and bidiagonal QR's
  shift/discriminant formulas are not proven scale-invariant near the
  derived boundary, which is why they need recentring instead of the minimal
  move the other routines use. Deriving scale-invariant forms of those
  formulas would let them adopt the minimal move too.

## Driving evidence

`backlog.md#LETO-DENSE-SCALE-RANGE-2026-09-24`,
`#LETO-FRANCIS-QUARTIC-SCALE-2026-09-24`; two independent PR reviews on
PRs #233 and #237; `crates/leto-ops/src/application/linalg/scaling.rs` and
`thresholds.rs` module documentation; `tests/ops/scale_range.rs` (exhaustive
per-exponent sweep) and `tests/ops/schur.rs`'s differential regression tests.
