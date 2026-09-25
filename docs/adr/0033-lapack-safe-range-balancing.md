<a id="adr-0033"></a>

# 0033. LAPACK safe-range balancing for dense factorizations

Status: Accepted (retroactive)

## Context

The dense factorizations (`symmetric_eigen_qr`, `symmetric_eigen_jacobi`,
`schur`, `eigenvalues`, `singular_values`/`svd_decompose`, `col_piv_qr`) form
sums of squares, Givens radii, Wilkinson shifts, and column norms. Unscaled,
these overflow or underflow long before the input matrix itself leaves the
range of the scalar (f64 SVD failed to converge below `2⁻²⁵⁸` and above
`2²⁵⁴`; `RealSchur::eigenvalues` returned `{3, 3}` for `{2, 4}` at `2⁻⁸⁶` in
f32).

The first fix (`LETO-DENSE-SCALE-RANGE-2026-09-24`) balanced every input
unconditionally: factor `2⁻ᵏ·A` instead of `A`, `k` the even exponent bringing
the largest entry into `[1, 4)`, and scale the results back. Independent
review found this masks a distinct problem: multiplying by a power of two is
exact only while every scaled entry stays representable. An entry far below
the matrix's largest one can underflow under a scale chosen for the largest
(`diag(1e300, 1e-300)` in f64), and — separately — the unconditional rescale
changed the exact rounding path even for inputs that needed no scaling at
all, turning a working `Ok` into a spurious non-convergence `Err` for some
moderately-scaled `F16` matrices (53 of 384 seeded cases in the review).

## Decision

Follow LAPACK `xSYEV`/`xGEEV` (`dsyev.f`'s `ISCALE` block): compute `‖A‖_max`
and compare it against the safe range `[rmin, rmax]`,
`rmin = √(safmin/ε)`, `rmax = 1/rmin` (`safmin` the smallest normalized
value, `ε` machine epsilon — `thresholds::safe_min`/`safe_range`, both
derived generically through the scalar's own arithmetic, never a per-format
constant). **When the norm already lies in `[rmin, rmax]`, factor the input
completely unscaled.** Only when it falls outside does the matrix balance by
the even power of two bringing its largest entry into `[1, 4)` — always
inside `[rmin, rmax]` for every scalar type carried here (as narrow as
`[0.25, 4]` for `F16`) — and the scale-carrying results are restored after.

Because `rmin·rmax = 1`, the safe range is symmetric in the binary exponent
around `1`, which is why "outside the safe range" and "bring to `[1, 4)`"
compose without contradiction: the target of the fallback scale is always a
strict subset of the safe range.

The Francis double-shift QR (`schur`, `eigenvalues`) and the Golub–Kahan
bidiagonal QR (the SVD family) gate on a narrower range,
`(√rmin, √rmax)` (`thresholds::product_safe_range`), because their
shift/discriminant formulas square an already-squared quantity — a degree-4
expression in the original entries, against the degree-2 the plain safe range
is derived for. Evidence: probing `schur`/`singular_values` of a general
(non-symmetric) `3×3` matrix across every f32 binade found
`LetoError::StorageError` ("failed to converge") specifically for norms
inside `(rmin, rmax)` but outside `(√rmin, √rmax)` — filed as
`LETO-FRANCIS-QUARTIC-SCALE-2026-09-24` pending a scale-invariant rewrite of
the shift formulas; the narrower gate is the interim mitigation, not a claim
that the underlying formulas are scale-invariant throughout the wider range.

Exactness is explicitly bounded, not asserted: scaling by a power of two can
still lose an entry far below the matrix's largest one, exactly as LAPACK's
own scaling can. That loss is covered by the factorization's backward-error
guarantee, never claimed as entrywise-exact. Inputs whose norm is already in
range are, by construction, bit-for-bit identical to the unscaled
computation — the property the unconditional-scaling design lacked.

## Rejected alternative

Keep the unconditional bring-to-`[1, 4)` scaling and instead special-case the
entries that would underflow. Rejected: this still perturbs the exact
rounding path for every already-well-scaled input (the majority), which is
what caused the reported `F16` regressions, and a per-entry underflow guard
does not address that at all — it only bounds the entrywise loss the review
separately raised.

## Consequences

- `linalg::scaling::balancing_exponent`/`balanced` (plain safe range) serve
  the symmetric tridiagonal QL, Jacobi, and column-pivoted QR.
- `linalg::scaling::balancing_exponent_for_products`/`balanced_for_products`
  (narrower, derived range) serve `schur`, `eigenvalues`, and the SVD family.
- `svd::pseudoinverse::pinv` additionally reports `LetoError::Overflow` when
  a retained singular value's reciprocal is not finite, rather than a wrong
  `Ok` of `±∞`/`NaN`.
- `F16`'s pre-existing Francis stagnation on non-symmetric input
  (`LETO-F16-FRANCIS-2026-09-24`) is unaffected by this change — it recurs at
  unit scale too, so it is a precision limit of the 11-bit format's
  double-shift arithmetic, not a scaling artifact.

## Driving evidence

`backlog.md#LETO-DENSE-SCALE-RANGE-2026-09-24`; PR review findings A–J on
PRs #233 and #237; `crates/leto-ops/src/application/linalg/scaling.rs` and
`thresholds.rs` module documentation; `tests/ops/scale_range.rs` (exhaustive
per-exponent sweep) and `tests/ops/schur.rs`'s differential regression tests.
