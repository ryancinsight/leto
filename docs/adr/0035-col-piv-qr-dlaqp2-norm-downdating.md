<a id="adr-0035"></a>

# ADR 0035: Column-pivoted QR selects pivots by LAPACK DLAQP2 norm downdating

Status: Accepted

Board item `LETO-COLPIVQR-NORM-DOWNDATE-2026-09-27`; delivered by PR #290.

## Context

`col_piv_qr` recomputed the squared tail norm of every remaining column at
every step, `O(p·n·m)` work for `p = min(m, n)` steps, the same order as the
reflector application. LAPACK 3.12.0 `DLAQP2` caches the norms, downdates each
by the entry the reflector just placed in the pivot row, and recomputes only
when the downdate cancels. The board item asked for that downdate "verified
against the exact-recompute baseline: identical pivot sequence".

Identical pivots are not obtainable from a downdated key. The cached key
carries the rounding of the downdate, so two columns whose exact tails differ
by less than that error can be ranked either way. LAPACK specifies no order
there, and a column-pivoted QR is correct for any order that keeps `A·P = Q·R`,
`Q` orthonormal, and the pivots dominant to the stated tolerance.

## Decision

**Pivot rule.** The pivot at step `k` is the first maximum of the cached
squared tail norms `c_j` over the columns `j ≥ k` (`dlaqp2.f` line 197,
`IDAMAX`). Squares are compared, so ties break exactly as they do on norms
except where rounding the root would merge distinct keys.

**Downdate.** After the reflector of step `k`, each later column's removed
entry is `a = R[k, j]`. The estimate `c − a²` is clamped at 0. With reference
`c_ref`, the key at the column's last exact computation, the key is recomputed
from rows `k + 1..m` when `estimate / c_ref ≤ TOL3Z = √u`, `u = ε/2` the unit
roundoff (`dlaqp2.f` line 187, `DLAMCH('Epsilon')`), and the recomputed value
becomes the new reference; otherwise `c` becomes the estimate (lines 236-249).
`DLAQP2`'s test `(1 − (|a|/vn1)²)(vn1/vn2)² ≤ TOL3Z` equals
`(vn1² − a²)/vn2² ≤ TOL3Z`, so the state is kept as squared norms and no
root is taken. A cached zero stays zero (line 231). Cost: `O(n − k)` per step.

**Rank test.** The step breaks when the exact tail norm of the selected column
is at most `tol = 1e-12·max_j ‖A[:, j]‖`, evaluated by `O(m − k)` summation, as
the exact-recompute baseline evaluated its maximum. The decision therefore
differs from the baseline only when the largest exact squared tail lies within
a factor `1 + τ` above `tol²`.

**Contract.** For every step `k < rank` and every later column `j`, the exact
tail `S_j = Σ_{i ≥ k} R[i, j]²` of the returned `R` satisfies
`S_j ≤ (1 + τ)·R[k, k]²`, with `τ` from the derivation below. Diagonal
monotonicity is its `j = k + 1` instance. Alongside it hold `‖A·P − Q·R‖_F ≤
η‖A‖_F` (`backward_error::col_piv_qr`) and the orthonormality bound below. `R`
is upper triangular by construction (the sub-diagonal is zeroed), so that is
not asserted. The order of columns with `S_j` within `τ` is unspecified. Where
a bound is vacuous (`≥ 1`) its clause is not asserted.

**Orthonormality bound.** `Q̂` accumulates the `p` reflectors into `I` by
right application. Each application moves a row by at most `η` of its norm, so
every row of `Q̂` lies within `(1 + η)^p − 1` of the exact unit row of `Q`, and
`Δ = Q̂ − Q` has `‖Δ‖_F ≤ e = √m·((1 + η)^p − 1)`. Then
`Q̂ᵀQ̂ − I = QᵀΔ + ΔᵀQ + ΔᵀΔ` with `Q` orthogonal gives
`‖Q̂ᵀQ̂ − I‖_F ≤ 2e + e²`.

**Derivation of `τ`** (`backward_error::col_piv_qr_pivot_slack`; `m` rows,
`p` steps, `η = γ_{8m+29}` the one-reflector bound, `χ = (1 + η)² − 1`).
A reflector application moves the squared norm of the rows `≥ k` of a column
by at most `χ` relatively, so the exact tails `s` of the stored entries stay
below `Λ·c_ref`, `Λ = (1 + η)^{2p}/(1 − γ_{m+1})`; the exact computation of
`c_ref` errs by `γ_{m+1}`. One downdate adds at most `(χ + u(2 + χ))Λ·c_ref`
to the key's absolute error `E` (drift of `s`, rounding of `a²`, rounding of
the difference), so `E_{t+1} = (E_t + (χ + u(2 + χ))Λ)/(1 − u)`, `E_0 = γ_{m+1}Λ`
in units of `c_ref`. A key that survives the test has `c > TOL3Z·c_ref`, hence
relative error `ρ = E_p/TOL3Z` and `s/(1 + ρ) ≤ c ≤ s/(1 − ρ)`. The pivot has
the largest key, so each competitor has `s_j ≤ s_pivot(1 + ρ)/(1 − ρ)`. The
returned `R` is `≤ p` further applications from the state at step `k`
(`(1 + η)^{2p}` on a tail sum) and `R[k, k]² ≥ s_pivot(1 − η)²`:

`1 + τ = ((1 + ρ)/(1 − ρ))·(1 + η)^{2p}/(1 − η)²`, `+∞` once `ρ ≥ 1`.

| format | `τ` at 4×3 | at 16×16 | at 256×256 |
|---|---|---|---|
| `f64` | 8.1e-6 | 1.1e-4 | 2.3e-2 |
| `f32` | 0.20 | vacuous | vacuous |
| `F16`, `Bf16` | vacuous | vacuous | vacuous |

**Tests.** `tests/ops/col_piv_qr.rs` asserts the contract on every fixture
through one generic routine over `f64`, `f32`, `F16`, `Bf16`: analytic ranks
(`f64` only for the rank-deficient matrix, whose rounding noise exceeds the
threshold elsewhere), seeded matrices up to 33×17, and scaled Hadamard columns
(squared norms `4, 64, 16, 256`) with the exact sequence `[3, 1, 2, 0]`
wherever `τ < 3`. Three fixtures make the recompute observable through the
permutation, each derived in its doc comment: a cancelled key that only the
recompute restores (`[0, 2, 1]`, all formats); a stale `f64` estimate of ratio
9.992e-15 to its reference, recomputed only under `TOL3Z = √u` on the unrooted
ratio (`[0, 2, 1]`, gap 4.0e-4 against `τ = 7.0e-6`); and a stale estimate of
ratio 1.0e-12 to the reference but 1.0e-7 to the current key, recomputed only
with the reference as denominator (`[0, 1, 2, 3]`, gap 2.5e-5 against
`τ = 1.07e-5`). Replacing `TOL3Z` by `ε`, `u` or 0, rooting the ratio, or
dividing by the current key each fails the suite (PR #290).

*Replaced assertions.* The downdate commit of PR #290 asserted exact
permutations and bitwise `Q`, `R` equality with an exact-recompute reference
on fixtures whose tails differ by a relative `3ε/4` or `ε`, below
`τ = 8.1e-6` (4×3, `f64`). Those assertions fixed an order the contract leaves
open, a wrong specification, the one admissible ground for changing a test. No
assertion on `main` is weakened: its underived `1e-9` tolerances become the
derived bounds (`5.3e-13` for the 4×3 `f64` reconstruction).

## Rejected: interval-certified exact order

Each cached key carried a conservative enclosure (Higham `γ` bounds, widened
by a reflector-drift term) and a pivot was accepted only when strictly
separated from every competitor, else every remaining key was recomputed in
baseline order. That preserves the baseline pivot order bit for bit. It
measured 1.5-3.7x slower than `main` (table below): the enclosure update
per column per step costs more than the recompute it avoids, and the drift
term makes intervals overlap often enough that the full recompute still runs.
`Bf16` never certifies. Exact order against a recompute is not a property of
the algorithm's contract, so its price buys nothing a caller can rely on.

## Measurement

`col_piv_qr_scaling`, `f64`, square orders, criterion median of each run, then
the median over 7 rounds interleaved base, PR, interval (order reversed on
even rounds). Base is `main` plus the bench file; the release bench binaries
were built from `git archive` exports of each tree. Each ran pinned to one
P-core at high priority on a 24-core Intel Core Ultra 9 285K host shared with
peer builds (total CPU load 16-88 % and 0-13 cargo/rustc processes at launch,
recorded per run), so the absolute times carry that noise; the per-round ratio
range is the bound on the ratio.

| order | base | DLAQP2 (this ADR) | per-round ratio | interval-certified | per-round ratio |
|---|---|---|---|---|---|
| 64 | 0.170 ms | 0.138 ms (0.81x) | 0.79-0.86 | 0.626 ms (3.68x) | 3.58-3.80 |
| 128 | 1.159 ms | 0.816 ms (0.70x) | 0.64-0.71 | 2.748 ms (2.37x) | 2.15-2.44 |
| 256 | 8.761 ms | 6.032 ms (0.69x) | 0.60-0.71 | 13.119 ms (1.50x) | 1.30-1.54 |

Every PR round was faster than its paired base round at every order.

## Consequences

- Pivot sequences, and so `Q` and `R`, differ from the former exact-recompute
  results for columns within `τ`; `A·P = Q·R`, orthonormality, and rank (except
  within `1 + τ` of the threshold) are unchanged to the stated bounds.
- `householder.rs` records that Hermes `axpy` fuses the multiply-add
  (`hermes-simd` `dispatch/axpy.rs`, `fmadd`), so bitwise agreement with a
  separately rounded multiply-add never held; the prior comment claimed it.
- The test-side bounds live in `backward_error.rs` beside the other derived
  bounds and are reused by any pivoted factorization with the same recurrence.

## Overturning evidence

A stack consumer that needs the pivot sequence reproducible against an exact
recompute (a recorded case, not a preference) reopens the choice: the answer is
an opt-in exact mode with its measured cost, not the default. A measured
`ρ`-sized rank flip in a downstream least-squares or rank-revealing solve also
reopens it, as does a `col_piv_qr_scaling` median at or above `main`.
