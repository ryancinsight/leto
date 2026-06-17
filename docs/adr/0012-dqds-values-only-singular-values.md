# ADR 0012 — Values-only singular values: dqds vs the bidiagonal Givens QR sweep

- Status: investigated, prototype reverted, scoped as a [major] follow-up
- Date: 2026-06-17
- Change class: [major] (new values-only kernel + shift/splitting machinery)

> **Correction (read first).** An earlier revision of this ADR framed the 64²
> gap as *algorithmic* — "nalgebra is fast because it uses dqds; leto's Givens is
> 2√+2÷ vs dqds' 0√+1÷." **That premise is false.** nalgebra 0.32's
> `SVD::try_new_unordered` uses the **same** implicit-shift **Givens** QR sweep
> leto does (verified in `nalgebra/src/linalg/svd.rs`: `wilkinson_shift`,
> `GivensRotation::cancel_y`, `compute_2x2_uptrig_svd`). So the ~1.9× gap is a
> **per-step / per-element implementation constant within the same algorithm**,
> not a Givens-vs-dqds difference, and the phase split below is approximate. dqds
> remains a *theoretically* cheaper kernel (fewer transcendentals) that would beat
> **both** implementations if made fast — but it is not the explanation for
> nalgebra's lead, and the prototype did not beat leto's own Givens path.

## Context: the measured disparity

`leto-ops::singular_values` (Golub–Reinsch: Householder bidiagonalization +
implicit-shift Givens QR sweep on the bidiagonal `(d, e)`) runs ≈**1.9–2.4×**
slower than `nalgebra` on the 64×64 `decomposition_compare` benchmark
(`pinned_values(n², 1e-3)`, a graded near-rank-1 matrix). Repeated micro-tuning
of the existing path had stalled, attributing it to an unidentifiable constant.

### Profiler-free phase attribution (this investigation)

A same-session probe split both libraries into their two phases (mean over 2000
reps, `target-cpu=native`, AVX2 host):

| Phase                 | leto     | nalgebra | ratio |
|-----------------------|----------|----------|-------|
| Householder bidiag    | 63.8 µs  | 37.2 µs  | 1.72× |
| values sweep + finish | 50.2 µs  | 22.3 µs  | 2.25× |
| total                 | 114 µs   | 59 µs    | 1.92× |

The Givens sweep is the **worse ratio (2.25×)** even though leto's shift
converges in **92 steps = 1.44 steps/singular-value** — better than the typical
~2/value, i.e. convergence is *excellent* and not the cause.

### Causes ruled out by direct experiment (no profiler needed)

- **Convergence / shift quality** — 92 steps, 1.44/value (counted).
- **Bounds checks** — rewriting `qr_step` with `get_unchecked` (sound; indices
  `∈ {p, k, k±1}`, `k ∈ [p,q-1]`, `q < len`) left the sweep time **unchanged**:
  the compiler already elided them.
- **Trait dispatch** — `RealScalar` f64 arithmetic is `#[inline(always)]` over
  native ops; zero overhead.
- **hermes `dot` per-call dispatch / middle-loop serialization** — `target-cpu=native`
  selects the inlinable compile-time arm; the dot middle-loop FMA fix (hermes
  `219f2fb`) did not move the 64² number (the values sweep is `VEC=false`,
  pure scalar — it does not call `dot` at all).

The micro-causes above are ruled out, but the residual is **not** isolated to a
single dominant factor: it is a per-step/per-element constant within the same
Givens algorithm nalgebra uses. Candidate remaining factors (unconfirmed, would
need instruction-level profiling): nalgebra's fixed-size `Matrix2x3`/`Vector2`
rotation kernels monomorphize to fully-unrolled small-vector code, its
`delimit_subproblem` re-runs after *every* step (possibly tighter active windows),
and its iteration count may differ from leto's measured 92. The phase split above
is approximate (nalgebra's standalone `Bidiagonal::new` is not identical to the
bidiagonalization inside its SVD), so the per-phase ratios should be read as
indicative, not exact.

## dqds as a candidate fix (cheaper kernel, not the cause of the gap)

LAPACK's values-only path (`dbdsqr`/`dlasq`) uses **dqds** — the differential
quotient–difference algorithm with shifts — on the *squared* qd array
`q[i] = dᵢ²`, `ee[i] = eᵢ²`. One sweep is the recurrence

```
d ← q[0] − τ;  for i: q'[i] ← d + ee[i];  t ← q[i+1]/q'[i];  ee'[i] ← ee[i]·t;  d ← d·t − τ;  q'[n-1] ← d
```

costing **0 √ and 1 ÷ per element**, with one `√` per converged value at the end.

**Theorem (exact shift-similarity).** A dqds sweep maps `(q, ee)` to a qd array
representing `B'ᵀB' = BᵀB − τI`; it is the Rutishauser `LR` similarity applied to
`T − τI`, so `eig(q', ee') = eig(q, ee) − τ`. The differential form propagates the
running Schur complement `d` instead of forming `q'`/`ee'` by subtraction,
avoiding cancellation, and reads `q[i+1]`/`ee[i]` before overwriting them, so it
runs **in place**. ∎

**Key consequence for safety:** value-correctness depends *only* on the transform
being algebraically exact and the final `σ = √λ`. The shift `τ` is purely a
speed/stability knob; non-convergence is loud (iteration cap → typed error),
never silently wrong.

## What was prototyped and measured (then reverted)

A complete, generic-over-`RealScalar` dqds kernel was implemented and verified
**correct** against the full SVD differential suite (17 tests incl. the nalgebra
battery, wide dynamic range, rank-deficient/zero σ, diagonal closed-form, f32).
Two shift/splitting strategies were measured on the 64² benchmark matrix:

| Variant                                   | sweeps | result        |
|-------------------------------------------|--------|---------------|
| no-split, shift = ¼·dmin estimate         | 384    | 0 fallbacks   |
| no-split, shift = ½·dmin estimate         | 300    | 1 fallback; **181 µs total — a regression vs Givens' 128 µs** |
| recursive interior splitting + ½ shift    | 403    | **broke the rank-deficient case**, no speedup |

**Findings.**
1. A *simple* dqds (single global shift, no block splitting) does **full-length**
   sweeps and is throttled to the global smallest eigenvalue; on this graded
   matrix it needs ~300 sweeps and is **slower** than the Givens path — which
   here converges exceptionally (1.44 steps/value), a near-best case for Givens.
2. Adding interior splitting (the real source of `dlasq`'s speed) is numerically
   delicate: the naïve recursion mishandled zero-`q` (rank-deficient) blocks and
   did not reduce sweeps without the proper Fernando–Parlett shift.

## Decision

**Keep the implicit-shift Givens QR sweep** for `singular_values` (correct, no
regression, excellent convergence). **Do not ship** a partial dqds: the simple
form regresses and the fast form requires the full `dlasq` split-and-shift
machinery (dlasq2/3/4 shift cases + ping-pong + block splitting with per-block
shift accounting). Scope that as a dedicated [major] item, gated by:

- differential parity with nalgebra across the existing battery **and** adversarial
  clustered/tiny/zero/wide-dynamic-range inputs (the rank-deficient case is the
  known trap);
- a measured win on the 64²/256² benchmarks (not just asymptotic) before merge —
  the prototype shows "asymptotically better" is not sufficient at `n = 64`.

## Consequences

- The 64² disparity is **not** algorithmic: leto and nalgebra run the same
  implicit-shift Givens sweep, so the ~1.9× is a per-step/per-element
  implementation constant that this investigation narrowed (ruling out
  convergence, bounds checks, dispatch, inlining) but did not isolate to a single
  cause. Closing it likely needs matching nalgebra's fixed-size rotation kernels
  / active-window tightness — or the dqds kernel below, which would undercut both.
- dqds stays a scoped [major] lever (fewer transcendentals per element) but is
  **not** the reason nalgebra leads; its value is an absolute speedup over Givens,
  contingent on the full `dlasq` machinery (the simple/split prototypes regressed
  or broke rank-deficient correctness).
- The verified-correct dqds transform/theory lives in this ADR and git history
  (prototype on the working tree was reverted to avoid shipping dead/slow code).
- Evidence tier: criterion measurements + 17-test differential suite +
  profiler-free phase/step attribution (phase split approximate). No
  machine-checked proof performed.
