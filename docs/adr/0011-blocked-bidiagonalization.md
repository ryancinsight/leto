# ADR 0011: Blocked bidiagonal reduction (`dgebrd`/`dlabrd`) for singular values

- Status: **Implemented, oracle-verified, then REVERTED as measured-regressive.**
  Superseded conclusion below (the "disparity" is a small-matrix artifact; leto's
  unblocked reduce is already at parity at scale).

## Outcome (measured)

The blocked reduction (`dlabrd` panel `X`/`Y` look-ahead + `dgebrd` two-GEMM
trailing update) was implemented in row-major from the grounded reference and
**verified correct**: 192² blocked singular values matched nalgebra to `1e-9·σ₁`
(the blocked = unblocked theorem holds in practice). But A/B benchmarks show it is
a **regression at every size**:

| size | unblocked | blocked | nalgebra |
| --- | --- | --- | --- |
| 256² | 4.69 ms | 5.68 ms | 3.67 ms |
| 512² | 32.7 ms | 38.0 ms | 31.9 ms |

Two facts overturn the original premise:
1. **leto's *unblocked* bidiagonalization is already at parity at scale** — 512²
   `1.03×` nalgebra, 256² `1.28×`. The `~2.25×` figure that motivated this ADR is a
   **small-matrix (64²) fixed-overhead artifact** (per-reflector allocation/setup
   amortizes poorly when `n` is small), *not* a structural large-`n` throughput gap.
2. **Blocking is the wrong lever for that artifact**: `dlabrd`'s `X`/`Y`
   accumulators ~double the flops to convert the trailing update to BLAS-3; that
   trade only pays when the BLAS-3/BLAS-2 speed ratio is large enough (nalgebra's
   tuned kernels), and it adds *more* fixed overhead — so it regresses exactly the
   small-`n` regime where the gap lives, and loses to the already-good unblocked
   path at large `n`.

Reverted per the anti-regression / justified-constructs discipline (a correct but
slower kernel is not shipped). The grounded algorithm above is retained as the
reference should a future need arise (e.g. a much faster GEMM, or `m ≫ n` where the
panel overhead amortizes differently). The genuine, bounded residual is the
small-`n` fixed overhead in the unblocked reduce (allocation reuse), tracked
separately — low ROI given parity at scale.

---
Original design (grounded reference, retained for the record):

- Status: Accepted (design; grounded in the LAPACK reference, ready to implement)
- Date: 2026-06-17
- Class: [minor] (additive fast path behind a size gate; no API/numeric contract
  change — the differential singular-value oracle is the gate)

## Context

`singular_values` is the one dense-LA kernel still off parity (~2.25× nalgebra at
64²). Profiling (warm 64²) splits it REDUCE ~82 µs + SWEEP ~48 µs vs nalgebra ~78 µs
*total*. A flop analysis proves the REDUCE gap is **not** allocation or
constant-factor: leto's 82 µs is already *below* a naive per-reflector SIMD
estimate (~131 µs), so the axpy applies are well vectorized. nalgebra's ~40 µs
reduce wins by **higher sustained flop/ns from a blocked GEMM trailing update**
(`dlabrd`/`dgebrd`) vs leto's per-reflector axpy passes (bandwidth-bound). The only
lever is blocking.

The implementation blocker was *grounding*: the correctness rests on the precise
two-sided look-ahead accumulators `X`/`Y`, which must not be reconstructed from
memory (codebase_fidelity: never invoke an algorithm from memory; never fabricate).
This ADR records the **exact algorithm grounded in the LAPACK 3.1.1 reference**
(`dlabrd.f`, `dgebrd.f`), removing that blocker.

## Decision

Add `linalg/bidiagonal/blocked.rs` — a blocked reduction for `m ≥ n` (upper
bidiagonal), gated on `BLOCK_MIN_DIM` (≈160, the measured QR-class crossover), used
by the values-only `bidiagonal_values`; below the gate the existing unblocked sweep
runs unchanged. The full `bidiagonalize` (with `U`/`V`) stays unblocked initially
(it is already faster than nalgebra).

### Theorem (blocked = unblocked)
The bidiagonal `B` (hence the singular values) equals the unblocked sweep's: every
reflector `dlabrd` generates is the one the unblocked sweep would, because `X`/`Y`
reconstruct each panel column/row *as if* all prior reflectors had hit the full
trailing block (the deferred two-sided contribution); the trailing GEMMs then
realise the identical bulk transform. Identical up to reflector-sign / FP-reorder
freedom the oracle tolerates. ∎

### Grounded algorithm — `dlabrd`, UPPER, panel of `NB` (0-based, row-major)

Row-major translation: a no-transpose `M·v` is a contiguous **row-dot** per output
(`dot_slice`); a transpose `Mᵀ·v` is a contiguous **row-axpy** accumulation
(`axpy_slice`). For each `i` in `0..NB` of the trailing submatrix `A` (`m×n`,
leading dim `lda`):

1. `A(i:m,i) -= A(i:m,0:i)·Y(i,0:i)` (no-trans) then `-= X(i:m,0:i)·A(0:i,i)` (no-trans).
2. `larfg(A(i:m,i)) → τq(i), d(i)`; set `A(i,i)=1` (implicit unit), tail = `v`.
3. If `i+1<n`:
   - `Y(i+1:n,i) = A(i:m,i+1:n)ᵀ·A(i:m,i)` (trans);
     `ty = A(i:m,0:i)ᵀ·A(i:m,i)`, `Y(i+1:n,i) -= Y(i+1:n,0:i)·ty`;
     `ty2 = X(i:m,0:i)ᵀ·A(i:m,i)`, `Y(i+1:n,i) -= A(0:i,i+1:n)ᵀ·ty2`; `Y(i+1:n,i) *= τq`.
   - `A(i,i+1:n) -= Y(i+1:n,0:i+1)·A(i,0:i+1)` then `-= A(0:i,i+1:n)ᵀ·X(i,0:i)`.
   - `larfg(A(i,i+1:n)) → τp(i), e(i)`; set `A(i,i+1)=1`, tail = `w`.
   - `X(i+1:m,i) = A(i+1:m,i+1:n)·w` (no-trans);
     `tx = Y(i+1:n,0:i+1)ᵀ·w`, `X(i+1:m,i) -= A(i+1:m,0:i+1)·tx`;
     `tx2 = A(0:i,i+1:n)·w`, `X(i+1:m,i) -= X(i+1:m,0:i)·tx2`; `X(i+1:m,i) *= τp`.

### Grounded driver — `dgebrd` block step
After the panel: update `A22 = A(i+NB:m, i+NB:n)` by two GEMMs (`tiled_gemm`):
`A22 -= V·Yᵀ` with `V = A(i+NB:m, i:i+NB)`, `Y = Y(i+NB:n, 0:NB)`; and
`A22 -= X·U` with `X = X(i+NB:m, 0:NB)`, `U = A(i:i+NB, i+NB:n)`. Restore
`A(j,j)=d(j)`, `A(j,j+1)=e(j)` for the panel. The final partial panel uses the
unblocked `dgebd2` (leto's existing per-reflector reduce, factored to write
`d`/`e` for the trailing submatrix).

## Verification plan
- `bidiagonalize`/`singular_values` reconstruction + nalgebra singular-value
  batteries at sizes straddling `BLOCK_MIN_DIM` (e.g. 192², 256²), plus the
  existing small cases (unblocked path, unchanged).
- A/B benchmark `singular_values` 256² blocked vs unblocked; keep the gate only if
  the blocked path wins (per the QR-blocking precedent).
- The reduction is implemented **only behind the verified oracle**: if the blocked
  `B` does not match the unblocked `B`'s singular values to the derived tolerance,
  the path is not shipped.

## Consequences
- Closes the REDUCE half of the `singular_values` gap; the SWEEP half (sequential
  Givens) remains, closeable later by `dqds`/`dbdsdc` (a separate ADR).
- Reuses the verified `tiled_gemm` (SSOT) for the BLAS-3 trailing update; the
  per-column `dlabrd` gemvs reuse `dot_slice`/`axpy_slice`.
- The full-`U`/`V` blocked path (forming the factors via the same reflectors) is a
  follow-up; values-only is the disparity driver.
