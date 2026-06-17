# ADR 0010: Blocked-reflector (compact-WY) vectorization for eig/SVD

- Status: Accepted (phased; Phase 0 done, Phases 1–3 planned)
- Date: 2026-06-16
- Class: [major] (post-1.0 the trailing-update path changes observable timing and
  the internal reduction/iteration kernels are restructured; the public surface
  and numerics are preserved, verified by the existing differential batteries)

## Context

After the apply-window confinement (ADR-adjacent, `gap_audit.md`) and the
`axpy_slice` vectorization of the contiguous reflector sweeps, the dense
non-symmetric eigensolver and the SVD retain a residual disparity to nalgebra
(LAPACK): `eig` ≈ 4.4×, full `svd` ≈ 3.4×, `singular_values` ≈ 2.3× at 64×64.

The residual is **structural, not a tuning gap**. Both algorithms spend their
dominant time applying *single* Householder reflectors of width 2–3:

- **eig** — the Francis bulge chase calls `apply_right` (`H ← H P`) row by row;
  each row touches only `len ∈ {2,3}` contiguous columns. The long dimension is
  the row index, which is strided by `n`, so no contiguous SIMD span exists. The
  `apply_left` direction was vectorized (contiguous column span), leaving the
  right-apply as the bottleneck.
- **SVD** — the Golub–Kahan bidiagonalization applies a sequence of single
  reflectors to the trailing matrix; the implicit-shift bidiagonal QR sweep
  applies a sequence of *single* `2×2` Givens rotations down strided columns. A
  single rotation has no width to vectorize.

A single narrow reflector/rotation is intrinsically memory-latency-bound: `O(n)`
work touching `O(n)` strided cache lines with `O(1)` arithmetic intensity. The
roofline ceiling is bandwidth, and SIMD cannot raise it for a width-3 operator.

LAPACK does **not** apply reflectors one at a time on these paths. It aggregates
`nb` consecutive reflectors into a **block reflector** and applies the block as a
matrix–matrix product (GEMM), which is compute-bound and routes through the tuned
`tiled_gemm` SIMD micro-kernel leto already owns. This raises the arithmetic
intensity from `O(1)` to `O(nb)` and is the standard route to BLAS-3 performance.

## Theorem (compact-WY representation, Schreiber–Van Loan 1989)

Let `Q = H₁ H₂ … H_r` be a product of `r` Householder reflectors,
`H_i = I − β_i v_i v_iᵀ`, with the `v_i` stored as the columns of `V ∈ ℝ^{m×r}`
(`v_i` zero above its pivot). Then there is an upper-triangular `T ∈ ℝ^{r×r}` with

    Q = I − V T Vᵀ.

`T` is built columnwise by the recurrence `T_{11} = β_1` and, for `i = 2…r`,

    T_{1:i-1, i} = −β_i · T_{1:i-1,1:i-1} · (V_{:,1:i-1}ᵀ v_i),   T_{ii} = β_i.

*Proof (induction on `r`).* For `r = 1`, `Q = I − β_1 v_1 v_1ᵀ = I − V T Vᵀ` with
`V = v_1`, `T = β_1`. Assume `Q_{r-1} = H_1…H_{r-1} = I − V_{r-1} T_{r-1} V_{r-1}ᵀ`.
Then `Q_r = Q_{r-1} H_r = (I − V_{r-1}T_{r-1}V_{r-1}ᵀ)(I − β_r v_r v_rᵀ)`. Expanding,

    Q_r = I − V_{r-1}T_{r-1}V_{r-1}ᵀ − β_r v_r v_rᵀ
            + β_r V_{r-1}T_{r-1}(V_{r-1}ᵀ v_r) v_rᵀ.

Set `V_r = [V_{r-1} | v_r]` and
`T_r = [[T_{r-1}, −β_r T_{r-1}(V_{r-1}ᵀ v_r)], [0, β_r]]`. Block-multiplying
`V_r T_r V_rᵀ` reproduces exactly the four terms above, so
`Q_r = I − V_r T_r V_rᵀ`, and `T_r` is upper-triangular because `T_{r-1}` is. ∎

### Corollary (BLAS-3 application, the performance lever)
Applying the block to a trailing matrix `C` is three GEMMs and is numerically a
sequence of orthogonal similarities (spectrum/​singular-value preserving):

    Qᵀ C = C − V (Tᵀ (Vᵀ C)),        C Q = C − ((C V) T) Vᵀ.

`Vᵀ C` (or `C V`) is `r×(cols)` — a GEMM; the small `T` product is `r×r·…`; the
rank-`r` correction is the final GEMM. Each routes through `Scalar::tiled_gemm`.
The forward error is the standard blocked-orthogonal bound `O(r·ε‖C‖)`, within the
backward-error tolerance the eig/SVD batteries already assert (see ADR note on the
`8·√(ε‖A‖)` defective-eigenvalue bound).

## Decision

Introduce a compact-WY block-reflector seam and route the three reduction/iteration
kernels through it, phased so each phase is independently verified against the
current (correct) unblocked path before the next begins. No public API or numeric
contract changes; the differential batteries are the gate.

### Deep vertical file hierarchy (new leaf module)

```
linalg/
  reflector_block/
    mod.rs       BlockReflector<T> (V panel view + T factor) + the compact-WY
                 theorem/proof; apply_left_block / apply_right_block via tiled_gemm
    accumulate.rs  build T columnwise from a panel of single reflectors (the
                   Schreiber–Van Loan recurrence) — SSOT for T construction
    panel.rs       zero-copy column-panel view over row-major storage (Cow where a
                   non-contiguous panel must be materialized; borrow otherwise)
```

- **Monomorphization / zero-cost:** `BlockReflector<T, const NB: usize>` — the
  block width is a const generic so the panel buffers are fixed-size and the GEMM
  shapes are known at compile time; one generic kernel monomorphizes per `(T, NB)`.
- **ZST / typestate:** a `Side` ZST (`Left`/`Right`) selects the apply form at
  compile time (DCE), avoiding a runtime branch in the hot trailing update.
- **DIP / SSOT:** the block apply depends only on `Scalar::tiled_gemm` (the
  existing backend seam), not on any concrete SIMD type; `accumulate.rs` is the
  single `T`-construction site reused by every consumer.
- **Zero-copy / CoW:** `panel.rs` borrows the reflector columns in place when the
  storage is contiguous; it materializes only a non-contiguous panel.

### Phases

- **Phase 0 (done):** contiguous single-reflector sweeps vectorized via
  `axpy_slice` (Householder apply + Francis left-apply); eig 5.9×→4.4×, svd
  4.1×→3.4×. Establishes the SIMD-apply baseline and the backward-error test
  tolerance that admits blocked reorderings.
- **Phase 1 — block reflector primitive:** `reflector_block/` with the compact-WY
  accumulation and `tiled_gemm`-based block apply. Verified in isolation:
  differential equality (within `O(r·ε‖C‖)`) of `apply_*_block` against `r`
  sequential single-reflector applies, over random panels and sizes. Wired into
  **blocked Hessenberg** (`dlahr2`-style panel reduction + GEMM trailing update),
  its first consumer, verified against the existing unblocked Hessenberg contract
  (Q orthogonality, `A = QHQᵀ`, trace/Frobenius invariants).
- **Phase 2 — blocked bidiagonalization (`dlabrd`-style):** two-sided panel
  reduction feeding the SVD; verified against the unblocked `bidiagonalize`
  (`A = U B Vᵀ`, bidiagonal structure, U/V orthogonality) and the SVD batteries.
- **Phase 3 — small-bulge multishift QR (eig) / blocked bidiagonal QR (SVD):**
  chase `nb`-wide bulges so the bulge-application trailing updates become GEMMs
  (LAPACK `dlaqr5`/`dbdsqr`-blocked). Highest value (the iteration is the dominant
  cost) and highest risk (index management); gated on the hardened
  8×8/16×16/defective batteries with the backward-error tolerance.

## Rejected alternatives

- **Strided SIMD of the single reflector apply.** A width-3 right-apply has no
  contiguous span; gather/scatter SIMD over the strided row dimension does not beat
  the scalar loop (verified: `axpy_rows` with `cols = 3` is bandwidth-bound). Does
  not change the roofline ceiling.
- **Transpose `H` for the right-apply.** Per-step transposition is `O(n²)` copy
  against `O(n²)` apply work — no asymptotic gain and double the memory traffic.
- **Forcing the existing within-block apply narrowing without blocking.** Closes a
  sub-2× constant only, and (ADR #20, machine-checked) perturbs defective
  eigenvalues; superseded by blocking, which is both correct and BLAS-3.

## Consequences

- A reusable compact-WY seam (`reflector_block/`) that every two-sided/one-sided
  reduction and bulge-chasing iteration shares — SSOT for blocked orthogonal
  updates, the same role `tiled_gemm` plays for products.
- Phases 1–3 are large and index-intensive (the historical #1 bug source in QR
  iterations); they are sequenced and individually differential-verified rather
  than landed as one drop, honoring correctness-over-breadth. Each phase is a
  separate WIP-limited item with the unblocked path as its oracle.
- Expected outcome: the trailing updates become compute-bound GEMMs, closing the
  bulk of the residual eig/SVD disparity; the Givens bidiagonal-QR sweep (Phase 3,
  SVD) is the last latency-bound holdout and may retain a constant factor.
