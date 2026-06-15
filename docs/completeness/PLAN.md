# Leto Completeness Plan: Full ndarray / nalgebra Parity

Status: active. Owner: leto workspace. Created 2026-06-14.
Reference targets: `ndarray 0.16`, `nalgebra 0.35` (workspace-pinned dev oracles).
Companion artifacts: [`parity_matrix.md`](parity_matrix.md) (the scored inventory),
`gap_audit.md` (prior consumer-driven audit), `benchmark_results.md` (perf baselines).

## 1. Objective and scope decision

Target: **literal full functional parity** — Leto exposes, for every public
operation in `ndarray` and `nalgebra`, an equivalent generic Leto API verified
for value-semantic correctness against the oracle and measured for performance
against it.

This **overrides** the prior consumer-driven policy in `gap_audit.md`
("a routine enters leto-ops only with a named Atlas consumer driver") and the
recorded non-goals (sparse formats, `Matrix3`/`Vector3` small-fixed types,
`IxDyn` dynamic rank, deferred operator overloading, full rank-revealing SVD).
Those exclusions are **reopened** as parity targets. Two ADRs are re-decided as
part of this plan rather than treated as closed:

- ADR 0001 (operator overloading deferred by orphan rule) → revisit: parity
  requires `Add/Sub/Mul/Div/Neg` on Leto array/view types. The orphan-rule
  constraint is real, so the decision is *how* (newtype receiver, owned-vs-view
  matrix of impls), not *whether*.
- ADR 0002 (const-rank only; `IxDyn` excluded) → revisit: full ndarray parity
  includes a dynamic-rank escape type. Decision needed: a Leto-owned `IxDyn`
  analogue vs. documented permanent exclusion with a const-rank dispatch story.

Scope boundary that remains: parity is measured against the **array + dense
linear-algebra** surface of these two crates. Items belonging to *other* Atlas
crates by the layer boundary (autodiff, NN kernels, GPU buffers — Coeus; FFT —
Apollo) are out of leto's parity surface and are not ndarray/nalgebra surface
either. Where ndarray companion crates are involved (`ndarray-linalg`,
`ndarray-rand`, `ndarray-stats`), they are tracked as separate columns because
they are the realistic parity bar for "what ndarray users do".

## 2. Definition of "complete"

A parity-matrix row is **Complete** when all three hold:

1. **Surface**: a generic Leto API (`<T: Scalar>` / `<T: RealScalar>` / const
   generics) covers the operation; no type-suffixed or per-rank clones.
2. **Correctness**: a differential test asserts value-semantic parity against
   the oracle (bounded epsilon derived per `numerical_discipline`, not tuned),
   over positive, boundary, and adversarial inputs.
3. **Performance**: a criterion comparison against the oracle is recorded. A row
   is Complete even if slower, but the gap is logged; a row may not *claim
   performance parity* until within the recorded regression band.

Completeness of the **whole** is the fraction of in-scope oracle operations at
`Complete`, reported per family in `parity_matrix.md`, plus an explicit
`Excluded` set each carrying a one-line rationale (the only sanctioned way a row
leaves the parity denominator).

## 3. Method to determine completeness

Five stages, each producing a durable artifact. Stages 1–2 are read-only audit
and parallelize across subagents; stages 3–5 are tree-mutating and obey the
one-in-flight WIP limit.

### Stage 1 — Oracle API inventory (read-only)
Enumerate the public surface of each oracle from its actual locked source /
`cargo doc`, never from memory (anti-hallucination):
- `ndarray 0.16`: `ArrayBase` inherent methods, constructors, `Zip`, iterators,
  slicing, `linalg` (`dot`), broadcasting, `stack`/`concatenate`, numeric trait
  reductions; plus companion crates `ndarray-rand`, `ndarray-stats`,
  `ndarray-linalg` as flagged columns.
- `nalgebra 0.35`: `Matrix`/`DMatrix`/`DVector` operations, the decomposition
  module (LU, QR, Cholesky, SVD, symmetric/`Schur`/`Hessenberg` eigen,
  `Bidiagonal`, `ColPivQR`, `UDU`, `FullPivLU`), norms, BLAS-like ops,
  geometry/transform types, slicing/view ops.
Output: raw operation lists committed under `docs/completeness/inventory/`.
Group items into families (constructors, elementwise, reductions, linalg, …).

### Stage 2 — Cross-reference and scoring (read-only)
For each oracle operation, locate the Leto counterpart (search public exports of
`leto`, `leto-ops`) and assign a status: `Complete` / `Verified` (surface+test,
no perf row) / `Partial` (subset of behavior) / `Missing` / `Excluded(reason)`.
Record consumer/driver column for prioritization but it no longer gates entry.
Output: the scored `parity_matrix.md` (this is the headline completeness number).

### Stage 3 — Differential correctness harness (this plan delivers the scaffold)
A test per matrix row asserting value parity against the oracle. Already
present:
- `crates/leto-ops/tests/ops/oracle_parity.rs` — dense linalg vs nalgebra
  (LU/det/inv, symmetric eigenvalues, Cholesky, singular values) + reverse
  reductions vs ndarray.
- `crates/leto-ops/tests/ops/parity.rs` — **new**: elementwise (add/sub/mul/div,
  scalar, unary exp/sqrt), reductions (sum-all, sum/mean axis, cumsum),
  matmul/transposed/batched/dot, structure (concat/stack), least-squares, and
  reverse-axis reduction, all vs ndarray/nalgebra.
Rule: a test exists only for surface Leto implements; `Missing` rows live in the
matrix, never as ignored stubs (no test-gaming).

### Stage 4 — Performance comparison harness (this plan extends the scaffold)
Criterion benchmarks running Leto and the oracle on identical pinned inputs.
Already present in `crates/leto-ops/benches/kernels.rs`:
- `bench_oracle_compare` — matmul (64/128/256²) vs ndarray+nalgebra; reverse
  reductions vs ndarray.
- `bench_parity_oracle` — **new**: elementwise add, unary exp, sum, vector dot
  (64k) vs ndarray.
Each new `Missing` operation closed in Stage 5 adds its oracle benchmark here.

### Stage 5 — Gap closure (tree-mutating, WIP-limited)
Work the matrix in triage order: correctness gaps → architecture drift (ADR
0001/0002 re-decisions) → missing surface → performance gaps → docs. Each closed
row is one atomic commit: generic implementation + differential test + oracle
benchmark + matrix row flip + CHANGELOG entry.

## 4. First measurement pass (executed 2026-06-14)

Correctness — `cargo test -p leto-ops --test ops_tests --all-features parity::`
plus existing `oracle_parity::`: **18 new + 4 existing differential tests green.**
Leto matches ndarray/nalgebra value-for-value on every currently-implemented
family tested (elementwise, scalar, unary, reductions, cumsum, matmul/batched/
dot, concat/stack, least-squares, LU/Cholesky/eigen/SVD).

Performance — `cargo bench -p leto-ops --bench kernels --all-features` (median,
sample-size 10, AVX2-class Win11 x86_64):

| Family (64k f64) | Leto median | ndarray median | Ratio (Leto/oracle) |
| --- | --- | --- | --- |
| elementwise add | 18.9 µs | 13.3 µs | ~1.43× slower |
| unary exp | 638 µs | 761 µs | ~0.84× (faster) |
| sum | 3.53 µs | 4.53 µs | ~0.78× (faster) |
| vector dot | 7.06 µs | 9.35 µs | ~0.76× (faster) |

Dense matmul (from `benchmark_results.md`, unchanged this pass): Leto
17.4 µs / 109 µs / 1.06 ms at 64/128/256² vs ndarray 8.5 / 66.5 / 496 µs —
**~1.6–2.1× slower**, the standing open performance target.

Read: Leto is at or ahead of ndarray on bandwidth-bound reductions/dot/unary;
behind on elementwise add (allocation/dispatch overhead at the wrapper) and on
compute-bound dense matmul (no register micro-kernel). Correctness parity is
unconditional on the tested surface.

## 5. Known scope-expansion risks (to resolve during Stage 1–2)

- **Surface explosion**: nalgebra's geometry/transform types (`Isometry`,
  `Rotation`, `Quaternion`, `Perspective`) and small-fixed `MatrixN`/`VectorN`
  are large families with no array-substrate analogue. Decision per family in
  the matrix: implement, or `Excluded` with rationale (these are the most likely
  permanent exclusions even under "literal full parity", because they are a
  different abstraction than a strided array).
- **`IxDyn` (ADR 0002)** and **operator overloading (ADR 0001)** are the two
  architectural re-decisions that block whole matrix columns; sequence them
  early in Stage 5 because downstream rows depend on the outcome.
- **Companion-crate scope**: `ndarray-linalg` pulls a LAPACK comparison bar far
  beyond nalgebra's pure-Rust decompositions; treat as a distinct, lower-priority
  column.

## 6. Next actions

1. Stage 1 inventory extraction (subagent fan-out over the two oracles) →
   `docs/completeness/inventory/`.
2. Populate `parity_matrix.md` to full oracle coverage; compute the baseline
   completeness percentage per family.
3. Re-decide ADR 0001 / ADR 0002 (both now [arch] parity blockers).
4. Begin Stage 5 closure in triage order, one in-flight row at a time.
