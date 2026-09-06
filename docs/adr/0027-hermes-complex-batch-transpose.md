# ADR 0027: Hermes complex matrix-batch transpose

- Status: Accepted
- Date: 2026-09-01
- Class: [minor, perf]

Revision 2026-09-06: [LETO-SQUARE-TRANSPOSE](../../backlog.md#leto-square-transpose)
extends the existing movement boundary to an in-place square. Provider gates
pass; downstream timing remains under verification. No square-transform speedup is
established by the earlier batch measurements below.

## Context

Apollo's retained 3-D fast Fourier transform (FFT) executes layout phases over
hundreds of adjacent small complex matrices. Its pinned phase probe showed
that register-resident square transposes reduce this workload, while applying
the same route broadly regressed large rectangular 2-D matrices by 5-52%.

Apollo ADR 0040 assigns layout movement to Leto. Leto core also deliberately
owns only layout and storage types; SIMD kernels belong in `leto-ops`, which
already depends on Hermes. An Apollo-local kernel or a new Hermes dependency in
Leto core would violate those boundaries.

## Decision

Add `leto_ops::transpose_complex_matrices`, a scalar-generic operation over
borrowed `Complex<T>` slices and caller-owned output. It validates checked
matrix and batch lengths plus both exact slice lengths before mutation. Empty
batches and zero-sized matrices are no-ops.

For at least 256 matrices with both sides at most 16, the operation requests
exact Hermes scalar widths 16, 8, then 4. The widest available width whose
complex register fits a complete square tile is selected once at the operation
boundary. Each kernel loads a square tile into registers, transposes it with
`ComplexReg::transpose_square`, stores it once, and copies every ragged row or
column tail. No scalar fallback is classified as SIMD capability.

All other shapes and unsupported exact widths retain Leto's existing generic
tiled assignment. Both paths preserve matrix order and allocate no storage.

Rejected alternatives:

- Keep the register kernel in Apollo. Rejected because layout movement is a
  Leto responsibility and a second implementation would duplicate the
  provider contract.
- Add Hermes to Leto core. Rejected because it would collapse the established
  storage/compute dependency boundary.
- Route every shape through register tiles. Rejected because the Apollo phase
  probe measured 5-52% regressions for large rectangular 2-D matrices.
- Request only the host's widest width. Rejected because an AVX-512 host may
  need an exact AVX2-sized 4x4 tile; exact-width descent preserves that route.

## In-place square movement

Apollo's FourStep final permutation currently exchanges scalar pairs. The
operation contains no FFT arithmetic, and its square shape permits in-place
tile exchange without an additional matrix buffer. The provider surface is
`transpose_square_inplace(&mut [Complex<T>], side)` with `T: LaneScalar + Pod`.
It checks `side * side` and exact storage length before mutation; empty storage
with side zero is valid. Overflow and length errors preserve the entire input.
`SquareTransposeError` retains `side` on overflow and `side`, `expected` and
`actual` on length mismatch. Its Copy/Eq variants contain only dimensions;
success and rejection allocate no storage. The operation does not inherit
unrelated solver diagnostics from `LetoError`.

For element coordinates `(r, c)`, the permutation exchanges offsets
`r * side + c` and `c * side + r`. Diagonal tiles transpose internally;
off-diagonal tile pairs load both sources before either destination is written.
Every position outside the complete tiled square is exchanged once above the
diagonal. Unsupported hardware widths use the same pairwise permutation.
No arithmetic touches the scalar payload, so all bits, including signed zeros,
subnormals and NaN payloads, must survive exactly. Reduced-precision hardware
frames do not imply native register shuffles on every backend.

Batch and square traversals share one complex register load/transpose/store
leaf. Hermes capability-carrying views provide safe exact-width chunks over
Eunomia's borrowed complex/scalar layout casts. The existing batch thresholds
remain unchanged; they do not supply evidence for square dispatch. Only the
actual register side specializes the tile arrays. No allocation, additional
scratch, dynamic dispatch or new dependency is introduced.

The working hypothesis is reduced strided scalar movement in the final
FourStep permutation. Two tiles must remain register-resident for the expected
benefit; emitted code checks spills and bounds checks before an unchanged
complete-engine census evaluates latency, allocation and executable size.
The experiment is rejected on supported regression, absent supported benefit
or executable growth. A provider microbenchmark alone cannot establish an
Apollo speedup. Keeping a second Apollo implementation, allocating a matrix,
and widening reduced-precision payloads are rejected on ownership, memory and
bit-preservation grounds respectively.

Tests compare the coordinate permutation and entire byte representation across
`f32`, `f64`, `F16` and `Bf16`, including offsets, ragged sides, special payloads,
invalid lengths and overflow. The entry baseline at `a2006ad` passes all five
existing focused movement tests under the committed Nextest budget. Debug and
release suites, the allocation observer, documentation and SemVer checks cover
the provider change; Apollo retains independent FFT and full-engine checks.

The 2026-09-06 provider diff passes 923 native tests, nine focused release
tests, 27 doctests (one existing ignored), minimal-feature compilation,
warning-denied Clippy, rustdoc and all 196 applicable minor SemVer checks
against `a2006ad`. The unchanged layout benchmark smoke passes 24 cases under
the 60-second supervisor. First and repeated successful square submissions
record zero calling-thread allocations/reallocations across all four scalars.
Evidence and source hashes are retained in Atlas's
`output/leto-square-transpose` under its 14-day/10-GiB policy. Runtime tests
cover the selected host backend and the explicit scalar path, not every ISA;
register residency, whole-engine latency and executable size remain separate
downstream acceptance checks.

The first Apollo build with provider `9672ddc` grows by 14,336 executable
bytes and is not accepted. Assembly `914754C9...2225269433` shows an AVX2
tile pair in registers, but AVX-512 outlines `array::from_fn` construction
and inherits Hermes's generic row-buffer permutation. The latter emits a
1,272-byte helper frame; source-level register types do not establish register
residency. Successful extent validation also destroys an eagerly constructed
`LetoError`. The next bounded experiment seeds an exact Copy array with its
first loaded row, fills the remaining rows in the kernel and constructs an
overflow error only on failure. It preserves the same permutation, extents,
hardware selection and tests. Any remaining permutation capability gap belongs
in Hermes. Timing remains unmeasured until the size/code-generation condition
is met; the initial artifacts are retained under
`output/apollo-square-transpose/array-construction`.

The seeded-array build (`07bd618`) removes the constructor helper and eager
error destruction, shrinking the executable by 1,536 bytes, but remains 12,800
bytes above baseline. Its AVX2 fill also emits redundant Skip-iterator control
flow and tile spills. The next revision iterates the structurally exact tail
slice and evaluates the [Hermes forwarding correction](../../../hermes/backlog.md#hermes-complex-permutation-inlining),
which is intended to keep the existing permutation in the proven target-feature frame.
This does not introduce a Leto-local native backend or weaken the acceptance
conditions. The second candidate is retained under
`output/apollo-square-transpose/seeded-array`.

The combined Leto `00fd88e` / Hermes `07c5e5f` candidate emits 6,873,600
executable bytes: 11,776 above baseline and 1,024 below the seeded-array build.
Assembly `CD76ED04...A8B57BA` removes the outlined array constructor and
permutation helper. Its emitted double-precision AVX2 and AVX-512 complete
tile pairs remain in registers; remaining slice bounds branches persist.
The 200-byte AVX2 and 488-byte AVX-512 frames include scalar remainder code,
which is duplicated with the public entry's scalar fallback. No batch or
single-precision square symbols occur in this library artifact, so it cannot
attribute the entire executable delta. The comparison remains rejected on
size, with no timing claim; detailed instruction counts and locators are in
Atlas's `output/apollo-square-transpose/feature-frame/codegen.md`.

The next source revision returns the completed square side from each hardware
kernel, then applies one scalar remainder traversal at the operation boundary
only when that side is smaller than the matrix side. Unsupported hardware
returns zero and reaches the same traversal. The full square and its border
are disjoint, so this changes code placement without changing permutation
coverage. The narrow error contract also removes formatted error allocation
and unrelated diagnostic dependencies. Existing successful workloads remain
unchanged; invalid short, long and overflowing submissions additionally check
exact error dimensions, unchanged bytes and zero allocations/reallocations on
their first and repeated calls across all four scalar types. These revisions
await their own gates and downstream codegen/size comparison.

## Established batch evidence

This is one additive public function in `leto-ops`; existing assignment APIs
and behavior do not change. Full, ragged, asymmetric, empty, invalid-length,
and overflow cases carry value-semantic coverage for `f32` and `f64`.
Validation is failure-atomic, and a warmed allocator census records zero
allocations and zero reallocations.

Two independently launched same-binary Criterion runs on the local Windows
AVX2 workstation place provider median reductions at 86.7-88.8% (`f32`) and
88.9-89.8% (`f64`) for 1,024 batches of 4x4 matrices, and 28.3-53.3% (`f32`)
and 26.1-30.5% (`f64`) for 256 batches of 16x16 matrices. Every provider/control
95% confidence-interval pair in the second run is disjoint. The control is
Leto's unchanged generic assignment in the same benchmark binary. These
measurements establish only the selected local layout regime; Apollo must
independently verify full FFT values, allocation behavior, and throughput.
