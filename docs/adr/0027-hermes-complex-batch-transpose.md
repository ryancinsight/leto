# ADR 0027: Hermes complex matrix-batch transpose

- Status: Accepted
- Date: 2026-09-01
- Class: [minor] [arch]

Revision 2026-09-06: [LETO-SQUARE-TRANSPOSE](../../backlog.md#leto-square-transpose)
extends the movement boundary to an in-place square and a checked core dense
copy. The iterator-based tile-span experiment is rejected on executable size
and AVX-512 spills; the dense-copy revision remains under verification. Batch
measurements below do not establish a square-transform speedup.

## Context

Apollo's retained 3-D fast Fourier transform (FFT) executes layout phases over
hundreds of adjacent small complex matrices. Its pinned phase probe showed
that register-resident square transposes reduce this workload, while applying
the same route broadly regressed large rectangular 2-D matrices by 5-52%.

Apollo ADR 0040 assigns layout movement to Leto. Leto core owns layout, storage
and the generic copies its views require; SIMD kernels belong in `leto-ops`,
which already depends on Hermes. An Apollo-local kernel or a new Hermes
dependency in Leto core would violate those boundaries.

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

All other shapes and unsupported exact widths use Leto's existing generic
tiled mover through `transpose_copy`. Both paths preserve matrix order and
allocate no storage for successful complex copies.

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
their first and repeated calls across all four scalar types. The extent
revision (`ce9d02b`) passes the recorded provider gates and emits one scalar
tail, allocation-free dimension errors and register-resident tile pairs.
Its 6,868,992-byte executable remains 7,168 bytes above baseline and is rejected
on size; the counterbalanced full-engine comparison supplies no acceptance
basis. Evidence remains in Atlas's `output/apollo-square-transpose`, with the
fourth codegen inspection under `extent-contract/codegen.md`.

### Cache-blocking revision, 2026-09-06

The fourth candidate's phase diagnostic attributes 546–567 microseconds per
call to the final transpose at double-precision length 262,144 on the selected
performance core (`phase-profile.txt`). Those phase envelopes include the
operation's work; they are not cache-miss measurements or a baseline comparison.
The source traversal nevertheless establishes a locality difference: its
two-element AVX2 tile consumes half of a 64-byte line in each lower row, then
sweeps the remaining matrix width before revisiting adjacent samples. For an
aligned 512-square, the early strip touches approximately
`510 * 64 + 2 * 510 * 16 = 48,960` bytes between those uses.

The next experiment restores Apollo baseline `9da1f9f7`'s 16-by-16 outer
blocking while retaining the existing register-tile movement. Two blocks hold
`2 * 16^2 * size_of::<Complex<T>>()` payload bytes, at most 8 KiB across the
four supported scalars. This is a payload bound, not a promise of cache
residency; split lines, associativity and neighboring state still matter.
The fixed outer block adds no specialization dimension or runtime topology
probe. Register side 2, 4 or 8 divides 16, and clipping the final block to the
complete register square preserves tile alignment.

Diagonal register tiles run once in a separate traversal of `full_side / SIDE`
tiles. A single blocked strict-upper-triangle traversal handles every remaining
tile pair; it covers both diagonal cache blocks and off-diagonal cache-block
pairs without cloning their exchange body. These regions are disjoint, and
each pair still loads both source tiles before either destination write. The
unchanged scalar tail covers the complement of the complete register square.
No volume copy, allocation, error-contract change or workload change occurs.
Independent review finds no coverage or bounds defect. The unchanged 923
native tests, nine focused release cases, Clippy, minimal-feature compilation,
27 doctests and 24 benchmark smoke cases pass. All 15 captured source hashes
remain unchanged through the run. Evidence is retained under Atlas
`output/apollo-square-transpose/cache-blocking/leto-gates`. Matching assembly
keeps complete tile payloads in registers, while traversal state grows. The
6,871,552-byte executable exceeds baseline by 9,728 bytes. The unchanged
16-run census and independent raw-sample audit find one supported gain, the
efficiency-core 262,144-point real half-spectrum transform: paired medians
decrease 17.40–35.35%, with no supported regression. Warm allocation records
and retained bytes match; cold peaks overlap but are not identical. This is
local AVX2 evidence with endpoint-load observation limits, not cache-miss
attribution or physical AVX-512 coverage. Size still rejects adoption.

Apollo next consolidates its remaining pure-copy FourStep passes on the
existing `transpose_complex_matrices` API and deletes its private copy
kernels and scalar trait hook. This consumer experiment changes no provider
threshold or algorithm. It must retain the gain and satisfy the same memory
and executable-size bounds; source deletion alone does not establish that.

That direct batch-API consumer candidate grows the executable to 6,892,544
bytes, 30,720 above baseline. Apollo's full correctness and allocation gates
pass, but its census reports two performance-core regressions. Assembly
retains a 1,476-instruction generic layout/assignment body and 354 cleanup
instructions around a 370-instruction canonical mover; these compilation-unit
counts do not fully attribute linked size. A checked public dense-operation
boundary over the existing canonical mover is the next provider design under
review. It preserves the public batch contract and avoids duplicating movement
or suppressing invariant diagnostics.

### Register tile span revision, 2026-09-06

The cache-block candidate retains eight AVX2 and sixteen AVX-512 tile bounds
branches in the inspected kernels, alongside the 9,728-byte executable growth.
These are code-generation findings, not measured bounds-check latency. The
next provider experiment changes only the shared register tile's row access.
It checks `stride >= SIDE` and the tile span's arithmetic, clips that span
once, then splits `(SIDE - 1) * stride` elements from the final `SIDE` elements.
The prefix contains exactly `SIDE - 1` complete stride chunks and no remainder.
Both the prefix's row width and the final row therefore cover one register.

Loads seed the existing exact register array from the final row and fill its
preceding entries from those chunks. Stores use the same disjoint span split;
stride padding remains untouched. The existing tile permutation and callers'
load-both-before-store ordering do not change. This introduces no unsafe code,
signature, scalar or ISA variant, allocation, dispatch or workload change.
The bounded hypothesis is fewer repeated row-offset calculations and extent
checks. Emitted division, outlined helpers, payload spills, larger frames or
text growth reject this form; unchanged value and allocation suites still
precede downstream performance acceptance.

Provider validation passes: 923 native tests, nine focused tests in both debug
and release, Clippy, minimal features, 27 doctests (one existing ignored),
warning-denied rustdoc and the unchanged 24-case bounded smoke. All fifteen
source hashes match; tile SHA256 is
`A06AE5B7BF4AF37DB56797A56D52AB6BF56063C88206DE574F705B994243F45B`.
The retained evidence is
`output/apollo-square-transpose/tile-span/leto-gates/final-checks.json`
under Atlas output retention. These checks establish behavior separately
from consumer codegen and performance acceptance.

Consumer codegen rejects the tile-span form. Against the pure-copy candidate,
the executable grows by 9,728 bytes, including 7,120 text bytes. AVX2 improves
from 303 to 270 instructions and from a 248-byte to a 216-byte frame, without
division or payload spills. AVX-512 instead introduces runtime division,
register payload staging and a 1,016-byte frame, up from 472 bytes. That fails
the existing all-ISA spill and size stop conditions. The next source revision
therefore restores the canonical per-row tile access from `6013768`; it adds
no ISA-specific alternative. The checked dense-copy change below targets a
separate, independently observed assignment cost.

### Checked dense-copy boundary, 2026-09-06

The pure-copy consumer build emits 6,892,544 bytes, 30,720 above the ISA
baseline. Its library assembly contains a 1,476-instruction generic assignment
function and twelve cleanup funclets totaling 354 instructions, while the
canonical dense mover contains 370 instructions. These counts identify the
retained layout/view round trip, not exact linked-byte ownership or latency;
the evidence is `output/apollo-square-transpose/pure-copy/codegen.md`.

Expose that mover in place as the additive core operation
`transpose_copy<T: Clone>(source, destination, rows, columns) -> Result<()>`.
It copies row-major `[rows, columns]` into row-major `[columns, rows]`, with
destination offset `column * rows + row` receiving a clone of source offset
`row * columns + column`. The two internal callers, assignment and view
materialization, use the same body. The old private name and the complex
batch's general-view construction are removed; no forwarding API or second
movement algorithm remains. No dependency or version change is needed.

The scalar-independent preflight checks the unsigned product, exact source
length, exact destination length, then the signed dense-layout extent, in
that order. Zero products with empty slices succeed without converting their
unused dimensions. Nonempty zero-sized slices still require a signed count
bound; element size alone cannot establish it. Validation errors execute no
clones or writes. Successful movement invokes Clone once per element, including
zero-sized elements; user Clone or Drop implementations may allocate or panic,
so failure atomicity does not promise rollback after an element panic.

Complex batches keep their original matrix/batch overflow and role-specific
length diagnostics and precedence. Their admitted scalar representations
are nonzero-sized, so exact nonempty safe slices already imply each matrix
fits the signed layout extent. The direct core call therefore introduces no
reachable batch failure or changed hardware threshold. Measured eager error
sites in the batch products and traversed core layout arithmetic construct
the same errors lazily; no error is discarded or replaced by a fallback.

Clippy classifies the static Overflow payloads as unnecessary lazy evaluation,
but retained assembly shows successful arithmetic constructing and calling
the broad enum's drop glue: batch products at lines 33200/33206, shape and
stride products at 33267/33273 and 33320/33326, physical offset arithmetic
at 428769–428797, min/max bounds at 428578–428630, and dense destination
extent at 31481. The Overflow arm returns without heap deallocation; the
observed cost is construction, call and discriminant dispatch. Statement-level
lint expectations preserve lazy construction at these sites. The new dense
extent check uses the same enum ownership mechanism, but has no pre-change
instance in that assembly; its own codegen remains to be measured. No
function-wide or crate-wide lint exception is introduced.

Existing numerical workloads remain unchanged. New core tests cover both
traversal orientations, tile boundaries, offsets and canaries, non-Copy Clone
values and clone counts, exact error precedence, empty dimensions and huge
zero-sized extents. Provider debug/release, allocation, documentation and
SemVer gates precede fresh consumer codegen, size and unchanged census checks.
Neither source deletion nor instruction counts establish an accepted speedup.

The final dense-copy source passes 930 native tests, 366 release tests,
28 doctests (one existing ignored), all-target Clippy, minimal features,
warning-denied rustdoc and 24 unchanged bounded smoke cases. Both packages
pass 196 SemVer checks against `a2006ad`, with 58 inapplicable checks each.
Independent source review finds no production defect. Exact source hashes,
commands and limitations are retained in
`output/apollo-square-transpose/dense-copy/leto-gates/final-checks.json`.
Consumer codegen, executable size and timing remain separate acceptance gates.

### Preflight visibility revision, 2026-09-06

The dense-copy consumer emits 6,872,576 bytes, 10,752 above the ISA baseline.
Its generic assignment and cleanup family is gone, and no ordinary successful
path retains the prior eager error drops. The count-one batch body nevertheless
divides both slice lengths by the checked matrix product: its opaque length
validator hides their equality. Core extent preflight also remains external.
The emitted evidence is `output/apollo-square-transpose/dense-copy/codegen.md`.

The next bounded experiment adds `#[inline]` only to the existing
`validate_length` and `transpose_extent` helpers. All checks, their order,
error values, traversal bodies and workloads remain unchanged. The hypothesis
is removal of chunk-count division and redundant batch/preflight bookkeeping.
Retained division, new payload spills, duplicated error formatting or text
and executable growth reject this form. No count-one specialization, new
error helper, algorithm copy or recovery of the entire size residual is
assumed; codegen and size checks precede a performance claim.

The two-annotation revision passes format, all-target Clippy and the unchanged
30 focused tests in both debug and release. Independent source review finds
no contract change. Its fourteen source hashes match the retained
`output/apollo-square-transpose/preflight-inline/final-checks.json` record;
the preceding dense-copy full-suite and API coverage remains applicable.

### Tile bounds diagnostic revision, 2026-09-06

The preflight-inline executable remains 10,752 bytes above the ISA baseline.
Its linked map retains separate Apollo-library and census copies of the
complex square AVX2 and AVX-512 kernels. Independent byte comparison finds
identical instructions after normalizing relocations, including identical
call targets. Distinct panic-location records reference the same tile-row
indexing site; AVX-512 also retains equal shuffle constants at different
addresses. These findings do not establish ThinLTO import as the cause.
The map and byte evidence live under
`output/apollo-square-transpose/preflight-inline/map/`.

The bounded source experiment replaces only the shared tile loader/store's
range indexing with checked slice extraction. Both failures enter one
non-generic cold function that owns its panic location and reports start,
width and matrix length without computing a new extent. It deliberately
does not forward the caller location. Public shape validation and typed
errors, seeded register arrays, dispatch, traversal and workloads are unchanged.
Search of the layout and core operation families found no existing diagnostic
with this tile-row contract.

The hypothesis is removal of per-instantiation panic-location references,
allowing otherwise equal linked kernels to fold. Equal shuffle constants may
still prevent folding, so neither deduplication nor a size reduction is
assumed. Unchanged focused debug/release and allocation tests precede consumer
codegen: new division or payload spills, retained duplicated kernels, or text
growth reject this form. Full consumer regression and baseline executable
no-growth acceptance remain unchanged. Commands, hashes and source-gate
results belong under `output/apollo-square-transpose/tile-diagnostics/`.

The source revision passes format, all-target Clippy and the unchanged 30
focused tests in both debug and release, including allocation checks.
Independent source review finds no defect. Fourteen tracked source hashes
remain unchanged during the gates; the final record separately identifies
the subsequent documentation-only result update and lease release. These
results establish behavior on the tested paths, not linked-code reduction.

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
