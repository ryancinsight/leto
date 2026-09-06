# ADR 0027: Hermes complex matrix-batch transpose

- Status: Accepted
- Date: 2026-09-01
- Class: [major] [arch]

Revision 2026-09-06: [LETO-SQUARE-TRANSPOSE](../../backlog.md#leto-square-transpose)
extends the accepted batch-layout decision to in-place square movement and a
checked core dense copy. Acceptance of the original batch regime does not
accept the square-movement campaign. Every measured consumer candidate below
fails at least one unchanged adoption gate. The `843febc` all-operation provider
boundary is rejected on size; the batch-inline correction is underway and has
no acceptance result. No release or manifest version change is authorized.

## Ownership and current experiment

Apollo's 3-D FFT executes hundreds of adjacent small complex matrices. Its
phase probe supports register tiles for that regime, but routing large
rectangular 2-D matrices through them regresses by 5–52%. Apollo ADR 0040 assigns
layout movement to Leto: core owns layout, storage and generic view copies;
`leto-ops` owns SIMD movement through its existing Hermes dependency.

The provider-entry experiment introduces the scalar role `ComplexLayout` for
the two complex movement operations. The domain owns its contract and square
error. Four concrete implementations bind f32, f64, F16 and Bf16 to one private
generic checked body per operation. A generic default or blanket implementation
would preserve consumer instantiation roots; arithmetic scalar traits impose
unrelated operations, and moving the role to Hermes reverses layout ownership.
No new dependency, dynamic dispatch, scalar arithmetic or algorithm copy enters.

The initial experiment keeps both non-generic methods uninlined under ThinLTO.
Its linked map establishes one provider square entry and one square kernel per
used ISA, but the executable remains 4,608 bytes above baseline. The next bounded
correction retains `#[inline(never)]` on square methods and applies `#[inline]`
to batch methods, preserving the same role and generic bodies. FourStep passes
immediate count one; the all-operation anchor retains count-256 dispatch and
chunk-count division that its former specialization eliminated. Batch-plus-core
map intervals total 5,936 bytes before and after, so this is caller/callee
evidence for a specialization experiment, not the cause of whole-file growth.

Reject the batch-inline correction if count-one execution retains batch-count
dispatch/division, square kernels regain consumer instantiations, or the original
size and full-engine acceptance conditions fail. No new public entry, role split,
compiler flag, workload or expected byte recovery is assumed.

The batch-inline source passes format, Clippy, warning-denied rustdoc, 30
focused debug tests, 30 release tests and 25 doctests, with all 316 captured
build inputs fixed. Evidence is retained under Atlas
`output/apollo-square-transpose/batch-specialization/leto-gates`.
Consumer codegen, executable size and timing remain unverified for this change.

### Migration and classification

Import `leto_ops::ComplexLayout`. Generic callers replace free operations with
`T::transpose_complex_matrices(...)` and `T::transpose_square_inplace(...)`,
adding `T: ComplexLayout`; concrete callers use the scalar receiver, for example
`f32::transpose_square_inplace(...)`. Root `SquareTransposeError` remains the
same type, with its defining module moved to `domain::layout`. The former
`application::layout` module is private; import its error from the crate root.
The old public free functions are removed without a compatibility facade.

The new bound and removed paths are source-breaking even with the same four
admitted scalars, hence [major] [arch]. SemVer confirms those intended removals;
it does not verify behavior or performance. A Rust-source search across other
registered stack members finds callers only in Leto and Apollo. Their inputs
and timed regions remain unchanged; Leto's benchmark changes only its call.

## Movement contracts

### Complex batches

`ComplexLayout::transpose_complex_matrices` borrows `Complex<T>` input and
caller-owned output. Validation checks matrix product, batch product, exact
source length, then exact destination length before mutation, retaining the
role-specific overflow and length diagnostics. Empty batches and zero-sized
matrices are no-ops.

For at least 256 matrices with both sides at most 16, request exact Hermes
scalar widths 16, 8, then 4. Select the widest available width whose complex
register fits a complete square tile once at the operation boundary. Load,
transpose with `ComplexReg::transpose_square`, then store each register tile;
copy every ragged row/column tail. Unsupported exact widths and other shapes
use core `transpose_copy`. Scalar fallback is not classified as SIMD.
Both paths preserve matrix order and allocate no successful-copy storage.

Exact-width descent matters: an AVX-512 host can still require an AVX2-sized
4x4 tile. Selecting only its widest width, applying register tiles to every
shape, keeping another Apollo kernel, or adding Hermes to Leto core are rejected
respectively on coverage, measured regression, duplicate ownership and layering.

### In-place squares

`ComplexLayout::transpose_square_inplace(&mut [Complex<T>], side)` checks
`side * side`, then exact storage length before writing; side zero with empty
storage succeeds. `SquareTransposeError` is Copy/Eq and retains `side` on
overflow or `side`, `expected`, `actual` on length mismatch. Success and rejection
allocate nothing; errors preserve the whole input and do not import unrelated
solver diagnostics from `LetoError`.

For coordinates `(r, c)`, exchange offsets `r * side + c` and `c * side + r`.
Diagonal register tiles transpose once internally; each strict-upper tile pair
loads both sources before either store. Every position outside the complete
register square is exchanged once above the diagonal. Hardware kernels return
`full_side`; one shared scalar tail handles its complement when `full_side <
side`. Unsupported hardware returns zero and uses that same pairwise traversal.
This keeps the complete square and border disjoint without cloning the tail.

The 16-by-16 outer block follows Apollo baseline `9da1f9f7`. Two blocks contain
`2 * 16^2 * size_of::<Complex<T>>()` payload bytes, at most 8 KiB for the four
scalars. Register sides 2, 4 or 8 divide 16; clipping to the complete register
square preserves tile alignment. A separate `full_side / SIDE` diagonal pass
and one blocked strict-upper traversal cover diagonal and off-diagonal cache
blocks without cloning exchange logic. Payload footprint is not cache residency:
split lines, associativity and neighboring state still matter.

Batch and square share one register load/transpose/store leaf. Hermes
capability-carrying views provide safe exact-width chunks over Eunomia's borrowed
complex/scalar layout casts. Only register side specializes the tile arrays.
There is no extra matrix, scratch allocation or volume copy. Movement performs
no scalar arithmetic: signed zeros, subnormals and NaN payloads survive bitwise.
Reduced-precision hardware frames do not imply native shuffles on every backend.

The final-permutation hypothesis requires both tiles to remain in registers.
A second Apollo implementation, matrix allocation or payload widening would
violate ownership, memory or bit-preservation requirements. Batch thresholds
and batch timings do not establish square dispatch or FFT performance.

### Checked core dense copy

The additive core operation is
`transpose_copy<T: Clone>(source, destination, rows, columns) -> Result<()>`.
Row-major `[rows, columns]` becomes `[columns, rows]`: destination
`column * rows + row` receives a clone of source `row * columns + column`.
Assignment and view materialization use this same body. The old private name
and batch's general-view construction are removed, with no forwarding API,
second mover, dependency or version change.

Preflight checks unsigned product, exact source length, exact destination
length, then signed dense-layout extent. A zero product with empty slices
succeeds without converting unused dimensions. Nonempty zero-sized elements
still require the signed count bound. Validation performs no clones or writes;
success invokes Clone once per element, including zero-sized elements. User
Clone/Drop can allocate or panic, so validation atomicity does not promise
rollback after an element panic.

Complex batches retain their original error precedence and hardware threshold.
Their nonzero-sized representations and exact safe nonempty slices already
bound each matrix's signed extent; direct core use adds no reachable batch
failure. Measured batch/core layout arithmetic constructs the same errors
lazily, never discarding them or substituting a fallback.

Statement-level Clippy expectations retain lazy static Overflow construction
because prior successful arithmetic invokes broad-enum drop glue. Evidence in
`output/apollo-square-transpose/pure-copy/codegen.md`: batch products at
33200/33206; shape at 33267/33273; strides at 33320/33326; physical offsets at
428769–428797; min/max bounds at 428578–428630; dense destination extent at
31481. Overflow's drop arm returns without heap deallocation: the observed cost
is construction, call and discriminant dispatch. The new dense extent has no
pre-change instance there, only the same enum mechanism. No function/crate-wide
lint exception is introduced; later candidate codegen verifies its success path.

## Acceptance and verification

Retention requires supported complete-engine benefit, no supported regression,
no executable growth and unchanged allocation bounds under the original census
and budgets. A provider microbenchmark, source deletion, instruction count or
phase envelope alone does not establish an Apollo speedup. Linked normal/map
.text identity binds ownership evidence; maps must show one provider square
kernel per used scalar/ISA, with no new payload spills or division. Expected
SemVer breaks and all affected consumer gates are mandatory.

Square tests instantiate f32, f64, F16 and Bf16 with coordinate and entire-byte
oracles over offsets, ragged sides, special payloads, invalid lengths and
overflow. First/repeated valid and short/long/overflowing calls assert zero
allocations/reallocations; invalid calls also assert exact dimensions and
unchanged bytes. Core tests cover both traversal orientations, tile boundaries,
canaries, non-Copy values/clone counts, error precedence, empty dimensions and
huge zero-sized extents. Existing numerical workloads and tolerances remain.

Host tests cover selected hardware plus the explicit scalar path, not physical
execution of every ISA. Codegen, whole-engine latency and footprint remain
separate gates. Output uses Atlas retention, including the recorded
`output/leto-square-transpose` 14-day/10-GiB policy. Paths in the evidence table
are relative to Atlas's `output/apollo-square-transpose/` unless stated otherwise.

## Experimental evidence and decisions

All consumer size deltas use the unchanged 6,861,824-byte ISA baseline. Historical
results remain evidence at their captured revision, not acceptance of current
source. Provider correctness gates do not override consumer rejection.

| Revision / experiment | Decisive observation and decision | Verification / retained evidence |
|---|---|---|
| Entry `a2006ad`; initial square provider | Five existing focused movement tests establish entry baseline. Square's initial additive contract passes provider checks; downstream acceptance remains separate. | 923 native tests; 9 focused release; 27 doctests (1 existing ignored); minimal features; warning-denied Clippy/rustdoc; 196 applicable minor SemVer checks; 24 unchanged smoke cases under 60 seconds. Four scalars have zero first/repeated successful allocations/reallocations. Atlas `output/leto-square-transpose` retains hashes. |
| `9672ddc`, array construction | Executable +14,336 B: rejected. Assembly `914754C9...2225269433` keeps AVX2 pairs in registers, but outlines AVX-512 `array::from_fn` and Hermes row-buffer permutation (1,272-byte helper frame); successful extent checks destroy eager LetoError. Seeded Copy-array fill and failure-only error construction are the next hypothesis, not a timing claim. | `array-construction/`; no timing acceptance before size/codegen checks. |
| `07bd618`, seeded array | Removes constructor helper and eager drop; -1,536 B versus preceding build but +12,800 B baseline: rejected. AVX2 Skip-iterator control and tile spills remain. Exact-tail iteration and the [Hermes forwarding correction](../../../hermes/backlog.md#hermes-complex-permutation-inlining) target those mechanisms without a Leto backend fork. | `seeded-array/`. |
| Leto `00fd88e` / Hermes `07c5e5f`, feature frame | 6,873,600 B (+11,776; -1,024 versus seeded array): rejected. Assembly `CD76ED04...A8B57BA` removes both outlined helpers; f64 AVX2/AVX-512 pairs stay in registers but retain bounds checks. Frames 200/488 B include duplicated scalar remainder; this library artifact contains no batch or f32 square symbols and cannot attribute whole-file growth. | `feature-frame/codegen.md`; no timing claim. |
| `ce9d02b`, shared tail and narrow extent error | One scalar tail, allocation-free dimension errors and register tile pairs; 6,868,992 B (+7,168): rejected. Counterbalanced full-engine comparison supplies no acceptance basis. Invalid allocation workloads extend, not replace, successful workloads. | Recorded provider gates; `extent-contract/codegen.md`. |
| Cache blocking, canonical tile source `6013768` | Phase attribution is 546–567 us for f64 N=262,144 final transpose on the selected P core; not cache-miss or comparative evidence. Restored 16-by-16 outer blocks keep payloads in registers with larger traversal state. Executable 6,871,552 B (+9,728): rejected on size despite one supported E-core real-half/262,144 gain, paired medians -17.40–35.35%, no supported regression. | 923 native; 9 focused release; Clippy/minimal features; 27 doctests; 24 smokes; all 15 source hashes fixed, independent bounds/coverage review clean. `cache-blocking/leto-gates`, `phase-profile.txt`; unchanged 16-run census/raw audit. Warm allocations/retained bytes match; cold peaks overlap, not identical. Local AVX2/end-point load limits, no physical AVX-512 or miss claim. |
| Apollo pure-copy consolidation | Deletes remaining private copy kernels/trait hook through existing batch API without provider threshold/body change. Executable 6,892,544 B (+30,720); correctness/allocation pass but census has two P-core regressions: rejected. Library retains 1,476 generic assignment instructions plus 12 cleanup funclets/354 instructions around a 370-instruction mover; these are not exact linked bytes. | `pure-copy/codegen.md`; motivates checked canonical core entry. |
| `f3a6dd8`, iterator tile span | Checks stride/span, splits `(SIDE-1)*stride` prefix from exact final SIDE row, seeds from final row and fills via exact chunks; stores use the same split with no remainder/padding loss. No unsafe/API/ISA/workload change. Consumer +9,728 B versus pure-copy, including +7,120 text: rejected. AVX2 303→270 instructions, frame 248→216 B, no division/spill; AVX-512 introduces division/payload staging and frame 472→1,016 B. Forward correction restores per-row `6013768`, without ISA-specific alternative. | Stop on division, outlining, payload spills, frame or text growth. 923 native; 9 focused debug/release; Clippy/minimal features; 27 doctests (1 ignored); warning-denied rustdoc; 24 smokes; 15 hashes fixed. Tile SHA `A06AE5B7BF4AF37DB56797A56D52AB6BF56063C88206DE574F705B994243F45B`; `tile-span/leto-gates/final-checks.json`. |
| `3ad43b73`, checked dense copy | Removes the view/assignment and cleanup family, with no ordinary success eager-error drops. Executable 6,872,576 B (+10,752): not accepted. Count-one batch still divides slice lengths by product because opaque validation hides equality; core preflight remains external. | Independent source review clean; 930 native/366 release; 28 doctests (1 ignored); all-target Clippy/minimal features/warning-denied rustdoc/24 smokes. Both packages: 196 SemVer checks against `a2006ad`, 58 inapplicable each. `dense-copy/leto-gates/final-checks.json`, `dense-copy/codegen.md`. |
| `9a47d6b`, preflight visibility | Adds only `#[inline]` to `validate_length` and `transpose_extent`; preserves checks/order/errors/body. Targeted batch chunk-count division and external preflight are the hypothesis; no count-one clone or promised full residual recovery. Executable remains +10,752 B. Linked square copies have identical instructions/call targets after relocation normalization, but distinct equivalent panic locations and AVX-512 constants. This does not prove ThinLTO import caused duplication. | Reject retained targeted division, new spills, duplicated formatting or text/file growth. Format/all-target Clippy; unchanged 30 debug and 30 release; 14 hashes fixed; source review finds no contract change; prior dense-copy full/API coverage applies. `preflight-inline/final-checks.json`, `preflight-inline/map/`. |
| `437b5028`, shared tile diagnostic | Checked row extraction sends failures to one non-generic cold diagnostic. AVX2 copies fold; AVX-512 remains duplicated through equal constants at different addresses. No new division/payload spill; -2,560 B versus prior, +8,192 baseline: rejected. E-core complex/1,024 regression: paired medians +0.59–4.22%, candidate lower bound 2,977,157 ps exceeds baseline upper 2,975,000 ps. Gains at complex/65,536 and real-half/262,144 do not override regression; no supported P-core direction. | Format/Clippy, unchanged 30 debug/release including allocation checks; independent source review clean. [Linked-code review](../../../../output/apollo-square-transpose/tile-diagnostics/codegen.md), [census audit](../../../../output/apollo-square-transpose/tile-diagnostics/audit-summary.json), unchanged 16-run census. |
| `00665a4`, forward restoration | Restores tile to exact `9a47d6b` source/SHA. Experiment remains reproducible at `437b5028`; validation/errors/dispatch/traversal/workloads unchanged, no rerun of rejected immutable timing. Restored comparison still +10,752 B; campaign remains in progress. | Format/all-target Clippy, unchanged 30 debug/release; 14 hashes fixed during gates, later changes only results/lease release. `tile-diagnostics/restoration/`. |
| `843febc`, all-operation provider entries | One provider square entry/kernel per ISA, normal/map .text identical. Executable 6,866,432 B (-6,144 restored; +4,608 baseline): rejected on size, no timing run. Two consumer error-Debug addresses remain. Batch/core intervals unchanged at 5,936 B; the general count-one call motivates the current inline correction, not a proven cause of residual size. | 930 native/366 selected release; 28 doctests (1 ignored); all-target Clippy/minimal configuration/warning-denied rustdoc/24 smokes; 316 build inputs and lock fixed. SemVer against `00665a4` reports intended free functions/error path/module removals; final module visibility passes supplemental format/Clippy/rustdoc. `provider-entry/leto-gates`; Apollo focused Clippy and 13 movement/workspace tests pass. |

The cache-blocking hypothesis also has a source locality basis: before outer
blocking, a two-element AVX2 tile touches half a 64-byte line in each lower row
and sweeps matrix width before adjacent samples. For an aligned 512-square,
the early strip touches approximately `510 * 64 + 2 * 510 * 16 = 48,960` bytes
between uses. This is an address-footprint model, not measured live cache
occupancy or a miss count. The shared block contract above preserves the
original derivation without inferring residency from its 8-KiB payload bound.

## Established batch evidence

The original addition leaves assignment APIs/behavior unchanged and covers full,
ragged, asymmetric, empty, invalid-length and overflow cases for f32/f64.
Validation is failure-atomic; the warmed census records zero allocations and
reallocations. Its later breaking role migration is described separately above.

Two independent same-binary Criterion runs on the local Windows AVX2 workstation
show provider median reductions of 86.7–88.8% (f32) and 88.9–89.8% (f64) for
1,024 batches of 4x4 matrices; 28.3–53.3% (f32) and 26.1–30.5% (f64) for
256 batches of 16x16 matrices. Every provider/control 95% confidence-interval
pair in the second run is disjoint. The control is unchanged generic assignment
in the same binary. This establishes only that local layout regime; Apollo
must independently verify FFT values, allocation behavior and throughput.
