# Borrowed CSR forward contract verification

Current target: **0.47.0**, accepted ADR 0036. Checked borrowed representation,
prepared arithmetic and row dispatch are implemented and verified. Coeus #533
uses the common row kernel with typed errors and its existing backend. The
measured SpMV regression closes after local accumulation and checked-view reuse;
paired medians now lie near parity, with explicit host-load uncertainty.
Earlier sections preserve proposal and intermediate evidence.

## Original audit revisions and scope

Provider base: Leto main `248fe4f63c674f6fca1f89ac19c1f817e67bc417`
(0.45.2). The lock selects Eunomia 0.9.1 at
`ee7c6911652ee7438566d5def1d6d4f519abc7fd`, Hermes 0.8.2 at
`8648f280daed75d872ee87c2b93a3c177a4c41c1`, and Moirai 0.9.2 at
`2b189f0a8aaf4a45aab1f2085c2ec748e483b8b3`.

Consumer audit: Coeus original base `cd8e984e0d9c4f3621bd89f7830646de845a22e5`
and independently recovered bridge `35e452686785cb409dd648c0565b3f9600e5de8d`
([draft #533](https://github.com/ryancinsight/Coeus/pull/533)). Those revisions
have identical tensor forward source; their canonical lock uses older
provider revisions. Coeus preservation
[draft #531](https://github.com/ryancinsight/Coeus/pull/531), head
`4b7d360efdd829538ea68905795c02b7bd4f4252`, remains intact.
Coeus main advances during this audit to
`dd27214f88a5c23e0e9abc58e014e58b902d67fb` (Moirai 0.9 dependency migration).
Its relevant sparse, scalar, backend and CPU storage sources remain identical
to the original base; validation of the updated dependency graph belongs to
the independent recovery draft, not this unimplemented provider proposal.

The initial 0.45.3 [patch] increment contains a proposed [ADR 0036](../adr/0036-borrowed-csr-product-contract.md),
a [co-located design](../../crates/leto-ops/src/application/sparse/README.md),
current-API characterization and development tracing dependencies. It adds
no production Rust change and makes no performance improvement claim. It
does not implement signed/stored-order views or a row dispatch scope.

## Mathematical witnesses

| Executed witness | Exact oracle and reason |
| --- | --- |
| Coalescing | Binary64 `(2^53 + 1) - 2^53` is +0 under separate ties-to-even operations; coalescing the equal-column `2^53` and `-2^53` first leaves 1. The real canonical Leto product computes 1. |
| Sorting | Stable column sorting of values `[2^53, 1, -2^53]`, columns `[1, 1, 0]`, moves `-2^53` first and yields 1 instead of +0. The current view rejects the original structure. |
| Arithmetic policy | For `e = EPSILON`, exact `(1+e)*(1-e) = 1-e^2`. The product rounds to 1 before separate addition of -1, yielding +0. FMA yields `-e^2`, exactly representable for binary32/64. Current vector output is +0; matrix output equals the real scalar slice hook, including vector body and ragged tail. |
| Row partition | Current standalone CSR construction rejects absolute offsets `[1,3]`; rebasing to `[0,2]` computes `3*7 + 5*11 = 76`. A prepared product must retain the full descriptor instead of rebuilding per-row arrays. |
| Empty dimensions | All 64 triples `(m,k,n)` in `0..=3` with empty CSR produce `m*n` matrix and `m` vector zeros, overwriting NaN sentinels wherever slots exist. |
| Failure atomicity | Both current products reject a one-slot output for a two-row result with typed `ShapeMismatch`; a NaN payload sentinel remains bit-identical. |

The rounding witness deliberately characterizes the existing ISA-dependent
matrix hook. It is not the future scalar-strategy acceptance test: that test
must require +0 bits exclusively. No future signed-index or scheduling test
is represented as passing merely because a current canonical case passes.
These are characterization tests, not red/green claims for a production fix.

## Executed gates (2026-10-15)

Host: Apple M4 Max, `aarch64-apple-darwin`; native Rust 1.97.0;
Miri 0.1.0 (`822182db4c`, 2026-10-13). Commands run
from `/private/tmp`, outside Atlas, with absolute manifest and shared target
paths. This prevents the Atlas development path overlay from changing the
canonical git dependency graph. A process-tree timeout bounds every long
command; native tests use Nextest's committed 30-second slow / 60-second
termination budget. No individual sparse test reaches the slow threshold.

- Native default-feature sparse suite: **33 passed**, 456 unrelated tests
  excluded by the explicit sparse filter, including all six new witnesses.
- Native sparse suite with `--no-default-features`: **33 passed**, the same
  456 unrelated tests excluded.
- Strict default-feature `leto-ops --all-targets` Clippy: **passes** with
  `-D warnings`.
- Strict `--no-default-features --all-targets` Clippy: **fails in unchanged
  baseline code**, `application/linalg/lu_batch.rs:112`,
  `clippy::question_mark`. No lint is suppressed. The separate
  `LETO-NOPARALLEL-CLIPPY-2026-10-15` backlog item records the fix and LU tests.
- Warning-denied `cargo doc -p leto-ops --no-deps`: **passes**.
- Formatting, diff whitespace and local Markdown links: **pass**. The ADR
  index is generated from the proposed decision's canonical header/status.
- Strict-provenance Miri: **all six witnesses pass in each borrow model**,
  isolation enabled, default seed 0. Stacked Borrows run
  `85d5defd-54f3-4935-95c7-db8a000a5937` takes 98.937 seconds; Tree Borrows run
  `80ffd963-bb25-43c4-b9e8-390b61c63351` takes 243.369 seconds. Each excludes
  483 unrelated tests and stays within the committed interpreter budget.

The Stacked Borrows full run receives one Nextest `LEAK` marker for the
canonicalization witness despite the test's successful exit. Nextest uses
that marker for [late stdout/stderr handle closure](https://nexte.st/docs/features/leaky-tests/),
not a Miri heap-leak finding. Its isolated rerun
`41b432e9-5fd1-4d91-abbf-76995ac9cbea` passes without the marker; the complete
Tree Borrows run has no marker. The runner cause remains undetermined. No
timeout, leak rule, input, assertion or Miri safety option is relaxed to hide
it; these results do not claim universally clean runner process teardown.

```sh
cargo +1.97.0 nextest run --offline --locked \
  --manifest-path "$LETO/Cargo.toml" --target-dir "$ATLAS/target" \
  --config-file "$LETO/.config/nextest.toml" \
  -p leto-ops --test ops_tests -E 'test(ops::sparse::)'
cargo +1.97.0 clippy --offline --locked \
  --manifest-path "$LETO/Cargo.toml" --target-dir "$ATLAS/target" \
  -p leto-ops --all-targets -- -D warnings
cargo +1.97.0 fmt --manifest-path "$LETO/Cargo.toml" --all -- --check
MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri nextest run --offline --locked \
  --manifest-path "$LETO/Cargo.toml" --target-dir "$ATLAS/target" \
  --config-file "$LETO/.config/nextest.toml" --profile miri \
  -p leto-ops --test ops_tests -E 'test(ops::sparse::contracts::)'
# Repeat with MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-tree-borrows'.
```

The first local test build corrects a missing `RealField` witness bound before
execution; the successful build uses the bound that actually defines
`EPSILON`. Tracing disables its default attribute feature, so all external
package versions and source revisions remain those of the base lock. Only
local package versions and the new direct development dependency edges change.

An initial Miri characterization run exposes an invalid fixture assumption:
two `powi(53)` calls need not produce an exact power of two under
[Rust's documented precision contract](https://doc.rust-lang.org/std/primitive.f64.html#method.powi).
The corrected fixtures use the exactly representable binary64 literal
`9_007_199_254_740_992.0`; no oracle, input domain or numerical assertion is
weakened. Miri's extra rounding-error modeling stays enabled. The same exact
fixture correction is applied to Coeus #533; its 293 native tests and five
tensor-forward strict-provenance Miri cases pass again.

## Fresh consumer and compatibility limits

A clean consumer outside Atlas, without a lockfile or local overrides,
requests `leto-ops` 0.45.2 from the exact base Git revision and runs online
`cargo +1.97.0 metadata --format-version 1`. Resolution fails before
compilation: Leto requests Eunomia `^0.9.0`, while the default Git branch now
provides 0.10.0. Locked tests do not establish a resolvable fresh source graph.
The specification patch changes no external version bound or locked package.
The same clean consumer then requests the published proposal commit
`890a73c77dd40f2adc7982dcaaabe26fd5ca5572` with version 0.45.3 and fails on the
same Eunomia bound. Both base and candidate failures execute online; neither
uses a copied lockfile or development override.

Production Rust sources and all external lock entries compare byte-for-byte
with the recorded base. No public symbol or production algorithm changes.
`cargo-semver-checks`, full workspace tests, package publication, native
non-host architectures and 32-bit Miri are not executed for this
documentation/test increment. A release or the later implementation still
requires its applicable compatibility and fresh-consumer gates.

## Prospective acceptance after ADR sign-off — not executed

| Gate | Required coverage | Mathematical or safety justification |
| --- | --- | --- |
| Borrowed representation | Valid `usize`/`i64` slices alias all inputs; negative offsets/columns, `i64::MAX`, and `2^32` on 32-bit reject before output writes; malformed lengths, endpoints and monotonicity | Lossless scalar decoding plus CSR bounds proves every address lies in its input slice. No same-length slice cast can satisfy this on unequal widths. |
| Ordering policy | Canonical accept/reject behavior unchanged; stored-order unsorted/duplicate cases pass with exact consumer outputs; canonical-only CSC calls reject the wrong typestate at compile time | Physical entry order defines the recurrence. Canonicalization is not floating-point equivalent. |
| Scalar coverage | Every existing Coeus integer type, four real float types and their complex family compiles without a new bound; exact nontrivial products and documented overflow behavior | `NumericElement` is already a Coeus supertrait. Native operators preserve the scalar contract without an ad hoc wider accumulator. |
| Rounding | Binary32/64 counterexamples require +0 for scalar mode; ties, adjacent values, cancellation, signed zero, NaN payloads, infinities, subnormals, explicit zero coefficients and complex values compare to unchanged Coeus | Bitwise comparison follows the same per-output recurrence. Where a scalar's NaN behavior differs across ISA, characterize the exact supported contract rather than introduce a tolerance. |
| Layout and errors | Coeus negative/overflow/column/storage errors and precedence; bad rank/shape; noncontiguous rejection; readable bounds; accepted nonzero contiguous offsets | Preserve existing validation before dispatch, slice only the validated logical region, and propagate allocation/mutable-access errors. |
| Output proof | Nonempty, empty-row, zero-shared and zero-output dimensions; sentinel/canary checks; all output rows visited once; no mutation after preflight rejection | Checked `m*n` gives disjoint row intervals; fill-before-update establishes initialization. |
| Dispatch safety | Real sequential/Moirai backends, 4,096-row threshold boundary, nested execution, worker panic and caller panic completion; no task escape or duplicate-row invocation through safe APIs | Completion must hold on return and unwind. Existing normal-return trait wording is insufficient for arbitrary external backends until clarified. |
| Memory and architecture | Strict-provenance Miri under Stacked and Tree Borrows; supported native targets and 32-bit target checks; zero-width paths; both initialized-output and failure paths | Proves dynamic aliasing/address validity for exercised paths; it does not replace the unsafe API proof. Earlier Coeus 32-bit Miri attempts stop in upstream alignment assertions and do not establish portability. |
| Allocation | Real allocator census after warmup: no CSR or dense input copies for contiguous operands; no per-row heap allocation; include wide outputs beyond 256 columns | Borrowed descriptors use constant auxiliary space; caller-provided output stores the recurrence directly. |
| Differential and properties | Reuse all 689 small canonical sparsity masks and add unsorted/duplicate stored sequences with meaningful shrinking; real end-to-end tensor products and exact value assertions | Independent dense or stored-order oracles prevent an empty/default kernel from passing. |
| Performance | Same four existing Coeus workloads, wide RHS, paired alternating order, full value checks and consumed output; include validation, initialization, allocation and scheduling | Existing direct serial-provider SpMV measured a 2.069 median time ratio at 16,384 rows. Kernel formula equivalence alone cannot establish preserved backend throughput. |
| Compatibility and delivery | Formatting, strict Clippy, docs, native tests, API compatibility, clean-consumer dependency resolution, exact-revision CI and provider-first integration | A locked build does not prove a fresh published consumer graph resolves. No merge or release gate is inferred from characterization tests. |

Existing performance evidence is in
[Coeus #533's verification record](https://github.com/ryancinsight/Coeus/blob/35e452686785cb409dd648c0565b3f9600e5de8d/docs/verification/sparse-recovery.md).
It is prior consumer evidence, not a benchmark of this unimplemented provider
proposal. Provider and consumer dependency owners must resolve their current
fresh-graph version drift before claiming release readiness.


## Approved implementation — 2026-10-15

Ryan Clanton explicitly approves ADR 0036 in the task conversation. Target
0.46.0 opens the provider implementation. The first implementation retains the
original locked provider graph against main `248fe4f` while the new current-main
dependency update is inspected. This is intermediate evidence, not final-head
acceptance:

- Acceptance tests fail on unchanged source: missing `CsrProduct`, `StoredOrder`
  and checked `CsrView::builder` (compiler E0432/E0599).
- Native sparse Nextest: 98 pass, 725 unrelated tests excluded, default budget.
- All-target strict Clippy passes; no diagnostics are suppressed.
- Strict-provenance Stacked Borrows Miri: eight new representation/product/
  dispatch/allocation tests pass, 816 unrelated excluded, run
  `6aaeb2b2-a02c-4372-be28-066ed9c0df86`, 31.026 s, no handle marker.
- Commands run from `/private/tmp`, with absolute manifest and the Atlas shared
  target, `--offline --locked`, Rust 1.97.0 or `+nightly` for Miri. Miri keeps
  isolation and default arithmetic error injection enabled.

Before delivery, repeat applicable gates after incorporating the newly merged
provider dependency update, then execute Tree Borrows, architecture, legacy API,
fresh-consumer and complete-product performance gates. No Coeus forward cutover
is certified by this intermediate provider result.


## Current provider graph and implementation acceptance

Implementation incorporates main `7172e57343dff9c699e5f51e99b3dd0dfe1105e9`
through merge `a5262fa`. That main change updates only manifests/lock to the
Eunomia 0.10 cohort: Eunomia `15fbb94c`, Hermes `7a584238`, Mnemosyne `481c7452`,
Moirai `44a962f0`. The subsequent candidate removes unintended `Copy` additions
on the existing strategy markers; only new borrowed descriptors are copyable.

- Full native `-p leto -p leto-ops`: **1198 pass, zero skipped**, run
  `ed8af077-078a-4c5d-ad5c-0f5ac1781298`, 19.558 s. This includes 256 generated
  stored-order products with shrinking and independent column oracles.
- Strict default-feature all-target Clippy passes on that full test graph.
- Actual-source `cargo semver-checks`, baseline `7172e573`, release type minor:
  **196 checks pass, 57 tool skips**. The first attempt correctly rejects the
  unintended strategy `Copy` implementations; removing them restores the
  compatible API. No SemVer diagnostic is suppressed.
- Doctests: **32 pass, one pre-existing ignored**, including three new compile-
  fail contracts for input lifetime, output exclusivity and unsafe invocation.
- Corrected-API strict Miri, eight representation/product/dispatch/allocation
  cases, one test process at a time: Stacked run
  `1ac777e0-1121-4dce-a2fd-d3945d89a7a5` (50.357 s), Tree run
  `ae8b5593-ffc4-4489-b04c-7b3672bdc2b1` (94.990 s), all pass with no runner
  handle markers. Earlier concurrent Tree run `87ceb3aa` passes with one late
  handle marker; serial results do not establish its cause.
- Added adjacent-value and nonzero-complex-component fixtures at binary32,
  binary64, F16 and Bf16, plus all 64 empty-dimension triples: Stacked run
  `c60f40f3-b1f5-4252-9eb7-8e6998df61ee`, **three pass**, 107.213 s. Tree repeat
  remains in progress at this checkpoint.
- 32-bit ARM strict Miri signed rejection/representability test passes, run
  `84a3a075-98f6-4c77-bfd4-a03c424a7909`, 4.565 s. Real Clang cross-compiles the
  alloca dependency (`CC_armv7_unknown_linux_gnueabihf=clang`, matching target
  flag); no dependency is replaced. i686 first requires the same compiler
  configuration, then stops at Eunomia's existing `align_of::<F64>() == 8`
  assertion, before these tests execute. 32-bit ARM evidence does not claim
  an executed i686 result.

Commands retain `--locked`, real Git providers, the native 60-second Nextest
budget, strict provenance, Miri isolation and default arithmetic error injection.
Full native and strict Clippy repeat after changes; only actual executed checks
are counted. Provider/consumer performance and fresh exact-Git resolution remain
completion gates, not conclusions derived from these correctness tests.


The Tree repeat of the three precision/empty cases passes at the published
`3fa5d8d3dea3317a2bcced1f62bdcb67457f816d` candidate: run
`d167b3be-8df0-4693-9f6d-f48e9264cb18`, 209.345 s, no handle marker. A clean
external Git consumer of that exact revision resolves from an empty local
manifest without inherited lock/path/patch overrides, aliases original signed
indices, and computes `[68, 0]` for `[3,5]` at columns `[1,0]` against `[7,11]`.
Strict feature-disabled Clippy and 107 sparse/LU native cases also pass.

Main then advances to `8b4ab407` (0.46 numeric cohort), independently of #362.
The sparse addition moves to 0.47.0; merge resolution retains both owners'
boards/changelog and all current provider constraints. Current Coeus graph
resolution rejects Hephaestus's old `leto ^0.45.2` edge before compilation; this
is a shared nominal-provider boundary, not a sparse numerical failure. Follow
provider-first delivery and prepare separate dependency-binding changes rather
than retaining old pins, duplicating provider types, or adding conversion shims.


## Initial performance attribution and one-column correction

A same-graph reference compiles all five sparse/pointer files byte-identically
from Coeus main `f9ebef34`. Candidate uses provider `752ccc5` with original
Coeus validation plus the checked provider builder. All dyadic result checks
pass before and after 21 alternating-order pairs, including width 257.

| Shape / RHS | Sequential candidate / main | Moirai candidate / main |
| --- | ---: | ---: |
| 1024x1024 / 1 | 2.401 | 2.418 |
| 16384x16384 / 1 | 4.204 | 2.013 |
| 1024x1024 / 16 | 0.101 | 0.102 |
| 4096x4096 / 64 | 0.080 | 0.205 |
| 4097x4096 / 257 | 0.214 | 0.326 |

The Moirai large-vector ratio has paired quartiles [1.963, 2.058]; this is a
blocking regression, not acceptance. Native disassembly of the sequential
adapter confirms `fmul; ldr partial_sum; fadd; str partial_sum` within its
inner entry/column walk. The generic dynamic-width row path prevents a vector's
single accumulator from staying local across entries. Backend dispatch remains
present, so calling the provider in parallel is insufficient by itself.

The correction retains one CSR traversal in `accumulate_row` and selects a
single local initialized accumulator for width one. It writes the completed
value once, preserving both scalar and slice-hook arithmetic; wider rows keep
the measured contiguous row path. No index/dense copy, per-row allocation,
reordering or alternate scheduler is introduced. All 104 sparse native tests
pass after the change (`d4508a93-3291-4b4e-8c8f-448ea37ebc0f`). Repeated paired
performance and both memory models remain required before accepting this fix.

Additional numerical tests cover least-subnormal halfway ties and stored
positive/negative zero multiplied by infinity at all four real precisions.
IEEE requires NaN class for the latter; Rust does not specify its payload.
The scalar path does not drop those stored entries. A positive public product
example and all three compile-fail contracts pass within 33 successful
doctests (one pre-existing ignored example).

## Exact local-accumulator candidate

Provider `a277833444ed6d54b5b58cf72bf8911c4c9cbb3a` keeps a one-column
accumulator local while sharing the same CSR traversal and arithmetic policy.
All 1200 native core/ops tests pass, zero skipped (Nextest `a2b13942`,
19.948 s); strict all-target Clippy passes in default and feature-disabled
operation modes. Native tests retain the committed 60-second per-test limit.
A broader invocation first stops at unchanged PyO3 0.23's rejection of the
host's default Python 3.14; that failed Python build is not a native-test pass.

All 18 deterministic sparse memory contracts pass with isolation and strict
provenance in both models, one test process at a time: Stacked `e0d2cf72`,
195.880 s, and Tree `2ab1a28b`, 469.433 s. Each excludes 811 unrelated cases;
no late-handle marker appears. The added subnormal/exceptional cases retain
extra arithmetic-error modeling. The 256-case generated property test runs
natively; deterministic memory witnesses exercise the unsafe address,
initialization, ordering, precision and dispatch boundaries under Miri.
The checked-address proof applies independently of random input generation.

The local-accumulator diagnostic run improves the initial one-column result,
but still measures roughly 1.4–1.6 times the unchanged Coeus recurrence.
It overlaps interpreter verification and is not a final throughput claim.
The consumer removes its duplicate healthy-path structure scan by reusing the
checked view. Exact-error multiple-fault witnesses first pass on unchanged
main, then all 151 real CPU integration cases pass with the revised adapter.
Performance acceptance remains open until that path is measured.

## Complete provider and consumer source evidence

All actual hosted checks pass at runtime source `a2778334`, including Rust,
locked-source SemVer, dependency/version/lock/ADR guards and the full Miri job
([run 38776871929](https://github.com/ryancinsight/leto/actions/runs/38776871929)).
Configured release and informational jobs skip; no registry receipt is claimed.

The actual row kernel executes eight signed/product/partition cases on
`armv7-unknown-linux-gnueabihf`, without default parallel features, under both
strict/isolation models: Stacked `6cafdb7b`, 123.766 s; Tree `ee7e731c`,
230.257 s. The all-sixteen-family product, stored-order/separate-rounding
witness, target-width 2^32 rejection, empty output and failure-atomic cases
pass. Real Clang cross-compiles the existing C helper. Full Coeus ARM32 still
stops before tests at Moirai's deque alignment assertion; its Tree command does
not execute. Provider-only execution is not full-consumer portability proof.

Three fresh, unseeded external consumers select actual review sources without
local paths/patches: Leto 0.47 (`a2778334`), Hephaestus 0.22.1 (`a153f475`),
and Apollo FFT 0.29.2 (`f8fa292c`). Metadata resolves with 7/7/9 canonical
providers respectively and zero excess identities. The real fresh Leto program
aliases all original signed arrays and computes `[98,0]` and
`[140,166,200,0,0,0]`, including an empty row; negative columns reject.

Coeus recovery retains main's S2/S3 refactors at `33f6f1f0`. Its final complete
review graph passes 1272 native tests (`d1f54759`, 5.784 s; eight existing
ignored cases), all-workspace/all-target strict Clippy, 159 doctests (two existing
ignored examples) and warning-denied documentation. Thirteen selected cases
pass each borrow model, including real backend panic completion and capture
release. An isolated interpreter process shuts down actual idle global workers
only after all callback assertions; no leak or isolation check is disabled.

The final 21-pair current-main comparison uses byte-identical reference files
and the same real current CPU graph. SpMV paired medians span 0.93–1.03; tested
Moirai SpMM ratios are 0.092, 0.182 and 0.291 for widths 16/64/257. Quartiles,
exact-output checks, source fingerprint and the host-load limit live in
[the current consumer record](https://github.com/ryancinsight/Coeus/blob/4c2a29c5d5e204e2a7c45eff66684b526e3164ae/docs/verification/sparse-forward-cutover.md).
These results do not claim a universal non-regression bound.

Compatible binding reviews #497/#674 retain Leto 0.46 and admit 0.47. They keep
all Rust bytes and external locks unchanged against their actual current bases.
Fresh Coeus defaults still reject Apollo ^0.29.2 because main exposes 0.29.1;
the complete locked review graph passes independently. Provider-first default
refresh and actual registry verification remain integration/release gates.
Preservation Coeus #531 remains open, draft and unchanged at `4b7d360e`.


## Consumer record advances through current main

Coeus preserves main `f0d01021` and its CUDA crossover/provider-cohort update,
including all 231 registry records. Hephaestus compatibility #497 preserves
both concurrent CUDA fixes and advances to 0.22.2 (`e867be0d`) because 0.22.1
is published independently. That provider source passes every actual hosted
check; hardware-required NVIDIA/AMD jobs remain explicit skips.

The final Coeus review graph repeats 1272 native passes (`b08e612b`, 6.042 s;
eight existing ignored), strict all-workspace/all-target Clippy, 159 doctests
(two existing ignored) and warning-denied documentation. Both thirteen-case
macOS memory jobs and both twelve-case Linux memory jobs pass on the prior
complete graph; Linux excludes only the real Moirai case because its topology
`opendir` is unsupported with Miri isolation. That unchanged 4097-row scenario
executes on macOS. Isolation/leak checks and all assertions remain enabled.

The final-lock 21-pair repeat observes 2–8% median SpMV overhead, with the
small-vector quartiles above 1; Moirai SpMM ratios span 0.09–0.32. The provider
retains the backend row-dispatch seam. These results qualify the earlier
near-parity observations; they do not establish universal performance
preservation. The updated proof, exact-value timing tables and source
fingerprints live in [the consumer review](https://github.com/ryancinsight/Coeus/pull/533).
No provider runtime changes occur in this record update.
