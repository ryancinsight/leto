# Borrowed CSR forward contract verification

Target: **0.47.0**, accepted [ADR 0036](../adr/0036-borrowed-csr-product-contract.md).
Provider base: `8b4ab407a90c15205405fa6c4d02d04d7a3835ad` (0.46.0).
Verified runtime: `b7670636912f3eb5aee543ac561f32f0a5173b38`.
[Provider review #362](https://github.com/ryancinsight/leto/pull/362) and
[consumer recovery #533](https://github.com/ryancinsight/Coeus/pull/533) remain
independent of preservation #531, unchanged at `4b7d360e`.
Later record updates preserve all Rust, manifests, locks, workflows and profiles.

## Contracts and mathematical justification

The [co-located design](../../crates/leto-ops/src/application/sparse/README.md)
and row-task Rustdoc define the proof obligations. These are proof sketches,
checked by the tests below; no theorem-prover execution is claimed.

- Sealed `CsrIndex` accepts usize and checked i64. Nonnegative, target-width
  representable indices decode individually for addressing. All three original
  slices remain borrowed; no slice reinterpretation or conversion allocation occurs.
- Length, endpoint and monotonicity checks prove `0 <= start <= end <= nnz`;
  every column is below k. StoredOrder permits duplicates and unsorted columns
  without sorting, coalescing or dropping explicit zero coefficients. The existing
  canonical usize constructor and owned representation keep their contracts.
- For each output (r,j), `s_0 = T::ZERO` and
  `s_(p+1) = s_p + values[p]*rhs[column[p],j]` in increasing physical p.
  ScalarStrategy performs separate native-T multiplication/addition under the
  existing NumericElement bound. SimdStrategy preserves Leto's prior slice-hook
  behavior. One authoritative CSR traversal serves both strategies and widths.
- Preparation checks `m*n`, `k*n` and exact lengths before output binding or
  mutation. The sealed width law `get(checked(n)?) == n` preserves those bounds:
  DynamicWidth stores usize and FixedWidth<N> stores unit. Vectors select checked
  width one; matrices retain the dynamic policy. No old strategy gains Copy.
- Distinct rows own disjoint `[r*n,(r+1)*n)` intervals. Every existing output slot
  is initialized and overwritten, including empty rows/shared dimensions. Width
  zero forms no element reference or input read. Wrong shapes reject before writes.
- The unsafe synchronous dispatch scope requires complete joins before return
  **and unwind**, unique in-range row admission and no later task use. Errors
  preserve the dispatcher's result; a panic may execute a subset, at most once.
  Compile-fail examples enforce safe lifetime, exclusivity and invocation boundaries.

Row validation combines representability and ordering in one pass but defers
ordering rejection until all offsets and endpoints are checked. Error priority
remains representation, endpoints, monotonicity, then columns. The old first
endpoint is read again for its check; no exactly-once-read claim is made.

## Specification-first evidence

The original API characterization establishes concrete sorting/coalescing and
FMA counterexamples. For binary64, `[2^53,1,-2^53]` sums to +0 in stored order,
whereas sorting or coalescing can produce 1. With epsilon equal to spacing at
1, separate `-1 + (1+epsilon)*(1-epsilon)` gives +0; FMA gives `-epsilon^2`.
Adjacent, tie, signed-zero, subnormal and exceptional witnesses retain native
precision and Miri arithmetic-error modeling; no empirical tolerance is tuned.

The new fixed-width contract first fails against the preceding API with E0432,
then passes exact widths 0/1/3, mismatch rejection and all sixteen scalar families.
The three multi-fault row witnesses pass the unchanged two-pass validator first
(Nextest `53667e4d`, 1/1) and retain exact reasons after fusion. The native
256-case stored-order property test uses shrinking and an independent reference.
Allocation guards execute real view/product operations without copying CSR arrays.

## Final runtime gates — 2026-10-15

Host: Apple M4 Max, aarch64-apple-darwin; Rust 1.97.0. Miri uses nightly
`822182db4c` (2026-10-13). Commands run outside Atlas with absolute paths and
separate target directories for different canonical dependency graphs.

| Gate | Executed result |
| --- | --- |
| Complete native leto + leto-ops | 1202 passed, zero skipped; `a90eb162`, 22.152 s |
| No-default sparse + LU native | 220 passed; 599 unrelated excluded; `2a162946`, 40.295 s |
| All-target strict Clippy, both feature modes | Pass, `-D warnings` |
| Doctests | 38 passed; one existing ignored example |
| Warning-denied public Rustdoc | Pass for both packages |
| Explicit minor API comparison, both packages | Each 196 passed, 57 tool checks skipped, versus actual base locks |
| Formatting and diff whitespace | Pass |

The API comparison uses complete all-feature Rustdoc JSON with private/hidden
items. Existing private-JSON link warnings remain diagnostic data; the separate
public warning-denied documentation gate passes. This does not turn the 57 tool
skips into executed checks or certify semantic obligations through signatures.
The no-default LU lint is fixed at its source with `?` and verified by real LU tests.

| Strict-provenance Miri scope | Stacked Borrows | Tree Borrows |
| --- | --- | --- |
| 20 deterministic sparse/allocation cases | `0fc8a78a`, 313.375 s | `b096e4c9`, 525.091 s |
| ARM32: 10 signed/product/partition cases | `06688d53`, 100.163 s | `cd178150`, 242.782 s |

Both models retain isolation, leak checks and default arithmetic-error modeling.
Host selection excludes 811 unrelated cases; ARM32 excludes 493 and disables
parallel-runtime defaults. ARM32 executes all scalar families, 2^32 rejection,
stored order, separate rounding, fixed widths, complete writes and failure atomicity.
Real Clang cross-compiles the existing C helper. This proves the exercised provider
address/scalar behavior; full Coeus ARM32 still stops before tests in Moirai's
unchanged deque alignment assertion. i686 stops at Eunomia's binary64 alignment
assertion. Neither is reported as a full-consumer pass.

Native per-test termination stays 60 s. The separate interpreter profile is
bounded at 600 s per test; process-tree supervisors additionally bound commands.
Cold native compilation has a 1500 s command budget without extending test limits.
Earlier shared-target failures are build/runner diagnostics, not executed passes:
provider doctests first hit E0463 with competing Hermes graphs; isolated graphs
pass. No shared target is cleaned and no other owner's files are modified.

## Consumer numerical and performance evidence

Coeus runtime `2135975e` at main base `f0d01021` uses the provider-owned row
kernel with the unchanged selected backend's `parallel_for(0, rows, ...)`.
It passes 1272 native tests (eight existing ignored), strict workspace Clippy,
159 doctests (two existing ignored), warning-denied docs, three explicit minor
API comparisons and thirteen memory cases in each strict borrow model.
Its [current record](https://github.com/ryancinsight/Coeus/pull/533) includes the
initial TCP-test failure, unchanged-source successful rerun and remaining limits.

The final comparison retains all five workloads, exact pre/post output oracles,
20 ms minimum calibration and 21 alternating-order pairs. Current-main reference
files remain byte-identical. No owned heavy verification overlaps measurement;
other host activity is uncontrolled. Ratios are candidate/main medians, not
universal non-regression bounds or confidence intervals.

| Shape / RHS | Sequential ratio [Q1,Q3] | Moirai ratio [Q1,Q3] |
| --- | --- | --- |
| 1024x1024 / 1 | 1.028 [1.018,1.048] | 1.023 [1.021,1.033] |
| 16384x16384 / 1 | 0.999 [0.990,1.018] | 1.025 [1.022,1.036] |
| 1024x1024 / 16 | 0.094 [0.093,0.095] | 0.093 [0.092,0.094] |
| 4096x4096 / 64 | 0.074 [0.074,0.075] | 0.179 [0.167,0.190] |
| 4097x4096 / 257 | 0.211 [0.210,0.211] | 0.299 [0.295,0.302] |

SpMV retains about 2–3% median overhead in three tested cases. Large sequential
SpMV is near parity. Large SpMV main/candidate times are 171.731/171.965 us
sequential and 114.023/117.479 us Moirai: backend parallelism remains observable.
The final Moirai SpMM ratios span 0.093–0.299. Earlier unfavorable results remain
revision-scoped in the consumer record and Git history: the initial adapter costs
2–4 times main, the pre-specialization graph 2–8%, and fixed-width metadata alone
still gives a 1.064 large-Moirai ratio. Cross-run host variation prevents assigning
all improvement causally to row-check fusion. All exact numerical oracles pass.

## Fresh consumers and integration boundary

A new unseeded external consumer of final provider `b7670636` resolves eight
canonical providers with zero excess identities and no path/patch overrides.
Its executed binary aliases all three original signed arrays, computes vector
`[98,0]` and matrix `[140,166,200,0,0,0]`, checks wrong-width ShapeMismatch fields
and inspects negative-column StorageError. This is value execution, not metadata only.

Compatible [Hephaestus #497](https://github.com/ryancinsight/hephaestus/pull/497)
and [Apollo #674](https://github.com/ryancinsight/apollo/pull/674) preserve every
Rust file and external lock record against their recorded bases. Both admit
`>=0.46.0, <0.48.0`; all their actual hosted checks pass, with hardware-required
and configured release/informational jobs explicit skips. Fresh source metadata
resolves 7/9 providers respectively, with zero excess identities.

Fresh Coeus `2135975e` still rejects Apollo ^0.29.2 before compilation because
its default branch exposes 0.29.1. The complete review lock passes separately;
it does not establish merged-default or registry availability. Provider-first
integration/publication and fresh registry-consumer receipts require separate
authorization. Branch reviews do not authorize main publication or Pages.

Provider runtime CI [38806768474](https://github.com/ryancinsight/leto/actions/runs/38806768474)
passes native/strict/docs/API/guards while its full hosted Miri job remains in
progress at record time. The PR checks remain the exact-head hosted receipt.
No main merge, tag, release, unavailable architecture pass or hardware pass is claimed.

## Reproduction

Run outside Atlas; set LETO to the actual checkout and use a target directory
reserved for its graph. Use the committed Nextest configuration and Rust 1.97.

```sh
cargo +1.97.0 nextest run --locked --manifest-path "$LETO/Cargo.toml" \
  --config-file "$LETO/.config/nextest.toml" --target-dir "$TARGET" -p leto -p leto-ops
cargo +1.97.0 nextest run --locked --manifest-path "$LETO/Cargo.toml" \
  --config-file "$LETO/.config/nextest.toml" --target-dir "$TARGET" \
  --no-default-features -p leto-ops -E 'test(sparse::) | test(lu)'
cargo +1.97.0 clippy --locked --manifest-path "$LETO/Cargo.toml" \
  --target-dir "$TARGET" -p leto -p leto-ops --all-targets -- -D warnings
# Repeat Clippy with --no-default-features; execute --doc tests and warning-denied public docs.
MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri nextest run --locked \
  --manifest-path "$LETO/Cargo.toml" --config-file "$LETO/.config/nextest.toml" \
  --target-dir "$MIRI_TARGET" -p leto-ops --profile miri --test-threads 1 \
  -E '(test(sparse::contracts::) & !test(properties::)) | test(sparse_allocations::)'
# Repeat with -Zmiri-tree-borrows. ARM32 additionally sets the real Clang target,
# --target armv7-unknown-linux-gnueabihf --no-default-features --test ops_tests
# and selects signed::, product:: and partitions:: contract modules.
```
