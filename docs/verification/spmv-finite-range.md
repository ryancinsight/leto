# CSR SpMV finite-range recovery [patch]

The published 0.46.2 four-accumulator kernel can form nonfinite lanes from
finite products although the former stored-order fold remains finite. This
preparation targets native Leto/LetoOps 0.46.3; no release is accepted here.
The separate 0.47 CSR/FFI contract and private Python version remain owned by
that integration lane. No owner source, policy, budget or setting is changed.

## Reproduced range and caller failure

Let L = round_native(3*MAX_FINITE/4), x=ones, and one stored row be
[L,-L,L,-L,L,-L,L,-L]. All inputs/products are finite. The old fold cancels
pairs to zero, but the four lanes form ±2L, overflow and combine to NaN.
A ninth trailing unit gives the analogous exact result one. Distinct sorted
columns and matching dimensions keep every input valid.

Both native widths fail these assertions before production changes. The same
rows followed by identity rows make an exact initial x=ones solution for
b=[0(or1),1,...,1]. Through the actual reviewed Athena #126 public GMRES API,
0.46.2 reports NonFinite at iteration zero. With recovery, both row lengths
and widths report InitialResidual, zero residual/iterations and one operator
application. Before and after use the same assertions and coherent graph.

Tests receive each native finite maximum as fixture data and assert its
finiteness. NumericElement's MAX_VALUE is an infinite reduction identity;
it is not used as the finite fixture bound. Initial setup diagnostics are
retained in the external packet, followed by valid finite-input reproductions.

## Algorithm and numerical limits

Compute the existing four-lane result R unchanged. If R is finite, return it
unchanged. Otherwise, replay the original stored-order native fold over the
same row and input, with no allocation. Thus all finite 0.46.2 values retain
bit patterns; the rare fallback reproduces the original native operation
sequence. A genuinely nonfinite stored fold still propagates its outcome.
The worst-case work remains O(nnz+m), with at most two row passes.

No wider accumulator, compensation, tolerance or public trait/API is added.
A finite exact real sum does not guarantee either native fold remains finite.
There is no universal correctly-rounded or preferred finite reduction order.
Classical no-overflow forward-error bounds depend on sum(abs(products));
large cancellation can make relative error in the small answer large.
Integer-valued inputs also need exactly representable products/intermediates
for bitwise agreement between orders.

The new controls preserve four-lane improvements: one extreme row sums to
finite zero while the old prefix fold overflows; another has exact result
four while the old fold loses three units. Finite cancellation differences,
including the earlier 4/5-to-zero characterization, are not misrepresented
as universally repairable floating errors. This patch targets the newly
introduced nonfinite range class while preserving finite-order semantics.

## Verification and acceptance

Six native tests exercise owned/allocated/borrowed/strided products, both
row lengths, retained accuracy/range wins and genuine NaN/infinite outcomes.
The two native range tests and two real GMRES tests fail before the production
edit and pass afterward at both debug and optimized settings. The pre-existing #375 small-integer test is unchanged.

All 1,209 workspace tests pass without skips, including the unchanged current
binding tests with an existing supported Python 3.9/NumPy 2.0 environment and
its correct relocated standard-library root. Strict all-target/all-feature
Clippy, minimal features, Rustdoc and actual Rust1.95 checks pass. Doctests
pass with the existing ignored example explicit; no ignore is added.
The future Python/NumPy 0.30/common-FFI cohort is not adopted or certified.

Exact before/after source modules compiled against the same real types cover
5,160 cases per native width at debug/release: 3,323 finite results remain
bitwise unchanged, 124 nonfinite lane results recover finite stored values,
and 1,713 remain nonfinite as the stored fold requires. All original corpus
modes remain; explicit all-ones modes add recovery coverage. Each profile
also executes four strict zero-allocation windows of 10,000 actual owned/
borrowed calls and a positive allocator control. This is corpus evidence,
not a universal error/range proof.

Explicit patch API comparisons against published 0.46.2 pass 223 checks with
30 tool-inapplicable skips for each native crate. Rustdoc JSON extraction
alone uses its unstable interface; actual MSRV compilation has no bypass.
Clean standard packaging, source/lock checks, independent review and hosted
evidence must be associated with the final head in the PR/review packet before
any publication. No favorable timing run, benchmark workload, assertion or
acceptance threshold may be substituted.

After accepted publication, downstream release graphs must select the repaired
version and exclude defective 0.46.2 where a repaired-floor guarantee is needed.
Athena #126 remains unmerged until that graph and material delta review clear.
Harmonia's measured performance hold and the remaining owner/service/release
holds remain independent. The slower Harmonia extraction is not adopted.

## Performance controls and remaining holds

The before/after kernels use identical locks and unchanged source fixtures.
Serial ABBA runs retain all raw Criterion samples and confidence intervals.
The kernel source configures 10 samples; a second explicit CLI profile raises
this to 100 without editing the fixture. Application controls use the unchanged
Athena #126 CG/GMRES Poisson grids, restart 50, tolerances and default 100 samples.
Changes below compare the mean of two run medians, not an acceptance threshold.

| Control | Samples/run | Before medians | Candidate medians | Change |
| --- | ---: | --- | --- | ---: |
| banded7/4096 | 10 | 11.128, 11.094 us | 24.524, 11.342 us | +61.40% |
| banded7/65536 | 10 | 179.955, 179.132 us | 181.200, 181.055 us | +0.88% |
| banded7/1048576 | 10 | 2872.380, 2879.619 us | 2897.613, 2920.287 us | +1.15% |
| banded7/4096 | 100 | 11.060, 10.933 us | 11.210, 11.141 us | +1.63% |
| banded7/65536 | 100 | 179.433, 176.850 us | 180.732, 178.872 us | +0.93% |
| banded7/1048576 | 100 | 2876.760, 2834.312 us | 2894.615, 2863.312 us | +0.82% |
| CG/64 | 100 | 1.4203, 1.4280 ms | 1.5357, 1.4207 ms | +3.80% |
| CG/128 | 100 | 11.4294, 11.3938 ms | 11.5378, 11.4711 ms | +0.81% |
| GMRES/64 | 100 | 8.6833, 8.9412 ms | 8.8672, 8.8188 ms | +0.35% |
| GMRES/128 | 100 | 252.6709, 251.5361 ms | 250.5760, 253.3067 ms | -0.06% |

All application samples converge. Untimed CG/GMRES reports match exactly
between variants and meet independently replayed stored-order true-residual
thresholds. Book examples test and build; the pinned benchmark-runner tests pass. The first 10-sample candidate small-kernel
run is markedly slower than its repeat; its cause is unverified and the run is
retained. The 100-sample kernel controls show a smaller measured cost, and CG 64
also varies between candidate runs. This shared-host evidence neither establishes
a universal performance bound nor grants performance acceptance. Independent
review must assess the measured cost and anomaly before release.

The complete existing workspace fails a fresh advisory check on private
PyO3 0.23.5: RUSTSEC-2025-0020 and RUSTSEC-2026-0177. The same dependencies and
failures predate this runtime fix. Separate all-feature checks of each native
publishable graph pass advisories and source checks; neither graph contains
PyO3/NumPy. No advisory is ignored and no policy is changed. The private Python
package remains publish=false at its existing version; its upgrade remains
with the separate common-FFI owner and does not gain acceptance from this patch.

## Fixed-group loop refinement

The initial numerical contract is independently accepted at the original
candidate; its measured guard cost remains a performance hold. The refinement
keeps the same four native chains, zero additions, tail-to-lane-zero updates,
final reduction tree and conditional stored-order fold. For each group k and
lane j, the update still uses stored index 4k+j. CsrView row slices give equal
value/index lengths; borrowed arrays of four express within-group bounds to the
compiler. No new allocation, precision conversion or input/shape rule is added.

The controlled experiment covers all eight combinations of indexed/grouped
lanes, fold/explicit-loop recovery and cold hints, plus unrepaired and retained
original-binary controls. All 165,120 native cases preserve the accepted finite
bits, recovery and nonfinite outcomes; 640,000 measured calls retain strict
zero-allocation windows with positive controls. Two analytical rounding tests
also pass before/after: (1+eps)*(1-eps) rounds to one before addition, so a lane
starting at minus one returns zero; fused/widened arithmetic would keep -eps².

A cold hint alone outlines the iterator fold and increases the recorded stack
frame from 96 to 176 bytes on the measured build. Grouped loops are consistently
faster in exploration. The selected source uses grouping with the original fold
and no cold hint. Stack sampling attributes both retained/selected runs to the
actual CSR kernel; profiling samples are not benchmark acceptance timings.

Eight repetitions of the exact original binary use the original 10-sample
profile and all three sizes. The small-kernel medians span 11.267–11.534us; the
historical whole 24.453–24.594us run is not reproduced. It is retained in full.
Historical core/frequency/thermal telemetry is unavailable, so no causal claim
or outlier removal follows. Shared-host sensitivity remains a measurement limit.

Fresh confirmation uses the unchanged kernels and real Athena Poisson sources,
identical paired locks, all 100 samples and balanced ABBA processes. The latest
provisional Athena manifests admit public Hephaestus 0.23; all 132 Athena Rust
files match the reviewed numerical repair. Both sides use that same graph;
absolute times must not be compared across different campaigns/cohorts.

| Confirmation | Before run medians | Selected run medians | Change |
| --- | --- | --- | ---: |
| kernel/4096 vs unrepaired | 11.300, 11.387 us | 9.864, 9.901 us | -12.88% |
| kernel/65536 vs unrepaired | 182.038, 182.614 us | 159.553, 160.711 us | -12.17% |
| kernel/1048576 vs unrepaired | 2930.657, 2926.948 us | 2588.740, 2558.896 us | -12.12% |
| kernel/4096 vs accepted-original | 11.437, 11.517 us | 9.907, 9.852 us | -13.92% |
| kernel/65536 vs accepted-original | 184.663, 186.760 us | 161.145, 161.374 us | -13.17% |
| kernel/1048576 vs accepted-original | 2963.052, 2985.271 us | 2578.666, 2586.990 us | -13.16% |
| CG/64 | 1.624, 1.633 ms | 1.470, 1.466 ms | -9.85% |
| CG/128 | 13.177, 13.230 ms | 11.842, 11.855 ms | -10.26% |
| GMRES/64 | 9.353, 9.643 ms | 9.312, 9.127 ms | -2.93% |
| GMRES/128 | 264.017, 266.351 ms | 257.287, 257.366 ms | -2.96% |

These bounded controls show no remaining net slowdown. They neither create a
new acceptance threshold nor infer f32/fallback-heavy/other-shape throughput,
universal performance, or permission to merge/publish. The changed head needs
fresh independent numerical/performance review and applicable hosted gates.
All earlier source receipts, raw timing datasets and release holds remain.
