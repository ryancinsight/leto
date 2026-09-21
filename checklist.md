# Leto Development Checklist

## ATLAS-LETO-HERMES-COMPLEX-TRANSPOSE-2026-09-01 [minor, perf] — Codex

- [x] Attribute Apollo's 3-D transpose phase with a pinned value-checked
      instrument and reject the Apollo-owned implementation under ADR 0040.
- [x] Add the Leto-owned checked complex matrix-batch operation with Hermes
      register tiles and a generic allocation-free fallback.
- [x] Prove f32/f64 full, ragged, asymmetric, empty, invalid-length, capability,
      and warm zero-allocation behavior; run cross-target warning-denied gates.
- [ ] Bind Apollo to the exact provider revision, delete the consumer-local
      experiment, and reproduce the allocation and performance evidence.
- [ ] Complete independent review, hosted gates, provider-first merge, and
      Apollo consumer delivery.

## LETO-CRATES-METADATA-1 [patch] — Owner: Codex

- [x] Add precise crates.io descriptions to both publishable packages.
- [x] Confirm the descriptions, repository metadata, and current workspace
      version `0.42.0` through locked no-dependency metadata inspection.
- [ ] Pass locked package dry runs and exact-head hosted gates; merge.
- [ ] Publish, index, and configure trusted publishing under the release
      authority; this task does not authorize registry publication.

**Current blocker:** `cargo package --locked` from the Atlas umbrella resolves
the development overlay and requests a lockfile rewrite. The offline package
command succeeds only by removing git source identities and adding
`[patch.unused]` entries, so that generated lockfile is not a valid release
proof and was discarded. Re-open when a standalone locked package job runs
without the umbrella overlay; then collect exact-head hosted verification.

## LETO-ATTENTION-GROUPED-MASK-001 [minor, arch] — Owner: Codex

- [x] Add a borrowed grouped keep-mask policy with a nonzero batches-per-group
      contract to the canonical rank-3 attention API.
- [x] Validate the complete grouped request before output mutation and map each
      execution batch to its mask group without materialization.
- [x] Add value-semantic grouped, causal-grouped, and failure-atomic coverage.
- [x] Update ADR-0002 and pass focused warning-denied and Nextest gates.

**Evidence:** all-target warning-denied Clippy passes; focused attention Nextest
passes 17/17, including grouped, causal-grouped, and later-group non-finite mask
failure atomicity. Independent review identified and closed the original
query-poisoned test oracle before delivery. Full-package Nextest collection was
blocked before execution by concurrent shared-target compilation and remains a
hosted merge gate.

## LETO-MATMUL-PERF-1 [minor] — Owner: Codex `/root` (complete)

- [x] Deliver the deferred topology policy as a bounded provider contract:
      `MatmulTilePolicy` selects only existing row-block monomorphizations,
      preserves the common 32-row route, and has fallback/tiny-cache tests.
      This closes implementation delivery without claiming benchmark improvement.
- [x] Run the required alternating-order, checksum-consuming,
      value-preserving policy comparison for the generic fallback. On the
      detected 3 MiB-L2 host, automatic 16-row blocking measured `329.68 µs`
      [327.85–333.38] and fixed 32-row blocking measured `352.80 µs`
      [305.31–419.24] for the wide strided case. The intervals overlap
      substantially, so the result is inconclusive. Production retains its
      existing fixed 32-row route.

- [x] Establish a quiet-host, counterbalanced dense matmul baseline against
      ndarray at 64×64, 128×128, and 256×256 before changing production code.
- [x] Profile the current row/block/column kernel and cache-topology decision;
      reject any tile, packing, dispatch, or allocation change without a
      statistically significant value-preserving improvement.
- [x] Add common dense C×C route evidence: remove the legacy `serial_cc_matmul`
      / `parallel_cc_matmul` bypass, route dense C×C through the policy-aware
      row-block kernel, and differentially verify fixed-1/fixed-32 output
      equality at 64×64. Record route medians separately from policy-ranking
      evidence; no adaptive speedup claim is made.
- [x] If the profile identifies a complete provider-owned fix, implement it
      in the canonical matmul module with differential tests and synchronized
      benchmark/PM evidence; otherwise close this item as an evidence-only
      audit with the measured blocker and no speculative rewrite.

**Evidence:** current default-feature release measurements report Leto versus
ndarray medians of `23.597/12.770 µs` (64×64), `123.63/113.07 µs`
(128×128), and `233.60/952.54 µs` (256×256). A focused threshold rerun reports
parallel `23.597 µs` versus serial `27.483 µs` at 64×64; serial is also
`223.69 µs` and `1.8522 ms` at 128×128 and 256×256. Flamegraph collection was
blocked by missing Windows `dtrace` and administrator-only `blondie`. No
production change is justified; future matmul work requires a working profile.

## LETO-PYTHON-RELEASE-1 [patch] — Owner: Codex `/root`

- [x] Add the pinned build-once GitHub Release and PyPI workflow.
- [x] Document the `leto-python` distribution, `leto_python` import, Cargo
      version source, supported CPython range, and OIDC publication contract.
- [x] Build, install, import, and inspect a production CPython 3.13 wheel
      locally as the historical `leto-python` 0.39.0 / `leto_python` artifact.
- [x] Reconcile the release tracking against the current workspace version
      `0.42.0`; the 0.39.0 wheel remains historical evidence only.
- [x] Create the protected `pypi` environment restricted to
      `leto-python-v*` tags.
- [ ] Pass hosted CI on the exact release-automation head.
- [ ] Register the PyPI pending trusted publisher.

## Next increments (ordered)

- [ ] `LETO-FFT-LAYOUT-THROUGHPUT` (owner: Codex
  `01a0253c-6013-7552-99cc-36bbbcf77f6d`): preserve the exact Hermes provider
  pin; measure Apollo's 2-D/3-D tiled gather/scatter boundary; implement and
  benchmark the justified caller-owned Leto permutation path; migrate Apollo;
  run value, differential, allocation, performance, documentation, and
  consumer gates.
- [x] [minor] Contiguous-slice view access (`as_slice`/`as_mut_slice` now offset-independent C-dense, `as_slice_memory_order`/`as_mut_slice_memory_order`, `is_c_contiguous`/`is_f_contiguous`/`is_contiguous` queries) — unblocks Apollo FFT hot kernels. Value tests: offset-contiguous subview, F-order block, strided-gap rejection, mutable offset-block write.
- [x] [patch] `map_inplace` (mapv_inplace analogue) and 1D `dot` (contiguous + strided). Value tests in `ops/unary_math.rs`.
- [x] [major] ADR: const-rank vs dynamic-rank boundary for Coeus integration — `docs/adr/0002-coeus-rank-boundary.md` (const-generic dispatch shim at the Coeus boundary; Leto stays const-rank).
- [x] [minor] Unary math-op ZST suite (`ExpOp`/`LnOp`/`SinOp`/`CosOp`/`SqrtOp`/`AbsOp`/`NegOp`/`RecipOp`/`PowfOp`) via `UnaryOp` + `unary_map`/`unary_map_into`, on the new segregated `RealScalar` trait. Routed through the existing traversal kernel.
- [x] [minor] `scalar_map`/`scalar_map_into` array–scalar arithmetic reusing `BinaryOp` markers.
- [x] [minor] Generalize `symmetric_eigen_jacobi` over `T: RealScalar` (native precision, no hidden widening). f32 genericity test added; f64 path unchanged.
- [x] [minor] Add `symmetric_eigenvalues_jacobi` for sorted eigenvalues without eigenvector allocation; implemented with a ZST no-vector rotation target and shared Jacobi diagonalization kernel.
- [x] [arch] std::ops operator overloading decision — `docs/adr/0001-elementwise-operator-overloading.md` (deferred; orphan rule; `scalar_map` covers the scalar case).
- [x] [minor] Broadcast-aware binary ops into caller-owned output layouts: `binary_map`/`add`/`sub`/`mul`/`div` broadcast each input to the output shape, preserve the equal-shape contiguous fast path, and reject zero-stride aliased mutable output layouts. Value tests cover dense and strided broadcast inputs; ndarray differential coverage validates broadcasted add.
- [x] [minor] `reshape`/`permute`/`to_contiguous`: dense row-major reshape/into_shape on layouts, arrays, and views; permute aliases over transpose; row-major materialization for strided/transposed/broadcasted arrays and views. Value tests and ndarray contract coverage added.
- [x] [minor] `concat`/`pad`/`split` (leto core `structure/`), batched rank-3 `matmul`, `cumsum`/`scan_axis`, seeded RNG (`uniform_with_seed`/`normal_with_seed`), and `zip2_mut_with` (3-operand). Value tests for each; RNG validated against closed-form mean/variance. `stack` deferred (needs `InsertAxis` rank helper — stable Rust lacks const-generic `N+1`).
- [x] [minor] `stack` via an `InsertAxis` rank helper mirroring `RemoveAxis` (rank `N -> N+1`, ranks 0..=7). Value tests: new leading/trailing axis, rank-2→3, transposed-input logical order, shape-mismatch rejection.
- [x] [patch] Leto-internal ndarray differential coverage for the new ops: `unary_map` (exp/sqrt), `scalar_map`, `concat`, `stack`, `batched_matmul` (per-batch ndarray dot), and `cumsum` (reference accumulate). `ops_tests` differential suite now 57 green.
- [x] [minor] Indexed zip parity: `indexed_zip_mut_with` and `indexed_zip2_mut_with` pass logical row-major `[usize; N]` coordinates into zip closures while preserving zero-copy view traversal and mutable-output alias rejection.
- [x] [arch] Push Leto rev 9d5a2bf (0.7.0) and verify consumers: Apollo (already pinned at 9d5a2bf) builds clean — `apollo-frft`/`apollo-gft` eigensolver consumers check green against the generic eigensolver. Coeus integration started — new `coeus-leto` const-rank dispatch shim (ADR 0002) committed+pushed (coeus cdaaeb9) with 6 cross-repo contract tests; leto/leto-ops pinned at 9d5a2bf.
- [x] [arch] Coeus consolidation: COMPLETE (verified 2026-06-15 against coeus
  HEAD `037fdd5`). coeus's CPU `BackendOps` (elementwise binary/unary, matmul,
  batched matmul, axis reductions, argmax/argmin, cumsum/suffix scan,
  concat/pad/split/stack, seeded RNG, to_contiguous/reshape/permute,
  cross-backend transfer, from_fn/eye/arange/linspace) all route through the
  `coeus-leto` const-rank dispatch shim (ADR 0002) into leto/leto-ops kernels,
  with cross-repo contract tests (`coeus-leto/tests/contract.rs`) and per-op
  differential tests (`coeus-ops/tests/*_leto_diff.rs`); coeus workspace 255
  tests green. Framing correction: `coeus-tensor` is NOT a duplicated layout
  layer to retire — it is the autodiff-integrated `Tensor`/COW wrapper over
  coeus-core's dynamic-rank layout, with CPU compute delegated to leto. The
  array-primitive duplication is what was retired (routed to coeus-leto); the
  tensor/autograd wrapper legitimately remains coeus-owned. Coeus keeps NN
  orchestration, optimizers, and higher sparse formats; Leto owns CPU attention
  and narrow CPU sparse parity kernels, while Hephaestus owns accelerator
  attention. No Leto-side capability gap remains for the CPU re-base.
- [x] [minor] Apollo internal FFT-kernel migration off ndarray using Leto's
  memory-order slice access. Apollo commit `324f380` exposes native Leto arrays
  across its transform families; its manifests and resolved Rust graph contain
  no `ndarray` or `ndarray-compat` edge.
- [ ] [arch] Stack-wide themis-0.9 re-pin cascade (downstream-blocking,
  meta/stack-owned). All leaf upstreams are pushed on themis-0.9
  (themis `7c38eb2` 0.9.11; mnemosyne `0174b80`; moirai `4aa94f1`; hermes
  `e6761ac` 0.9.9), and apollo already migrated. leto cannot move unilaterally:
  fresh `cargo generate-lockfile` fails because the pinned upstream revs
  cross-reference each other's OLD (themis `^0.8.0`) revs — e.g. hermes `e6761ac`
  still pins mnemosyne `1e014d25`. Resolution must re-pin + re-push in dependency
  order (themis → mnemosyne → moirai/hermes → leto → apollo/coeus); apollo only
  builds on 0.9.11 today via local path-patches that bypass the git revs. Until
  then leto stays on the themis-0.8.0 lock (`--locked` builds/tests pass;
  consumer rev-bumps to leto 0.24.0 wait on this cascade). See gap_audit §D.
- [x] [patch] Current Leto 0.5.0 artifact verification: `cargo fmt --check`; `cargo test --all-features`; `cargo clippy --all-targets --all-features -- -D warnings`; `cargo doc --workspace --exclude leto-python --all-features --no-deps`. Historical note: full workspace docs were previously blocked by the tracked `numpy 0.23`/rustdoc ICE in `leto-python`; 0.19.6 updates the Python FFI dependencies and rechecks full docs.
- [x] [patch] Add ndarray/nalgebra oracle validation gates for current linalg
  and reduction contracts. Verification: `oracle_parity` compares Leto LU,
  Cholesky, symmetric eigenvalues, singular values, and reverse reductions
  against nalgebra/ndarray with value-semantic assertions. Gates run:
  `cargo fmt --check`; `cargo test -p leto-ops --test ops_tests oracle_parity
  --all-features`; `cargo check -p leto-ops --benches --all-features`;
  `cargo clippy --workspace --all-targets --all-features -- -D warnings`;
  `cargo test --workspace --all-features`; `cargo nextest run --workspace
  --all-features`; `cargo doc --workspace --exclude leto-python
  --all-features --no-deps`; `cargo test --doc --workspace --all-features`;
  `git diff --check`.
- [x] [minor] Close dense matmul oracle performance gap: the historical 0.19.7
  gate consumed
  Hermes fused multi-row AXPY and improves Leto medians to 17.430 µs
  (64x64), 108.98 µs (128x128), and 1.0631 ms (256x256), but ndarray/nalgebra
  remain faster at 8.492/8.775 µs, 66.527/62.935 µs, and 495.95/505.35 µs.
  Rejected: removing the dense row-block zero-skip branch, RHS-column packing
  plus `Scalar::dot_slice`, replacing Hermes AXPY with a generic scalar row
  update, existing Hermes `tiled_gemm` for f64 dense matmul, reducing parallel
  row-block scheduling for small dense matrices, `MATMUL_ROW_BLOCK=16`, and
  first-shared-row output initialization. Rejected after 0.19.7:
  Hermes column-chunk `axpy_rows`, `MATMUL_ROW_BLOCK=64`, and row-block
  fused-branch/alpha-buffer hoisting, and generic 4x4 registered dense tiles.
  Added after 0.19.7: `hermes_simd::axpy_rows_batch` is consumed only for the
  measured 128-row dense regime (212.64 µs → 98.853 µs on the local themis-0.9
  stack); broad depth-batched routing was rejected after 64x64/256x256
  regression.
  Current corrective gate: `cargo fmt --check`; `cargo clippy --workspace
  --all-targets --all-features -- -D warnings`; `cargo test --workspace
  --all-features`; `cargo nextest run --workspace --all-features`; `cargo doc
  -p leto -p leto-ops --all-features --no-deps`; `git diff --check`. Full
  workspace docs are no longer blocked by the reopened `leto-python`/`numpy
  0.23` rustdoc ICE after the 2026-07-05 PyO3-extension doc-target exclusion.
  The 2026-07-23 current-build audit closes this evidence item without a
  production change: default-feature Leto measures 23.597/123.63/233.60 µs
  against ndarray 12.770/113.07/952.54 µs at 64/128/256; disabling parallelism
  measures 27.483/223.69/1.8522 ms. Flamegraph collection is blocked on this
  Windows session by missing dtrace and administrator-only blondie. A future
  kernel increment should target an allocation-controlled
  reusable packing scratch or a verified external micro-kernel provider with
  profile evidence.
- [x] [patch] Direct registry dependencies were audited and later aligned with
  the current NumPy FFI constraint: workspace manifests now use `ndarray` 0.16,
  `pyo3` 0.23, and `numpy` 0.23. The reopened `leto-python` rustdoc ICE is
  resolved without moving the FFI constraint: `leto-python` is no longer a Rust
  doc target, while full workspace docs still build. Full Git dependency update
  is still blocked upstream by
  Mnemosyne's `themis ^0.8.0` requirement vs Themis main 0.9.5.
