# Changelog

All notable changes to Leto are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
SemVer 2.0.0. Pre-1.0 minor bumps may include additive API surface.

## [0.27.0] - 2026-06-15

### Added

- Zero-copy sliding-window iteration (`Array::windows`/`ArrayView::windows` →
  `Windows`) in a new `application/iter/windows.rs` leaf (ndarray `windows`
  parity). Yields every `ArrayView<'a, T, N>` of a fixed window shape by sliding
  one step per axis; each window reuses the parent's strides and only shifts the
  offset (no element read or copy, overlapping windows share storage via shared
  borrows). Documents the window-count theorem `∏ᵢ (sᵢ − wᵢ + 1)` with proof.
  `DoubleEndedIterator` + `ExactSizeIterator` over a linear window-start counter
  decoded with `index_from_flat` (SSOT). Verified: count theorem across window
  shapes, row-major window content, full-window-equals-original, transposed/
  strided zero-copy correctness, double-ended meet-once, and zero/oversize-extent
  rejection.

### Validation

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p leto --test core_tests windows --all-features` (6 tests)

## [0.26.0] - 2026-06-15

### Added

- Logical-order element iteration (`Array::iter`/`indexed_iter`,
  `ArrayView::iter`/`indexed_iter` → `ElementIter`/`IndexedIter`) and
  `IntoIterator for &ArrayView` (ndarray `iter`/`indexed_iter` parity).
  `ElementIter` yields `&T`; `IndexedIter` yields `([usize; N], &T)`. Both walk
  every logical element in row-major order through the view's strides, so a
  transposed/strided/broadcast view iterates in the same logical order as its
  contiguous materialization. Both are `DoubleEndedIterator` (shared
  `[front, back)` cursor) and `ExactSizeIterator`; `fold`/`rev`/`map` etc. come
  free from the std `Iterator` blanket surface. A shared `elem_at` helper resolves
  offsets (SSOT). Verified: row-major order, transposed logical order, indexed
  pairs, double-ended meet-once + `rev`-equals-reverse, `&view` for-loop, and
  empty arrays.

### Changed

- Refactored the single `application/iter.rs` into a vertical `application/iter/`
  leaf hierarchy (`axis.rs`, `element.rs`, `mod.rs`) by iteration concern. The
  public paths (`leto::AxisIter`/`AxisIterMut`, `leto::application::iter::*`) are
  unchanged; all existing consumers compile and pass without edits.

### Validation

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p leto` (23 lib + 86 core_tests + others)
- `cargo test -p leto-ops` (156 ops_tests + others, AxisIter consumers green)

## [0.25.0] - 2026-06-15

### Added

- Whole-array argmin/argmax (`argmin_all`, `argmax_all`) in `leto` core
  `application/reduction/min_max.rs` (ndarray-stats `argmin`/`argmax` parity),
  returning the N-dimensional index `[usize; N]` of the global extremum (const-
  generic rank, so the multi-index is statically sized). Scans logical row-major
  order; first occurrence wins on ties. A single `arg_reduce_all` kernel backs
  both via a strict-comparison predicate (SSOT, mirroring the existing axis
  `axis_arg_reduce`). Verified by rank-1/rank-2 multi-index oracles, first-
  occurrence tie-break, a value-agrees-with-`min_all`/`max_all` cross-check, and
  empty-array rejection. This promotes the argmin/argmax parity row to Verified.

### Validation

- `cargo fmt --check`
- `cargo clippy -p leto --all-targets --all-features -- -D warnings`
- `cargo test -p leto --lib reduction::tests --all-features` (23 tests)

## [0.24.0] - 2026-06-15

### Added

- Multivariate covariance and Pearson correlation (`covariance`,
  `pearson_correlation`) in a new `leto` core `application/statistics/`
  bounded context (ndarray-stats `cov` / `pearson_correlation` parity). Both
  follow the numpy/ndarray-stats `rowvar = true` convention (rows are variables,
  columns observations) and return the symmetric `v × v` summary matrix.
  `covariance` is two-pass numerically stable (variables centered before the
  cross-products) and reuses the `degrees_of_freedom` contract shared with the
  variance reductions (SSOT); `pearson_correlation` delegates to `covariance`
  (SSOT) and documents the ddof-invariance theorem and the constant-variable
  NaN contract. Verified by closed-form sample/population oracles, a
  diagonal == `var_axis` cross-check, symmetry, perfect ±1 correlation cases,
  the `R = C / (σσ)` normalization identity with the `|R| ≤ 1` bound, and
  empty/excess-ddof rejection.

### Validation

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo nextest run --workspace --all-features` (295 tests)
- `cargo test -p leto --test core_tests statistics --all-features` (7 tests)
- `cargo test -p leto --test core_tests quantile --all-features` (7 tests)
- `cargo test -p leto --test core_tests variance --all-features` (10 filtered tests;
  includes covariance cross-checks)
- `cargo test --doc --workspace --all-features` (5 doctests)
- `cargo doc -p leto -p leto-ops --all-features --no-deps`
- `git diff --check`

## [0.23.0] - 2026-06-15

### Added

- Quantile and median reductions (`quantile_all`/`median_all`/`quantile_axis`/
  `median_axis`) with an `Interpolation` strategy enum (Linear/Lower/Higher/
  Nearest/Midpoint) in `leto` core `application/reduction/quantile.rs`
  (ndarray-stats / numpy parity). Generic over `num_traits::Float`; documents the
  fractional-rank `h = q·(n−1)` theorem. A single shared `quantile_of_slice`
  kernel backs both the whole-array and per-axis paths (SSOT); the axis path
  reuses one `out_size × axis_len` scratch buffer across all lanes. Verified by
  closed-form analytical oracles for every interpolation method, per-lane
  equivalence between `quantile_axis` and `quantile_all`, and rejection of
  empty input, out-of-range `q`, and NaN data.

### Changed

- `var_axis` no longer allocates a redundant per-output gather buffer: it indexes
  the C-contiguous `mean_axis` result directly, removing one `Vec` allocation per
  call with no change in results.

### Validation

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo nextest run --workspace --all-features` (288 tests)
- `cargo test -p leto --test core_tests quantile --all-features` (7 tests)
- `cargo test -p leto --test core_tests variance --all-features` (5 tests)
- `cargo test --doc --workspace --all-features` (5 doctests)
- `cargo doc -p leto -p leto-ops --all-features --no-deps`
- `git diff --check`

## [0.22.0] - 2026-06-15

### Added

- Variance and standard-deviation reductions (`var_all`/`std_all`/`var_axis`/
  `std_axis`) in `leto` core `application/reduction/variance.rs` (ndarray-stats
  parity), generic over `num_traits::Float`, with `ddof` (population/sample) and
  the numerically-stable two-pass theorem in rustdoc. Axis variants reduce rank
  by one via the shared `AxisIter`/`RemoveAxis` machinery (SSOT). Verified by
  closed-form references, invalid `ddof` rejection, and an ndarray
  `var`/`std`/`var_axis` differential.

### Validation

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo nextest run --workspace --all-features` (281 tests)
- `cargo test -p leto --test core_tests variance --all-features`
- `cargo test --doc --workspace --all-features` (5 doctests)
- `cargo doc -p leto -p leto-ops --all-features --no-deps`
- `git diff --check`

## [0.21.0] - 2026-06-15

### Added

- Symmetric indefinite unpivoted `U D Uᵀ` factorization (`udu_decompose`,
  `MatrixDecompose::udu`, `UduDecomposition`) in
  `linalg/udu/{mod,decompose,solve}.rs`. The module documents the constructive
  UDU theorem, determinant corollary, and triangular-solve contract; it exposes
  `u`, `diagonal`, `det`, `solve`, and `inv`. Verification covers
  reconstruction `A = U D Uᵀ`, determinant parity with nalgebra, solve/inverse
  parity with nalgebra, and non-square/nonsymmetric/zero-pivot rejection.

### Changed

- Reconciled linalg PM artifacts: rank-revealing SVD, rank-deficient
  pseudoinverse, non-symmetric eigenvalues, full-pivot LU, column-pivoted QR,
  Hessenberg, trace, rank, and Kronecker are tracked as delivered surfaces;
  remaining nalgebra gaps are Schur vectors, pivoted symmetric-indefinite
  factorization, matrix functions, and consumer-driven fixed-size/geometry
  decisions.

### Validation

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo nextest run --workspace --all-features` (276 tests)
- `cargo test -p leto-ops --test ops_tests udu --all-features`
- `cargo test --doc --workspace --all-features` (5 doctests)
- `cargo doc -p leto -p leto-ops --all-features --no-deps`
- `git diff --check`

## [0.20.0] - 2026-06-15

### Added

- QR with column pivoting `A P = Q R` (`col_piv_qr`,
  `MatrixDecompose::col_piv_qr`) in `linalg/col_piv_qr/{mod,decompose}.rs` on the
  shared Householder primitive (DRY), with the column-pivoted-QR theorem + rank
  corollary. Rank-revealing (pivots the largest-tail-norm column, so the `R`
  diagonal is non-increasing); exposes `q`/`r`/`permutation`/`rank` and a
  full-column-rank `solve_least_squares`. Verified: reconstruction `A P = Q R`,
  `Q` orthogonality, `R` upper-triangular, least squares vs leto's QR solver and
  nalgebra normal equations, and rank-deficiency revelation.
- LU with complete (full) pivoting `P A Q = L U` (`full_piv_lu`,
  `MatrixDecompose::full_piv_lu`) in `linalg/full_piv_lu/{mod,decompose,solve}.rs`
  with the complete-pivoting existence theorem + rank/determinant corollary.
  Rank-revealing (orders pivots by decreasing magnitude) and maximally stable;
  exposes `rank`/`det`/`l`/`u`/permutations/`solve`/`inv`. Verified against
  nalgebra `FullPivLU` (det/solve/inverse), reconstruction `P A Q = L U`, and
  robust rank-deficiency revelation (where it correctly reports rank 2 on a case
  the Gram-spectrum `matrix_rank` inflates).
- Golub–Kahan bidiagonalization `A = U B Vᵀ` (`bidiagonalize`,
  `MatrixDecompose::bidiagonalize`; ADR 0006, `m ≥ n`) — the classical SVD-prep
  reduction, in `linalg/bidiagonal/{mod,reduce}.rs` with the reduction theorem +
  singular-value-preservation corollary. Introduced a **shared Householder
  reflector primitive** (`linalg/householder.rs`, SSOT) and refactored the
  Hessenberg reduction onto it (DRY; no duplicated reflector code; Hessenberg
  tests unchanged-green). Verified on the convention-independent contract
  (reconstruction, `U`/`V` orthogonality, upper-bidiagonal structure) plus
  singular-value preservation vs both leto's `singular_values` and nalgebra's SVD.
- Non-symmetric eigenvalues, real and complex (`eigenvalues`,
  `MatrixDecompose::eigenvalues` → `Vec<num_complex::Complex<T>>`; ADR 0006
  Phase 2). Hessenberg-reduce (reused — SSOT) then a **single-shift Wilkinson
  complex QR iteration** with one-eigenvalue-at-a-time deflation and exceptional
  shifts, in a leaf hierarchy `linalg/eigenvalues/{mod,complex,qr}.rs`. A
  compute-local `Cplx<T>` provides complex arithmetic over `RealScalar` (whose
  sealed bound `num_complex`'s operators can't use); the complex Givens rotation
  is derived in rustdoc and unit-tested in isolation (zeroes the off-element,
  unitary). The Schur-form theorem + the QR-similarity argument are documented.
  Verified against a nalgebra `complex_eigenvalues` battery (real/complex spectra,
  sizes 2–5) plus exact known spectra (diagonal, `1±i`, `±i`) and symmetric
  all-real agreement. (Schur *vectors* `Q`/`T` remain a follow-up.)
- Upper Hessenberg reduction `A = Q H Qᵀ` by Householder reflectors
  (`hessenberg`, `MatrixDecompose::hessenberg`; ADR 0006) in a
  leaf hierarchy `linalg/hessenberg/{mod,householder,reduce}.rs` with the
  reduction theorem + spectrum-preservation corollary in rustdoc. Generic over
  `RealScalar`, native precision. Verified on the convention-independent
  contract (reconstruction, `Q` orthogonality, upper-Hessenberg structure,
  symmetric→tridiagonal), orthogonal-similarity invariants (trace, Frobenius),
  and nalgebra Frobenius parity. (The Francis double-shift QR for the real Schur
  form / non-symmetric eigenvalues builds on this in the next phase.)
- Rank-revealing SVD via one-sided Jacobi (`svd_rank_revealing` (+`_with_tolerance`),
  `MatrixDecompose::svd_rank_revealing`; ADR 0005). Unlike the Gram-matrix
  `svd_decompose`, it **accepts rank-deficient input** (surfaces zero singular
  values, keeps `V` fully orthonormal) and never forms `AᵀA` (no condition-number
  squaring). `linalg/svd.rs` refactored into a leaf hierarchy
  (`svd/{mod,gram,jacobi,pseudoinverse}.rs`) with a monotone-convergence proof
  sketch in rustdoc. `pinv` is now unified onto this path, so it handles
  **rank-deficient** matrices too (`A⁺ = Σ_{σᵢ>τ} σᵢ⁻¹ vᵢ uᵢᵀ`). Differential vs
  nalgebra `SVD`/`pseudo_inverse` across tall/wide/deficient shapes, plus
  reconstruction, orthonormality, and both Moore-Penrose identities.
- Matrix `trace`, numerical `matrix_rank` (+ `_with_tolerance`), and Kronecker
  product `kron`, in a vertical leaf-module hierarchy (`linalg/properties/{trace,
  rank}.rs`, `linalg/products/kronecker.rs`) with theorem/proof rustdoc.
  `trace` is `Scalar`-generic (integers included); `rank` is SVD-spectrum-based
  (SSOT delegation to `singular_values`, no second SVD path). Exposed both as
  free functions and via fluent traits `MatrixProperties` (`trace`/`rank`/
  `rank_with_tolerance`) and `MatrixProduct::kron`. Differential tests vs
  nalgebra `trace`/`rank`/`kronecker` plus oracle-independent identities
  (Kronecker mixed-product `(A⊗B)(C⊗D)=(AC)⊗(BD)`, `tr(A⊗B)=tr(A)·tr(B)`).
- Moore-Penrose pseudoinverse `pinv` (free function + `MatrixSolve::pinv`) via
  the rank-revealing one-sided Jacobi SVD (`A⁺ = V Σ⁺ Uᵀ`, numerically sound —
  no normal-equations condition-number squaring). Covers tall, wide, square, and
  rank-deficient inputs. Differential test vs nalgebra `pseudo_inverse` plus the
  Moore-Penrose identities `A A⁺ A = A` and `A⁺ A A⁺ = A⁺`.

- Fluent rank-2 linear-algebra trait layer over the existing strided matrix
  (`Array2`/`ArrayView2`), consolidating the ndarray "strided array" and
  nalgebra "matrix methods" models into one type (ADR 0003). Role-segmented
  traits — `MatrixProduct` (`matmul`), `MatrixNorm` (`norm_l1`/`norm_l2`
  Frobenius/`norm_max`), `MatrixDecompose` (`lu`/`qr`/`cholesky`/`svd`/
  `singular_values`/`symmetric_eigen`/`symmetric_eigenvalues`), and
  `MatrixSolve` (`solve`/`solve_least_squares`/`inv`/`det`) — are blanket-impl'd
  for any rank-2 receiver via the `AsMatrixView` bridge. Each method is a
  zero-cost delegator to the existing free-function kernel (single source of
  truth; no kernel duplicated). Operator overloading remains deferred (ADR 0001).
- Elementwise arithmetic operators on `Array` (ADR 0004, supersedes ADR 0001's
  deferral): `&a + &b`, `&a - &b`, `&a * &b`, `&a / &b` (equal-shape), `&a op
  scalar` (bounded by the new sealed `ScalarOperand` marker), and `-&a`, all in
  `leto` core as the allocating convenience tier over one shared `iter_elements`
  traversal. `*` is **elementwise** (Hadamard, ndarray semantics); matrix product
  remains the explicit `matmul` method. The leto-ops `binary_map`/`scalar_map`
  family stays the SIMD/broadcasting performance tier. Unequal-shape array
  operators panic (the sanctioned operator exception).
- Completeness program against full ndarray 0.16 / nalgebra 0.35 parity:
  `docs/completeness/PLAN.md` and `docs/completeness/parity_matrix.md`; a
  differential-correctness harness (`tests/ops/parity.rs`) and a leto-vs-ndarray
  performance comparison bench group (`bench_parity_oracle` in
  `benches/kernels.rs`).

### Validation

- `cargo fmt --check`
- `cargo clippy -p leto-ops --all-targets --all-features -- -D warnings`
- `cargo test -p leto-ops --all-features` (ops_tests 122 green)
- `cargo test --doc -p leto-ops --all-features` (4 doctests green)

## [0.19.7] - 2026-06-13

### Changed

- Pinned `hermes-simd` to pushed Hermes revision `efac045`, which exposes the
  fused multi-row AXPY dispatch used by `leto-ops` dense matmul.
- `leto-ops` dense row-blocked matmul now routes positive-stride output row
  blocks through Hermes fused multi-row AXPY, reducing per-row SIMD dispatch
  overhead while preserving zero-copy caller-owned output.

### Performance

- Criterion oracle comparison (`--sample-size 10`) improved Leto dense matmul
  medians versus 0.19.5 baselines: 64x64 21.443 µs → 17.430 µs, 128x128
  127.63 µs → 108.98 µs, and 256x256 2.4357 ms → 1.0631 ms.
- Dense matmul still does not meet replacement-performance parity:
  ndarray/nalgebra medians were 8.492/8.775 µs at 64x64, 66.527/62.935 µs at
  128x128, and 495.95/505.35 µs at 256x256.

### Validation

- `cargo test -p leto-ops matmul --all-features`
- `cargo bench -p leto-ops --bench kernels --all-features
  "oracle_compare/matmul_(leto|ndarray|nalgebra)_(64|128|256)x(64|128|256)"
  -- --sample-size 10`
- `cargo fmt --check`; `cargo clippy --workspace --all-targets
  --all-features -- -D warnings`; `cargo test --workspace --all-features`;
  `cargo nextest run --workspace --all-features`; `cargo doc --workspace
  --exclude leto-python --all-features --no-deps`; `git diff --check`

## [0.19.6] - 2026-06-13

### Changed

- Updated direct registry dependency constraints: `bytemuck` 1.25, `ndarray`
  0.17, `nalgebra` 0.35, `pyo3` 0.28, `numpy` 0.28, `proptest` 1.11, and
  `criterion` 0.8.
- Migrated the thin PyO3 binding GIL-release calls from `Python::allow_threads`
  to `Python::detach` for PyO3 0.28 compatibility.

### Validation

- Full Git dependency update remains blocked upstream: pinned Mnemosyne still
  requires `themis ^0.8.0`, while Themis main reports `0.9.5`.

## [0.19.5] - 2026-06-13

### Added

- Extended dense matmul oracle benchmarks to compare Leto, ndarray, and
  nalgebra at 64x64, 128x128, and 256x256.

### Validation

- Rejected `MATMUL_ROW_BLOCK=16`: focused matmul tests passed, but the release
  benchmark process ended with `STATUS_ACCESS_VIOLATION`.
- Rejected first-shared-row output initialization: focused matmul tests passed,
  but 64x64 dense matmul regressed to 26.807 µs median and the release
  benchmark process ended with `STATUS_ACCESS_VIOLATION`.

## [0.19.4] - 2026-06-12

### Validation

- Dense matmul investigation continued against ndarray/nalgebra. Hermes
  `tiled_gemm` integration for f64 dense row-major matmul was rejected after
  `oracle_compare/matmul_leto_128x128` regressed to 317.46 µs.
- Parallel row-block scheduling remains beneficial for current small dense
  oracle cases: all-features 128x128 median 144.15 µs vs serial-SIMD
  170.25 µs; all-features 64x64 median 21.759 µs vs serial-SIMD 23.665 µs.

## [0.19.3] - 2026-06-12

### Changed

- `leto-ops`: matmul output initialization now fills dense output storage and
  unit-stride output rows through slices before falling back to per-element
  strided writes. This removes offset recomputation from the zeroing phase for
  contiguous outputs without changing the contraction kernel or allocating
  scratch buffers.

### Validation

- Rejected two dense-matmul kernel models: RHS-column packing plus
  `Scalar::dot_slice` regressed 128x128, and replacing Hermes AXPY row updates
  with an inlined scalar loop regressed 128x128. Dense matmul remains the active
  ndarray/nalgebra parity gap.

## [0.19.2] - 2026-06-12

### Changed

- `leto-ops`: criterion benchmarks now disable plot generation and borrow the
  ndarray reverse view in oracle reduction cases, keeping long oracle runs on
  the timing path instead of the Windows plotters/view-move failure path.

### Validation

- Sequential oracle investigation confirms dense matmul remains behind
  ndarray/nalgebra at 64x64, 128x128, and 256x256. A row-block branch-removal
  experiment was rejected after the canonical dense 256x256 benchmark showed
  unstable/regressed behavior.

## [0.19.1] - 2026-06-12

### Added

- `leto-ops`: ndarray/nalgebra oracle parity tests for LU solve/determinant/
  inverse, symmetric eigenvalues, Cholesky lower factors, singular values, and
  reverse-last-axis reductions.
- `leto-ops`: criterion oracle comparison benchmarks for dense 128x128 matmul
  against ndarray/nalgebra and reverse-last-axis 256x256 reductions against
  ndarray.

### Validation

- Results parity is covered by value-semantic differential tests against
  nalgebra and ndarray.
- Performance parity is not yet satisfied for dense 128x128 matmul:
  Leto median 259.03 µs vs ndarray 114.60 µs and nalgebra 103.68 µs.
  Reverse-last-axis reductions are at parity or faster than ndarray on the
  recorded benchmark shapes.

## [0.19.0] - 2026-06-12

### Added

- `leto-ops`: `NormKind::combine` default method for combining partial
  accumulators without forcing row partials through the final norm transform.

### Changed

- `leto-ops`: whole-array `sum` and `norm` now borrow physical row slices when
  the last-axis stride is `±1`, including reverse-last-axis views. Reverse rows
  route through existing dense slice reducers with no materialized copy.

### Performance (criterion, recorded in benchmark_results.md)

- `reductions/sum_reverse_last_axis_256x256`: 5.1575-5.2534 µs
  (**−21.56% median**, p < 0.05).
- `reductions/norm_l2_reverse_last_axis_256x256`: 9.1467-9.9752 µs
  (**−18.00% median**, p < 0.05).

## [0.18.1] - 2026-06-12

### Changed

- `leto-ops`: dense `matmul` now routes unit-stride RHS/output rows through a
  const-generic row-block kernel on top of the existing Hermes AXPY row
  update. The kernel reuses each RHS row across 32 output rows and writes
  caller-owned output in place with no temporary allocation.

### Performance (criterion, recorded in benchmark_results.md)

- `matmul/dense_64x64`: 28.1 µs recorded table baseline → 22.536 µs
  current median (**~−19.8%**).
- `matmul/dense_256x256`: 1.529 ms recorded table baseline → 1.4016 ms
  current median (**~−8.3%**).

## [0.18.0] - 2026-06-12

### Added

- `leto-ops`: optional `topology` feature, direct optional `themis`
  dependency, and public `CacheGeometry`/`cache_geometry` API for reading L1,
  L2, and cache-line geometry. With `topology` enabled, L1/L2 capacities are
  selected by walking the borrowed `themis::CacheLevel` slice; without the
  feature, documented fallback constants are returned.

### Changed

- Stage C3 matmul blocking now has a typed topology source but no hot-kernel
  dispatch change. No performance claim is attached to this release; blocking
  remains gated on criterion evidence against the AXPY row kernel.

## [0.17.0] - 2026-06-12

Dense `norm_l1`/`norm_max` route through the new hermes abs-reduction
kernels, closing the last scalar-fold dense norm paths.

### Added

- `leto-ops`: `RealScalar::{abs_sum_slice, abs_max_slice}` with scalar-fold
  defaults; f32/f64 override through `SimdOperations::{abs_sum_slice,
  abs_max_slice}` → `hermes_simd::{abs_sum, abs_max}` (lane-wise `abs` fused
  into the fold, delivered hermes 7f01309).
- `leto-ops`: criterion cases `norm_l1_64k`, `norm_max_64k` plus pinned
  scalar-fold reference series preserving the pre-0.17.0 dense-path body as
  the in-run before-number.

### Changed

- `NormL1`/`NormMax` implement `NormKind::accumulate_slice`, so any dense
  memory-order view reduces through hermes instead of the element fold
  (the path `NormL2` has used since 0.11.3).

### Performance (criterion, recorded in benchmark_results.md)

- `reductions/norm_l1_64k`: 34.174 µs (scalar-fold reference) → 4.069 µs
  (**−88.1%, 8.4×**).
- `reductions/norm_max_64k`: 39.961 µs (scalar-fold reference) → 5.293 µs
  (**−86.8%, 7.5×**).

## [0.16.1] - 2026-06-12

### Changed

- `leto-ops`: `zip_mut_with` strided fallback applies the shared
  `TileGeometry`/`line_elements` cache-line micro-tiling when either operand's
  last-axis walk skips whole lines, mirroring the binary/unary map paths.
  Mixed `T`/`U` element sizes choose the smaller elements-per-line count.

### Performance (criterion, recorded in benchmark_results.md)

- `zip/zip_mut_with_transposed_256x256`: 47.6 µs → 40.7 µs (**−14.5%**,
  non-overlapping CIs). Residual vs the binary tiled map is the opaque
  closure body.

## [0.16.0] - 2026-06-12

Stage C2 closure: matmul row updates dispatch through the new Hermes AXPY
provider; sum gains the dense memory-order fast path norms already had.

### Added

- `leto-ops`: `Scalar::axpy_slice` (`out[i] += alpha * x[i]`) with
  `SimdOperations::axpy_slice` strategy routing — f32/f64 through
  `hermes_simd::axpy` (fmadd lanes, scalar tail, zero temporaries), half and
  integer types through the scalar fallback loop.

### Changed

- `leto-ops`: matmul `multiply_row` unit-stride fast path now calls
  `Scalar::axpy_slice` over the row slices instead of the per-element
  read-modify-write loop. Strided (non-unit column stride) rows keep the
  pointer walk.
- `leto-ops`: `sum` detects any dense memory-order slice
  (`as_slice_memory_order`) and feeds `T::sum_slice` directly; summation is
  logically order-independent, so memory order is a valid evaluation order
  (same justification as the 0.11.3 norm path).

### Performance (criterion, recorded in benchmark_results.md)

- `matmul/dense_256x256`: 2.210 ms → 1.529 ms median (**−31%**,
  non-overlapping CIs). `dense_64x64` 27.4 µs → 28.1 µs (within noise).
- `reductions/sum_transposed_256x256`: 44.9 µs → 4.48 µs (**−90%**), now at
  the dense-detection level of `norm_l2_transposed`.

## [0.15.0] - 2026-06-12

Minor performance increment: cache-line micro-tiling for column-walk unary
`map_into` traversal (Stage C3, atlas ADR 0002 leto slice).

### Changed

- `leto-ops`: unary `map_into` strided fallbacks (serial + parallel) now use
  the shared `TileGeometry`/`line_elements` policy when input or output
  last-axis walks skip whole cache lines. Mixed input/output scalar maps choose
  the smaller element-per-line count, preserving a single generic traversal
  without type-specific variants.
- `line_elements` now treats zero-sized element types as non-tiling operands,
  preserving generic `map_into` support for strided ZST inputs without a
  divide-by-zero path.
- `leto-ops` criterion baselines now include unary contiguous and transposed
  `map_into` cases.

### Performance (criterion, recorded in benchmark_results.md)

- `unary_map/map_into_transposed_256x256`: 57.631 µs
  (56.477–58.379 µs CI) → 35.303 µs (34.221–36.468 µs CI), **−38.7%**
  median with non-overlapping confidence intervals. Contiguous `map_into`
  remains within observed run-to-run noise; no contiguous speedup is claimed.

## [0.14.4] - 2026-06-11

Patch performance increment: cache-line micro-tiling for column-walk strided
elementwise traversal (Stage C3, atlas ADR 0002 leto slice).

### Changed

- `leto-ops`: the binary strided fallbacks (serial + parallel) micro-tile the
  last two axes at `64 / size_of::<T>()` elements per side (8 for f64; the
  tile is derived from the 64-byte cache line, not tuned) when some operand's
  last-axis walk skips whole lines (`|stride| ≥ elements-per-line`). Within a
  tile the column-strided operand revisits the same lines across tile rows,
  restoring full line utilization. Unit and reverse-unit strides already
  consume lines fully and keep the cheaper row-walk; rank < 2 keeps row-walk.
  Parallel workers own disjoint (slab, row-block) pairs, preserving the
  aliasing-rejection guarantee.
- New `TileGeometry`/`line_elements` helpers in `application/index.rs` beside
  `RowMajorTraversal` (one geometry SSOT for tiled traversal).

### Performance (criterion, recorded in benchmark_results.md)

- `elementwise_add/transposed_256x256`: 50.65 µs → 28.4 µs (**−43.5%,
  p < 0.05**); `contiguous_64k` statistically unchanged (p = 0.40). The
  strided-vs-contiguous gap closes from ~3.6× to ~1.8×; the cumulative
  improvement from the original 1.206 ms baseline is **42×**.

## [0.14.3] - 2026-06-11

Patch audit increment for Stage C2 Hermes SIMD coverage.

### Changed

- Recorded the leto-ops SIMD coverage audit in backlog/checklist artifacts.
  Current Hermes-backed coverage is dense elementwise slice arithmetic and
  dense reductions/dot/min/max through `Scalar`; matmul remains scalar because
  Hermes does not expose a zero-allocation scalar-AXPY/fused row-update API
  through the current dependency surface.

### Performance

- Rejected two measured matmul candidates instead of shipping regressions:
  const-generic dense blocking measured `64x64` at ~48.5 µs and `256x256` at
  ~3.37 ms versus the existing ~28.3 µs and ~2.25 ms baselines; a generic
  `mul_add` hook measured `64x64` at ~245.6 µs and `256x256` at ~12.5 ms.

## [0.14.2] - 2026-06-11

Patch nalgebra-parity increment.

### Changed

- `leto-ops`: `singular_values` now computes only the smaller Gram-matrix
  spectrum and returns zero singular values for rank-deficient finite inputs.
  `svd_decompose` still rejects rank-deficient matrices because full singular
  vector completion requires a rank-revealing SVD contract.

### Tests

- Added value-semantic tall and wide rank-deficient singular-value coverage,
  while preserving explicit `svd_decompose` rejection tests.

## [0.14.1] - 2026-06-11

Patch nalgebra-parity increment.

### Changed

- `leto-ops`: `svd_decompose` and `singular_values` now accept wide
  full-row-rank matrices by diagonalizing `A A^T` and deriving right singular
  vectors with `V = A^T U Σ^-1`. Tall and square inputs keep the existing
  `A^T A` path; rank-deficient inputs still reject explicitly.

### Tests

- Added value-semantic wide SVD coverage proving reconstruction, singular
  values, and right singular-vector orthonormality.

## [0.14.0] - 2026-06-11

Minor nalgebra-parity and eigensolver memory-efficiency increment.

### Added

- `leto-ops`: `symmetric_eigenvalues_jacobi` and
  `symmetric_eigenvalues_jacobi_with_tolerance` compute sorted eigenvalues for
  real symmetric matrices without allocating or rotating an eigenvector matrix.

### Changed

- All package manifests now default both `parallel` and `mnemosyne-memory`.
  `leto` maps Mnemosyne memory to the existing Mnemosyne-backed storage
  implementation, `leto-ops` forwards memory into `leto`, and `leto-python`
  forwards both provider features to its Rust dependencies.

- `leto-ops`: the symmetric Jacobi implementation now routes rotations through
  a monomorphized `RotationTarget` strategy. Full decomposition uses an
  eigenvector workspace; eigenvalues-only uses a zero-sized no-vector target.
  The numerical diagonal update is shared by both paths and remains native
  precision over `T: RealScalar`.

### Tests

- Added value-semantic coverage proving the eigenvalues-only path matches the
  full decomposition, preserves strided-view semantics, and rejects the same
  invalid inputs.
- Added Apollo migration fixture coverage for mutable rank-1 lanes sliced out
  of rank-3 Leto arrays along all three axes, matching the ndarray-free 3D FFT
  axis-pass access pattern.

## [0.13.1] - 2026-06-11

Patch performance increment completing the row-walk traversal policy: every
strided fallback in leto-ops now routes through `RowMajorTraversal` (one
offset computation per innermost row/lane, stride-increment walks) — full
SSOT for strided traversal.

### Changed

- `leto-ops`: all four zip variants (`zip_mut_with`, `zip2_mut_with`, and the
  indexed forms) row-walk their strided fallbacks. The indexed forms update
  the last logical coordinate incrementally, so closures still receive exact
  indices with the per-element div/mod decomposition gone.
- `leto-ops`: `map_inplace`'s non-dense fallback row-walks.
- `leto-ops`: `scan_axis_into` lane walks — one offset per scan lane, then
  stride increments along the scan axis (reverse scans start at the lane's
  far end and walk the negated stride), replacing per-element
  `get`/`get_mut` offset products.

### Performance (criterion, recorded in benchmark_results.md)

- New `zip/zip_mut_with_transposed_256x256` case: 553.4 µs → 55.9 µs
  (**−89.9%, 9.9×, p < 0.05**).

## [0.13.0] - 2026-06-11

Minor nalgebra-replacement increment for thin SVD.

### Added

- `leto-ops`: `SvdDecomposition`, `svd_decompose`,
  `svd_decompose_with_tolerance`, and `singular_values`.
- The initial SVD surface supports tall or square full-column-rank matrices.
  It derives the thin decomposition from `A^T A` plus the existing symmetric
  Jacobi eigensolver, returning `U`, descending singular values, and `V`.
  Wide and rank-deficient inputs are rejected explicitly.

### Tests

- Added value-semantic SVD coverage for reconstruction, closed-form diagonal
  singular values, strided input, f32 generic execution, and invalid input
  rejection.

## [0.12.0] - 2026-06-11

Minor nalgebra-replacement increment for SPD linear algebra.

### Added

- `leto-ops`: `CholeskyDecomposition::det`, `CholeskyDecomposition::inv`,
  `cholesky_solve`, `cholesky_det`, and `cholesky_inv`.
- Cholesky inverse reuses the same private forward/back substitution helper as
  `solve`, preserving one authoritative SPD solve path.

### Tests

- Added value-semantic Cholesky determinant and inverse coverage: determinant
  checks against the nalgebra oracle, convenience API parity, and `A·A⁻¹ = I`
  through Leto `matmul`.

## [0.11.3] - 2026-06-11

Patch performance increment for dense L2/Frobenius norms.

### Changed

- `leto-ops`: added a generic `Scalar::dot_slice` reduction hook. Native
  f32/f64 route through Hermes SIMD dot dispatch; reduced-precision f16/bf16
  keep the existing native scalar fallback.
- `leto-ops`: rank-1 `dot` and dense-slice `NormL2` now reuse the shared
  `dot_slice` contract, keeping one authoritative dot reduction path.

### Performance (criterion, recorded in benchmark_results.md)

- `reductions/norm_l2_64k`: 28.07 µs → 5.508 µs (**−80.0%, p < 0.05**).
- `reductions/norm_l2_transposed_256x256`: 28.67 µs → 5.550 µs
  (**−80.7%, p < 0.05**) when the transposed view exposes a dense memory
  slice. Negative-stride `norm_l2` remains on the row-walk fallback.

### Tests

- Existing norm and dot value-semantic tests cover the shared dot-slice path,
  including reduced-precision fallback and strided fallback coverage.

## [0.11.2] - 2026-06-11

Patch performance increment for strided whole-array reductions and norms.

### Changed

- `leto-ops`: strided whole-array `sum` and generic `norm` traversal now use
  `RowMajorTraversal`, computing each innermost row base once and walking by
  the last-axis stride. This removes the per-element row-major index
  decomposition from reverse-last-axis and transposed reduction fallbacks.
- `crates/leto-ops/benches/kernels.rs`: added criterion baselines for
  transposed and reverse-last-axis `sum`/`norm_l2` reductions.

### Performance (criterion, recorded in benchmark_results.md)

- First measured strided reduction baselines: transposed `sum` 40.73 µs,
  transposed `norm_l2` 28.67 µs, reverse-last-axis `sum` 30.55 µs, and
  reverse-last-axis `norm_l2` 30.21 µs.

### Tests

- Added negative last-axis stride value-semantic coverage for whole-array
  `sum` and norms.

## [0.11.1] - 2026-06-11

Patch performance increment for strided unary and binary map traversal.

### Changed

- `leto-ops`: strided `map_into`, `mapv`, and `binary_map` now traverse by
  innermost logical rows. Each row computes base offsets once and then advances
  by signed last-axis strides, reducing per-element index decomposition and
  offset multiplication on Apollo/Coeus strided views.
- Shared the row traversal shape/chunk calculation through
  `RowMajorTraversal` to avoid maintaining separate unary and binary copies of
  the same traversal policy.

### Performance (criterion, recorded in benchmark_results.md)

- `elementwise_add/transposed_256x256`: 1.206 ms → 49–51 µs (**−95.9%,
  23.7×, p < 0.05**); `contiguous_64k` statistically unchanged (p = 0.56);
  untouched kernels (matmul, reductions) verified unchanged.

### Tests

- Added negative last-axis stride differential tests against ndarray for unary
  `mapv` and binary `add`, covering the signed-stride row-walk path.

## [0.11.0] - 2026-06-11

Stage A2 ndarray-parity increment: indexed mutable zip traversal for
position-aware Apollo/Coeus migration paths.

### Added

- `leto-ops`: `indexed_zip_mut_with` — `zip_mut_with` plus logical
  row-major `[usize; N]` coordinates passed into the closure, matching the
  `ndarray::Zip::indexed` use case without allocation or runtime dispatch.
- `leto-ops`: `indexed_zip2_mut_with` — indexed three-operand mutable zip for
  caller-owned outputs with two read-only inputs.

### Tests

- Value-semantic indexed zip tests cover logical coordinate use on dense
  rank-2 arrays and transposed strided views, including three-operand indexed
  traversal.

## [0.10.0] - 2026-06-10

Stage A1 third increment: Householder QR + least squares, and Cholesky (SPD).

### Added

- `leto-ops`: `application/linalg/qr.rs` — `qr_decompose` (Householder, m ≥ n,
  compact packed form: R upper, reflector tails below the diagonal, heads/β
  alongside; `Q` is never materialized — solves apply reflectors directly,
  the fast and memory-lean form) and `QrDecomposition::solve_least_squares`
  (`Qᵀ·rhs` by reflector application + back-substitution; exact solve at
  m = n). Rejections: underdetermined shape, non-finite input, exactly-zero
  pivot-column norm (documented exact contract — near-deficiency is
  conditioning, not detected by unpivoted QR).
- `leto-ops`: `application/linalg/cholesky.rs` — `cholesky_decompose`
  (`A = L·Lᵀ`, reads only the lower triangle so symmetric storage works
  unchanged) and `CholeskyDecomposition::solve`. Positive-definiteness is
  verified constructively (non-positive pivot rejects). Driver: CFDrs
  `cfd-math` SPD paths.

### Tests

- QR: square solve cross-checked against LU, overdetermined least squares vs
  the independent nalgebra SVD oracle, residual-orthogonality optimality
  property (`Aᵀ(Ax−b) ≈ 0`), underdetermined/zero-column rejection.
- Cholesky: factor vs nalgebra `cholesky().l()`, solve vs LU, transposed
  strided-view symmetry invariance, indefinite/non-square rejection.
- Both: f32 genericity cross-check (QR vs Cholesky agreement).

## [0.9.0] - 2026-06-10

Stage A1 second increment: LU with partial pivoting, solve, determinant,
inverse. Driver: CFDrs `cfd-math` dense solver paths.

### Added

- `leto-ops`: `application/linalg/lu.rs` — `lu_decompose` (partial pivoting,
  packed unit-L/U factors, permutation + parity), `LuDecomposition<T>` with
  `solve` (forward/back substitution), `det` (parity × U diagonal), and `inv`
  (identity-column solves); plus `solve`/`det`/`inv` convenience entry points.
  Generic over `RealScalar`, native-precision elimination (no hidden
  widening). `det` of a singular matrix returns zero; `solve`/`inv` reject
  singular and non-finite inputs with distinct error reasons.

### Tests

- nalgebra differential oracle (`lu().solve`, `determinant`, `try_inverse`),
  pivot-parity coverage via zero leading pivot, `inv·A = I` value check,
  strided/transposed logical-value decomposition (`det(Aᵀ) = det(A)`),
  singular/non-square/non-finite rejection, and f32 genericity.

## [0.8.0] - 2026-06-10

Stage A1 (nalgebra replacement) first increment: norms, plus a vertical
`linalg` module consolidating dense linear algebra.

### Added

- `leto-ops`: `application/linalg/norms.rs` — `NormKind` ZST contract with
  `NormL1` (`Σ|x|`), `NormL2` (`sqrt(Σx²)`; Euclidean over rank-1, Frobenius
  over rank-2+ through one generic entry point), and `NormMax` (`max|x|`)
  markers; one generic `norm::<K, T, N>` traversal (memory-order contiguous
  fast path + strided logical fallback) with `norm_l1`/`norm_l2`/`norm_max`
  wrappers. Generic over `RealScalar`, native-precision accumulation.
- `leto-ops`: top-level re-export of `symmetric_eigen_jacobi_with_tolerance`.

### Changed

- `leto-ops`: the eigensolver moved into the new `application/linalg/` module
  (`linalg/eigen.rs`); all public re-export paths (`leto_ops::symmetric_eigen_jacobi`,
  `SymmetricEigenDecomposition`) are unchanged.

### Tests

- Norm differential oracle vs nalgebra (`DVector::norm`/`lp_norm`/`amax`,
  `DMatrix::norm` Frobenius), strided/transposed layout-independence,
  logical-selection-only strided slices, empty-view zero, and exact
  reduced-precision (3-4-5 in `f16`) coverage.

## [0.7.0] - 2026-06-10

### Added

- `leto`: `InsertAxis` rank helper (`domain/insert_axis.rs`) — the dual of
  `RemoveAxis`, mapping rank `N -> N + 1` at compile time on stable Rust through
  the shared `RankMarker` ZST, for ranks 0..=7 (output 1..=8).
- `leto`: `stack` (`application/structure/stack.rs`) — stacks equal-shaped
  rank-`N` views along a new axis (inserted at `0..=N`), producing rank
  `M = N + 1` C-contiguous output in logical row-major order. Output rank is
  resolved via `InsertAxis`; call as `stack::<T, N, M>(..)`.

### Tests

- ndarray differential oracles added for `unary_map` (exp/sqrt), `scalar_map`,
  `concat`, `stack`, `batched_matmul` (per-batch `dot`), and `cumsum`, raising
  the `leto-ops` differential suite to 57 tests.

### Notes

- Closes the `stack` item deferred in 0.6.0. The rank-preserving structural ops
  (`concat`/`pad`/`split`) remain unchanged.

## [0.6.0] - 2026-06-10

Phase 6/7 gap remediation: structural ops, batched contraction, scans, seeded
RNG, and multi-operand zip. All const-rank per ADR 0002.

### Added

- `leto`: structural array operations in a new `application/structure/` module —
  `concat` (along an existing axis), `pad` (per-axis before/after with a fill
  value), and `split` (zero-copy subviews along an axis). `concat`/`pad`
  allocate C-contiguous output and read inputs in logical (row-major) order, so
  strided/transposed inputs are handled correctly.
- `leto-ops`: `batched_matmul` for rank-3 `[B,M,K] x [B,K,N] -> [B,M,N]`, with
  batch broadcasting when either operand's batch dim is 1. Dispatches each batch
  to the rank-2 `matmul` kernel (one authoritative contraction).
- `leto-ops`: prefix/suffix scans in a new `application/scan.rs` — `ScanOp`
  trait with `CumSumOp`/`CumProdOp` markers, `ScanDirection` (Forward/Reverse),
  `scan_axis`/`scan_axis_into`, and `cumsum`/`cumsum_into` wrappers.
- `leto-ops`: deterministic seeded random constructors `uniform_with_seed` and
  `normal_with_seed` (Box-Muller), backed by a new `Xorshift64` PRNG domain
  type (`domain/rng.rs`). Validated against closed-form distribution statistics.
- `leto-ops`: `zip2_mut_with`, the three-operand analogue of `zip_mut_with`
  (`ndarray`'s `Zip::from(out).and(a).and(b)`).
- `leto-ops`: `RealScalar::from_f64` (construction-time conversion for sampling
  and constants).

### Notes

- `stack` (rank-increasing concat, rank `N -> N+1`) is deferred: stable Rust
  lacks const-generic rank arithmetic, so it needs an `InsertAxis` rank helper
  mirroring `RemoveAxis`. Tracked in backlog Phase 6.
- `batched_matmul` value tests cover explicit batches and batch broadcast; its
  per-batch path is the rank-2 `matmul` already differentially tested against
  `ndarray`.

## [0.5.0] - 2026-06-10

### Added

- `leto`: `Layout::reshape`, `Array::reshape`, `Array::into_shape`,
  `ArrayView::reshape`, and mutable reshape view variants for dense row-major
  layout reinterpretation without materializing storage.
- `leto`: `permute`/`permute_mut` named aliases over transpose semantics for
  arrays and views.
- `leto`: `to_contiguous` materialization for arrays and views, copying
  strided, transposed, or broadcasted layouts into canonical row-major storage.

## [0.4.0] - 2026-06-10

### Added

- `leto-ops`: `binary_map`/`add`/`sub`/`mul`/`div` now broadcast each input
  view to the caller-owned output layout when shapes are compatible, covering
  Coeus-style `[N, 1]` and `[1, C]` elementwise tensor paths without allocating
  broadcasted inputs.

## [0.3.0] - 2026-06-10

ndarray/nalgebra gap remediation toward Apollo hot-kernel and Coeus backend
readiness. See `gap_audit.md` for the full analysis and `docs/adr/` for the
two architectural decisions this release records.

### Added

- Offset-independent contiguity model on `Layout`: `is_contiguous` (dense in C
  or F order regardless of offset) and `is_c_dense`, alongside the existing
  canonical `is_c_contiguous`/`is_f_contiguous` (offset 0).
- Contiguity queries on `ArrayView`/`ArrayViewMut`: `is_c_contiguous`,
  `is_f_contiguous`, `is_contiguous`.
- Memory-order slice access on views: `as_slice_memory_order` (both views) and
  `as_mut_slice_memory_order` (mutable view), the `ndarray::as_slice_memory_order`
  analogue Apollo's in-place FFT butterfly kernels require. `as_slice`/
  `as_mut_slice` now expose dense C-order blocks at non-zero offsets (sliced and
  axis-iterated subviews).
- `leto-ops`: `RealScalar` trait (segregated transcendental extension of
  `Scalar`) implemented for `f32`/`f64` (native) and `f16`/`bf16` (documented
  `f32` fallback).
- `leto-ops`: named unary math operations as ZST/value-carrying markers
  (`ExpOp`, `LnOp`, `SinOp`, `CosOp`, `SqrtOp`, `AbsOp`, `NegOp`, `RecipOp`,
  `PowfOp`) routed through the shared traversal kernel via the `UnaryOp` trait
  and `unary_map`/`unary_map_into` entry points.
- `leto-ops`: `map_inplace` (the `ndarray::mapv_inplace` analogue).
- `leto-ops`: `scalar_map`/`scalar_map_into` reusing the `BinaryOp` markers for
  array–scalar broadcast arithmetic.
- `leto-ops`: rank-1 `dot` product (contiguous fast path + strided fallback,
  native-precision accumulation).

### Changed

- `symmetric_eigen_jacobi`/`symmetric_eigen_jacobi_with_tolerance` and
  `SymmetricEigenDecomposition` are now generic over `T: RealScalar` (previously
  `f64`-only with `Vec<f64>` eigenvalues), running in native precision with no
  hidden widening. The `f64` call sites are unchanged by inference.

### Notes

- std::ops operator overloading on `Array` is intentionally not added; the
  orphan rule prevents `leto-ops` from implementing foreign operator traits for
  the foreign `Array` type. See `docs/adr/0001-elementwise-operator-overloading.md`.
- The const-rank vs dynamic-rank boundary for Coeus integration is decided in
  `docs/adr/0002-coeus-rank-boundary.md` (const-generic dispatch shim at the
  Coeus boundary); implementation is tracked in backlog Phase 6.
