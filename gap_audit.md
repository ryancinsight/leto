# Leto Gap Audit: ndarray / nalgebra Replacement for Atlas

Audit date: 2026-06-12. Evidence tier: codebase scan of `leto` (0.19.6),
`D:/atlas/repos/apollo`, `D:/atlas/repos/coeus`, current docs.rs pages for
`ndarray 0.17` and `nalgebra`, and upstream Atlas crates. Counterparts:
`ndarray 0.17`, `nalgebra` (already removed from Apollo).

## Consumer Position

- **Apollo** (spectral transforms): partially migrated. Pins
  `leto rev=fd1d87b` with `["std", "ndarray-compat"]`; exposes
  `forward_leto`/`inverse_leto` boundaries on FFT, CZT, DHT, NUFFT, SHT,
  Radon, STFT; nalgebra removed (FrFT/GFT eigendecomposition now uses
  `leto_ops::symmetric_eigen_jacobi`, GFT adjacency uses `leto::Array2<f64>`).
  ndarray remains the internal CPU compute substrate and differential oracle.
- **Coeus** (tensor/autodiff, burn replacement): CPU array layer **fully
  consolidated onto leto** (re-verified 2026-06-15 against coeus HEAD
  `037fdd5`; pins leto `d8d34c61`, older than current leto HEAD `723a63c`).
  coeus's CPU `BackendOps` route every array primitive (elementwise, matmul +
  batched, axis reductions, argmax/argmin, cumsum/suffix, concat/pad/split/
  stack, seeded RNG, to_contiguous/reshape/permute, cross-backend transfer,
  from_fn/eye/arange/linspace) through the `coeus-leto` const-rank dispatch shim
  (ADR 0002) into leto/leto-ops, covered by `coeus-leto/tests/contract.rs` and
  `coeus-ops`/`coeus-tensor` `*_leto_diff.rs` suites (coeus workspace 255 tests
  green). Coeus retains its sealed `ComputeBackend`, autodiff, NN kernels
  (conv/pool/attention/optimizers), higher sparse formats/backends, and
  wgpu/CUDA backends. `coeus-core` keeps a dynamic-rank `Layout` and
  `Storage`/`StorageMut` traits that
  `coeus-leto` converts to leto's const-rank views at the boundary — this is the
  intended ADR 0002 seam, not residual duplication.

## Layer Boundary Decision (proposed, [arch])

Leto owns the non-differentiable array substrate: layout/strides, storage,
views, slicing, broadcasting, elementwise binary/unary math, reductions,
matmul (incl. batched), shape ops (concat/pad/split), dense linear algebra, and
narrow CPU CSR sparse-dense parity kernels. Coeus owns autodiff, NN kernels (conv,
pool, attention), optimizer fusion, higher sparse formats/backends, and device
(GPU) backends. Apollo owns transform kernels. FFT stays in Apollo; Coeus
already routes `fft_1d` there.

## A. Gaps vs ndarray 0.16 (Apollo-facing)

Present and verified: const-rank `Array/ArrayView/ArrayViewMut` (+ rank
aliases 1–3), C/F layouts, ndarray-style `SliceArg` slicing, transpose,
broadcast, axis iteration, `zeros`/`ones`/`from_elem`/`from_vec`/
`from_shape_vec`/`from_shape_fn`/`into_vec`, `map_into`/`mapv`/`map`,
`zip_mut_with`, sum/mean/min/max (all + keep-dim axis), argmin/argmax, 2D
matmul, variance/std (all + axis, finite `ddof`), quantile/median (all +
axis, five interpolation strategies), covariance/Pearson correlation
(rowvar), CoW storage, Mnemosyne storage, ndarray-compat conversions.

| Gap | ndarray counterpart | Consumer driver | Class |
| --- | --- | --- | --- |
| Contiguous-slice access on views (`as_slice`, `as_slice_mut`, memory-order variant) | `as_slice_memory_order_mut`, `is_standard_layout` | Apollo FFT butterfly kernels require contiguous mutable slices (~20 call sites) | Closed |
| Multi-array zip (3+ operands) and `Zip::indexed` | `Zip::from(..).and(..).and(..)`, `Zip::indexed` | Apollo precision-downgrade, scaling, and position-aware paths | Closed (`zip2_mut_with`, `indexed_zip_mut_with`, `indexed_zip2_mut_with`) |
| `mapv_inplace` / in-place unary mutation | `mapv_inplace` | Apollo normalization (1/N scaling) (~5 sites) | Closed |
| Reshape / `into_shape` on contiguous arrays | `into_shape_with_order` | Apollo (low frequency), Coeus `reshape` (required) | Closed |
| Scalar–array elementwise ops (array + scalar, array * scalar) | `&a + 1.0`, `mapv` shortcuts | Apollo scaling, Coeus bias/scale paths | Closed |
| Broadcast-aware binary ops into caller-owned output | broadcasted elementwise ops | Coeus passes `a_layout`, `b_layout`, `c_layout`; Apollo validation and scale paths | Closed |
| std::ops operator impls on arrays/views (`Add`, `Sub`, `Mul`, `Div`, `Neg`) | operator overloads | Ergonomics for both consumers; std-trait integration mandate | Deferred by ADR 0001; current scalar/binary map APIs cover driven cases |
| `concat`/`stack` along axis | `ndarray::concatenate`, `stack` | Coeus `cat()`; Apollo validation builders | Closed (`concat`; `stack` via `InsertAxis` rank helper) |
| Dynamic-rank escape type at I/O boundaries | `IxDyn` | Apollo generic-over-dimension helpers (~30 sites use `Array<T, D>`); Coeus layout is dynamic-rank | Closed (`ArrayD`, `LayoutDyn`, zero-copy rank bridge; ADR 0007 boundary carrier, compute still via const-rank recovery) |
| 1D dot / vector ops | `Array1::dot` | Apollo, Coeus | Closed |
| Elementwise unary math suite (`exp`, `ln`, `sin`, `cos`, `sqrt`, `abs`, `neg`, `powf`) as named ZST ops | `mapv` with std float fns | Coeus `UnaryOp` enum (17 math/activation variants build on these) | Closed |
| `cumsum` / prefix scans along axis | (ndarray lacks native; Coeus has) | Coeus `cumsum`, `suffix_sum` | Closed (`scan_axis`, `cumsum`, fwd/rev, CumSum/CumProd) |
| Random constructors (uniform/normal, seeded) | `ndarray-rand` | Coeus init (`Xorshift64`, Box-Muller); keep deterministic, seed-based | Closed (`uniform_with_seed`, `normal_with_seed`) |
| Pad / split along axis | (manual in ndarray) | Coeus shape ops | Closed (`pad`, `split`) |
| Batched (rank-3) matmul | (via einsum/manual) | Coeus batched contraction — boundary decision places it in leto | Closed (`batched_matmul`, batch broadcast) |
| Variance / standard deviation | ndarray-stats / ndarray `var` | Array statistics parity | Closed (`var_all`, `std_all`, `var_axis`, `std_axis`; two-pass, finite `ddof`) |
| Quantile / median | ndarray-stats / numpy | Array statistics parity | Closed (`quantile_all`, `median_all`, `quantile_axis`, `median_axis`; five interpolation strategies, NaN/range rejection) |
| Correlation / covariance | ndarray-stats | Array statistics parity | Closed (`covariance`, `pearson_correlation`; rowvar, two-pass centered covariance, exact empty/ddof rejection) |

Non-goals confirmed: conv/pool/attention/optimizer kernels, higher sparse
formats/backends beyond CPU CSR SpMV/SpMM, autodiff — these stay in Coeus. GPU
buffers stay behind Coeus's `ComputeBackend`.

## B. Gaps vs nalgebra (linear algebra)

Apollo's nalgebra removal is complete; remaining gaps are forward-looking
for Coeus/consumer needs, not blocking any current consumer. nalgebra's
documented decomposition surface includes Schur/Hessenberg, symmetric
eigendecomposition, SVD, LU, QR, and Cholesky; Leto only admits the subset with
a named Atlas consumer driver.

| Gap | nalgebra counterpart | Status |
| --- | --- | --- |
| Symmetric eigensolver | `SymmetricEigen` | **Closed** — `symmetric_eigen_jacobi` (+ tolerance variant), generic over `T: RealScalar`, native precision, Jacobi rotations |
| Symmetric eigenvalues-only path | `SymmetricEigen::eigenvalues` / eigenvalue access | **Closed in 0.14.0** — `symmetric_eigenvalues_jacobi` (+ tolerance variant) shares the Jacobi diagonalization kernel through a monomorphized `RotationTarget`; the eigenvalues-only path uses a zero-sized no-vector target and avoids eigenvector storage |
| LU / solve / inverse / determinant | `LU`, `try_inverse` | **Closed** — `lu_decompose`, `solve`, `det`, and `inv`, generic over `T: RealScalar`; CFDrs dense solver driver |
| QR + least squares | `QR` | **Closed** — Householder `qr_decompose` and `solve_least_squares`; CFDrs least-squares driver |
| Cholesky | `Cholesky` | **Closed** — SPD `cholesky_decompose` and solve; CFDrs SPD driver |
| Thin full-rank SVD | `SVD` subset | **Closed; performance-updated in 0.35.0** — `svd_decompose` routes to bidiagonal QR (`svd_via_bidiagonal`) for tall/square/wide full-rank matrices and rejects rank-deficient inputs explicitly |
| Rank-deficient singular values | `SVD::singular_values` subset | **Closed; accuracy-updated in 0.34.3** — `singular_values` uses implicit-shift bidiagonal QR, so it avoids `AᵀA` and returns zero/tiny singular values without squaring the condition number |
| Full rank-revealing SVD / pseudoinverse | `SVD`, pseudo-inverse helpers | **Closed in 0.20.0** — ADR 0005 one-sided Jacobi SVD (`svd_rank_revealing`) plus rank-deficient `pinv`; verifies reconstruction, orthonormal `V`, nalgebra singular-value/pseudoinverse parity, and Moore-Penrose identities |
| Norms (L1/L2/Frobenius) | `norm`, `norm_squared` | **Closed** — `NormKind` ZSTs with `norm_l1`, `norm_l2`, and `norm_max` |
| Non-symmetric eigenvalues | `eigenvalues`, `complex_eigenvalues` | **Closed in 0.20.0** — ADR 0006 shifted complex QR after Hessenberg reduction; verifies exact spectra and nalgebra `complex_eigenvalues` parity |
| Real Schur vectors/form | `Schur` | Open — [major], eigenvalue spectrum is delivered; Schur vectors/quasi-triangular form require a consumer driver |
| UDU / LDLᵀ | `UDU` | **Closed in 0.21.0 for unpivoted UDU** — `udu_decompose` / `MatrixDecompose::udu`; verifies reconstruction, determinant/solve/inverse parity, and zero-pivot rejection. Pivoted Bunch-Kaufman remains open |
| Small fixed-size matrix/vector types | `Matrix3`, `Vector3` | Non-goal — const-rank `Array<T, S, 2>` covers the layout; no consumer driver |

Policy: linalg routines enter leto-ops only with a named consumer driver and
a differential oracle (ndarray-linalg/nalgebra as dev-dependency oracle, per
the existing ndarray-oracle pattern).

Driver-attribution correction (2026-06-15): the LU/QR/Cholesky/SVD "CFDrs
…driver" attributions above are aspirational, not actual. CFDrs (HEAD `0f578e1a`)
does **not** depend on `leto`/`leto-ops` today — it uses `nalgebra` 0.33 and
`ndarray` directly. These decompositions are validated by the nalgebra/ndarray
differential oracle (`oracle_parity.rs`) and stand on general-parity grounds;
they are not currently exercised by a live Atlas consumer. A real CFDrs
migration to leto dense linalg remains an unstarted, separately tracked item.

## C. Coeus backend integration ([arch]) — COMPLETE (2026-06-15)

The CPU array-kernel consolidation is done (verified against coeus HEAD
`037fdd5`). The plan-of-record integration path resolved as:
1. Leto provides the CPU array kernels (unary suite, broadcast-aware binary into
   caller-owned output, reductions incl. argmax/cumsum, matmul + batched,
   concat/pad/split/stack, seeded RNG). DONE.
2. Coeus routes its CPU `BackendOps` through the `coeus-leto` const-rank dispatch
   shim (ADR 0002) into those kernels — the duplicated array-primitive traversal
   loops in coeus are retired. DONE, with `coeus-leto/tests/contract.rs` and
   `*_leto_diff.rs` differential suites; coeus workspace 255 tests green.
3. Coeus keeps `ComputeBackend` ownership, wgpu/CUDA backends, autodiff, NN
   kernels (conv/pool/attention), higher sparse formats/backends, optimizers.
   As designed.

Framing correction: `coeus-tensor` is **not** a duplicate of leto to delete — it
is the autodiff-integrated `Tensor`/COW wrapper over `coeus-core`'s dynamic-rank
`Layout`, with CPU compute delegated to leto via `coeus-leto`. coeus-core's
dynamic-rank layout + `Storage` traits, converted at the `coeus-leto` boundary,
are the intended ADR 0002 seam. No leto-side capability gap remains for the CPU
re-base. Remaining cross-repo work is the apollo internal FFT-kernel migration
(apollo-owned) and the themis-0.9 re-pin cascade (§D), which gates clean consumer
rev-bumps to leto 0.24.0.

## D. Residual Risk Register

Update 2026-06-15 (v0.24.0): §A indexed zip parity, the Stage A1
consumer-driven nalgebra surface, Stage C2 dense norm SIMD coverage, and
Stage C3 unary/binary/zip column-walk line micro-tiling are closed through
symmetric eigenvalues-only, LU, QR, Cholesky, norms, full-rank thin SVD,
rank-deficient singular values, rank-revealing SVD/pseudoinverse,
non-symmetric eigenvalues, Hessenberg/bidiagonal/full-pivot/column-pivot
reductions, unpivoted UDU, variance/std, quantile/median, and
covariance/correlation reductions, Hermes-backed dense reductions, and
cache-line tiled strided elementwise traversal. The optional themis topology
dependency is wired through `leto_ops::CacheGeometry`; dense matmul now has a
measured fixed row-block kernel backed by Hermes fused multi-row AXPY;
reverse-last-axis whole-array reductions now borrow unit-stride physical row
slices. Current ndarray/nalgebra oracle tests cover LU solve/determinant/
inverse, symmetric eigenvalues, Cholesky lower factors, singular values, and
reverse-last-axis reductions. Criterion oracle comparison shows reverse
reductions faster than ndarray, but dense matmul is still slower than
ndarray/nalgebra: Leto 17.430 µs / 108.98 µs / 1.0631 ms for
64x64/128x128/256x256 vs ndarray 8.4923 µs / 66.527 µs / 495.95 µs and
nalgebra 8.7752 µs / 62.935 µs / 505.35 µs. Topology-adaptive tile sizing and
non-unit truly strided reductions remain open. See CHANGELOG and the two ADRs
in `docs/adr/`. Remaining work is cross-cutting: Apollo internal FFT-kernel
migration, the themis-0.9 re-pin cascade, dense matmul oracle performance
parity, Schur vectors, pivoted symmetric-indefinite factorization, matrix
functions, and any consumer-driven fixed-size/geometry decisions.

## Performance gap analysis (2026-06-15, AVX2 Win11 x86_64, criterion sample-10)

Decomposition surface, leto/nalgebra median ratio (`decomposition_compare`
benchmark group):

| Kernel | 32×32 gap | 64×64 gap | Bound / cause |
| --- | --- | --- | --- |
| SVD (default `svd_decompose`) | ~10.6× → **~3.5×** | ~18× → **~4.1×** | was one-sided Jacobi; **now** bidiagonal QR (Golub–Reinsch) — see below |
| eigenvalues | ~16× → **~5.8×** | ~16× → **~7.4×** | **was** complex single-shift QR; **now** real Schur (Francis), no-Q path — see below |
| matexp | ~6.6× | — | Padé reuses matmul (itself ~2× slow); even/odd split cut its products 6→4, but wall-clock is matmul-bound (tracks matmul gap) |
| QR | ~3.6× | ~4.2× | scalar Householder; no panel/blocking |
| LU | ~3.5× | ~4.1× | scalar partial-pivot |
| Cholesky | ~3.4× | — | scalar |
| dense matmul | ~2.0× | ~1.5× | AXPY (rank-1) scheme: O(n³) output traffic; needs register-blocked GEMM micro-kernel (tile-accumulating SIMD primitive → upstream hermes) |
| matpow | ~1.5× | — | matmul-bound (tracks matmul) |

Priority finding: the largest gaps are **SVD** and **eigenvalues**, *not* matmul.

- **eigenvalues — RESOLVED (partial, [patch] ×2)**: (1) consolidated onto the
  real Schur (Francis double-shift) iteration, deleting the complex single-shift
  QR (`eigenvalues/{complex,qr}.rs`, `Cplx`) — one QR iteration in the crate
  (SSOT), real arithmetic. (2) **no-Q Francis path**: `francis::run` is
  const-generic over `ACCUMULATE_Q`; eigenvalues-only passes `false`, so the
  Schur-vector update is DCE'd (zero cost) and standardization is skipped; block
  extraction factored into one shared helper. (3) **active-block confinement**:
  the eigenvalues-only apply is restricted to the active window `[lo, hi]` (left
  columns `≤ hi`, right rows `≥ lo`); skipped entries are strictly
  upper-triangular and never feed back, so the spectrum is bitwise identical
  (proof: `hi` decreases, `lo` is monotone non-decreasing for fixed `hi` via
  exact-zero deflation). 64×64 `eig` 1.69 → 1.50 ms. The safe confinement is the
  perturbation-free subset of the LAPACK `dlahqr` WANTT=false window.

  Residual gap (~5×) — **#20 CLOSED with machine-checked evidence**. Two further
  levers were investigated and rejected:
  - **Within-block `[k, k+len]` narrowing** (the rest of `dlahqr` WANTT=false):
    diverges from the nalgebra oracle on the hardened 16×16 case at eigenvalue
    `6.1e-15 ± 1.75e-7i`. Root cause is *not* a bug and *not* bad scaling — it is a
    **defective (high-multiplicity) zero eigenvalue**: machine-checked
    `det(A) = -8.7e-30 ≈ 0`, nullity 3 (smallest singular values
    `[5.55, 1.3e-15, 0, 0]`). A defective eigenvalue perturbs as `O(√(ε‖A‖))`, and
    `√(ε‖A‖) = 1.54e-7` matches the spurious `1.75e-7` imaginary part nalgebra
    reports for the defective-0. Both algorithm variants are backward-stable; they
    legitimately differ by `√(ε‖A‖) > 1e-7`. The narrowing is correctness-equivalent
    (leto's `~1e-15` is in fact nearer the true real-0) but cannot match the oracle
    at the over-tight 1e-7 absolute tolerance, for a sub-2× apply gain over the
    confinement already landed — not pursued.
  - **Matrix balancing (`dgebal` Parlett–Reinsch scaling)** — implemented, verified
    spectrum-preserving (radix-2 exact), then **reverted**: on the well-scaled
    integer benchmark matrices it produces `D ≈ I`, so it neither conditions the
    defective eigenvalue (the narrowing still failed *with* balancing) nor speeds
    convergence, while adding O(n²·sweeps) overhead that **regressed** `eig`
    (64×64 1.50 → 1.64 ms). Net-negative for the target metric with no consumer
    need; removed per subtractive/anti-over-engineering discipline.

  (4) **SIMD apply** (now the eig test is backward-error-correct, see below):
  the shared Householder apply and the Francis **left**-apply route their
  contiguous inner sweeps through `Scalar::axpy_slice` (SSOT SIMD), row-oriented
  with a reused scratch buffer and a `SPAN_SIMD_MIN = 32` threshold (narrow spans
  stay scalar). Bitwise-identical to the column-oriented form. 64×64 `eig` 1.50 →
  **1.11 ms** (5.9× → **4.4×**); 32×32 284 → **242 µs**. The brittle `1e-7`
  oracle was first replaced by the analytically-correct backward-error bound
  `8·√(ε‖A‖)` (defective-eigenvalue derivation above, machine-checked), which
  unblocked all backward-stable reorderings.

  Residual gap (~4.4×) is the Francis **right**-apply (3-wide per row, no
  contiguous span to vectorize) plus the per-step reflector overhead. Closing it
  needs the blocked/aggregated-reflector (compact-WY / small-bulge multishift)
  rewrite so the apply becomes a GEMM — tracked as `[major/arch]` (#21), not a
  patch. **#20 CLOSED**: the within-block `[k, k+len]` narrowing and balancing
  were both investigated and rejected (above).
- **SVD — RESOLVED ([patch]+[minor])**: both singular values and the full thin
  SVD now use the implicit-shift **bidiagonal QR** (`svd/bidiagonal_qr.rs`,
  reusing `bidiagonalize`). No `AᵀA`, so accuracy is `κ(A)` not `κ(A)²` (resolves
  `diag(1,1e-6)` to 1e-15). `svd_via_bidiagonal` accumulates `U`/`V` via Givens
  (const-generic `VEC`; DCE'd for values-only); `svd_decompose` routes to it
  (Gram path deleted, SSOT). Current validation: full SVD 32×32 ~292→103.6 µs
  (10.6×→3.5×), 64×64 ~3.1 ms→758.8 µs (18×→4.1×); values-only
  `singular_values` 57.1 µs (32×32) and 275.0 µs (64×64), a 25.5%/39.5% local
  median reduction after skipping factor accumulation. One-sided Jacobi
  (`svd_rank_revealing`) retained for rank-deficient / maximal accuracy.
  **SIMD bidiagonalization** (shared Householder apply now routes through
  `Scalar::axpy_slice`, see the eigenvalues entry): 64×64 full SVD 758.8 →
  **417 µs** (4.1× → **3.4×**), `singular_values` 275.0 → **133 µs** (3.8× →
  **2.3×**). Residual gap is the scalar **Givens** bidiagonal-QR sweep (strided
  2×2 rotations, sequential bulge chase).

  **ADR 0010 Phase 2 (blocked U/V factor formation) — implemented, verified, then
  REVERTED as valueless.** The factor formation was rewritten to store the panel
  reflectors and form `U = L₀…L_{n-1}` / `V = R₀…R_{n-2}` by blocked compact-WY
  `apply_block_right` (one-sided, no `dlabrd` look-ahead); verified correct by a
  256² `A = U B Vᵀ` reconstruction + orthogonality + singular-value contract. But
  the A/B measurement is decisive: **full SVD 256² is 164 ms blocked vs 163 ms
  unblocked — no difference.** The entire 256² SVD cost (~163 ms vs nalgebra
  46.6 ms, 3.5×) is the **sequential Givens bidiagonal-QR sweep**; the U/V
  formation is < 1 ms. Blocking it is therefore cargo-cult (complexity with no
  present performance need) and was removed. **The genuine SVD lever is Phase 3
  — accelerating the Givens sweep itself** (`dbdsqr`), which is inherently
  sequential (each rotation feeds the next in the bulge chase) and is exactly why
  it is the residual. `apply_block_right` reverted with it (no value-adding
  consumer; `apply_block_left`/blocked-QR Phase 1 retained, a measured 256² win).

  **ADR 0010 Phase 3 (SVD) — DONE, disparity resolved.** Without restructuring the
  serial bulge chase, the per-Givens `U`/`V` **column** rotation (striding the
  row-major factors — a cache line per row, no SIMD) was made contiguous by
  accumulating `U`/`V` **transposed**, so each rotation mixes two contiguous rows:
  bitwise-identical factor, cache-friendly auto-vectorized loop. **256² full SVD
  164 → 34.7 ms (4.7×, now *faster* than nalgebra 46.6 ms); 64² 1.31 → 0.60 ms**
  (clean A/B), verified by the SVD reconstruction + nalgebra batteries (commit
  `9bef76e`). The trick does *not* transfer to eig (`H` is the read/written working
  matrix, not an isolated accumulator), so the eig residual still needs the
  multishift block rewrite.
- **matmul — OPEN**: register-blocked GEMM micro-kernel needs a tile-accumulating
  SIMD primitive owned upstream in hermes (multi-repo, peer-agent lane); a prior
  scalar 4×4 tile regressed (no SIMD in the tile). Coordinated effort.

- `stack` (rank `N -> N+1`): CLOSED ([minor]) — implemented via the `InsertAxis`
  rank helper (dual of `RemoveAxis`, ranks 0..=7). `concat`/`pad`/`split`/`stack`
  all closed.
- Dynamic-rank boundary: DECIDED ([major]) in
  `docs/adr/0002-coeus-rank-boundary.md` — const-generic dispatch shim at the
  Coeus boundary, shim owned by Coeus, Leto stays const-rank. Phase 6 leto-side
  capabilities authored const-rank.
- `symmetric_eigen_jacobi`: CLOSED ([minor]) — now generic over `T: RealScalar`,
  native precision, no hidden widening. Residual: no wider-accumulator variant;
  consumers needing higher working precision than storage convert first
  (explicit). f16/bf16 transcendentals use the documented f32 fallback.
- `symmetric_eigenvalues_jacobi`: CLOSED ([minor]) — sorted eigenvalues without
  eigenvector allocation, implemented by a zero-sized no-vector rotation target
  over the shared Jacobi diagonalization kernel. Evidence tier:
  value-semantic full-vs-values parity and strided-view tests.
- Contiguous-slice view access: CLOSED — `as_slice`/`as_mut_slice` are now
  offset-independent C-dense; `as_slice_memory_order`/`as_mut_slice_memory_order`
  expose F-order/offset blocks. Apollo hot-kernel migration still unproven end
  to end (boundaries exist; internal FFT compute still on ndarray), but the
  named blocker is removed.
- std::ops operator overloading: DEFERRED ([arch]) in
  `docs/adr/0001-elementwise-operator-overloading.md` (orphan rule). `scalar_map`
  covers array–scalar arithmetic; no consumer blocked.
- Indexed zip parity: CLOSED ([minor]) — `indexed_zip_mut_with` and
  `indexed_zip2_mut_with` provide `Zip::indexed`-style logical coordinates for
  one- and two-input mutable zip traversals.
- Stage C3 column-walk elementwise traversal: CLOSED for binary and unary
  `map_into` plus `zip_mut_with` ([patch]/[minor]) — all use shared cache-line
  `TileGeometry`.
  Evidence tier: value-semantic strided tests plus criterion differential
  timing before/after the optimization. Remaining cache-aware CPU kernel work
  is blocked matmul cache hierarchy selection and themis topology wiring.
- Stage C2 dense norm SIMD coverage: CLOSED for `norm_l1`/`norm_l2`/`norm_max`
  over dense memory-order slices. `norm_l1` and `norm_max` now route f32/f64
  through Hermes absolute-value reductions; reduced precision keeps the native
  scalar fallback. Evidence tier: value-semantic norm tests plus criterion
  in-run scalar-reference comparison. Remaining reduction work is truly
  non-dense strided reductions, which need per-lane partial accumulators.
- Coverage of new ops: value-semantic tests plus ndarray differential oracles
  now cover the unary math suite (`exp`/`sqrt`), `scalar_map`, `concat`,
  `stack`, `batched_matmul` (per-batch ndarray `dot`), and `cumsum` (reference
  accumulate), alongside the existing map/reduction/matmul differentials. RNG is
  validated against closed-form mean/variance (correct per policy, not ndarray).
  Remaining: differential coverage is leto-internal; consumer-side (Apollo/Coeus)
  migration tests are the next cross-repo step.
- `leto-python` rustdoc ICE via `numpy 0.23`: REOPENED after the FFI
  dependency alignment back to NumPy 0.23/PyO3 0.23. Full workspace docs fail
  in rustdoc intra-doc link resolution inside `numpy 0.23.0`; `cargo doc -p
  leto -p leto-ops --all-features --no-deps` passes.
- Differential coverage: ndarray oracle covers map/reductions/matmul, unary
  suite, concat/stack, batched matmul, and cumsum. RNG uses closed-form
  references. Indexed zip currently rests on value-semantic traversal tests.
- Oracle performance parity: reverse-last-axis `sum`/`norm_l2` is at parity or
  faster than ndarray on the recorded 256x256 benchmark. Dense matmul is not
  at parity with ndarray/nalgebra and blocks replacement-performance claims.
  The 0.19.7 Hermes fused multi-row AXPY path improved Leto medians from
  21.443 µs / 127.63 µs / 2.4357 ms to 17.430 µs / 108.98 µs / 1.0631 ms for
  64x64/128x128/256x256, but ndarray/nalgebra remain faster. The 0.19.2
  zero-skip branch-removal experiment was rejected after canonical dense
  256x256 instability/regression. The 0.19.3 packed-RHS dot and
  scalar-row-update experiments also regressed 128x128. The 0.19.4 Hermes
  `tiled_gemm` f64 dense path regressed 128x128, and small-matrix serial
  scheduling was slower than row-block parallelism. The 0.19.5
  `MATMUL_ROW_BLOCK=16` and first-shared-row output initialization experiments
  did not meet the release benchmark stability/performance gate. Post-0.19.7
  Hermes column-chunk `axpy_rows` regressed 64x64/128x128/256x256 and ended
  with `STATUS_ACCESS_VIOLATION`; `MATMUL_ROW_BLOCK=64` also regressed against
  the 32-row baseline. Row-block fused-branch/alpha-buffer hoisting produced
  no statistically significant 128x128 improvement and also ended with
  `STATUS_ACCESS_VIOLATION`. Generic 4x4 registered dense tiles regressed the
  oracle shapes and are not retained. Hermes `axpy_rows_batch` improves the
  local themis-0.9 128x128 Leto oracle median from 212.64 µs to 98.853 µs when
  gated to that row regime, but broad depth-batched routing regressed other
  oracle shapes and is not retained. Next work needs allocation-controlled
  reusable packing scratch or a verified external micro-kernel provider with
  profile evidence.
- themis-0.9 migration + dependency resolution (re-diagnosed/MEASURED
  2026-06-15). Three coupled facts:
  1. **`Cargo.lock` is gitignored** in leto (`.gitignore`), so there is no
     committed lock — contrary to the prior "commit Cargo.lock" claim. leto's
     standalone build depends on a locally-generated lock.
  2. **Fresh pure-git resolution is broken.** `hermes-simd` (`efac0454`, the
     measured-good fused-AXPY matmul pin) and `mnemosyne-arena`/`moirai-iter`
     transitively require `themis ^0.8.0` with no rev, so `cargo
     generate-lockfile` floats themis to the default-branch HEAD (0.9.11) and
     fails `^0.8.0`. cargo will not unify a rev-pin with the transitive
     version-spec, and `[patch]` to the same git source is rejected — so no
     leto-local pin change resolves it. Builds worked only via a frozen
     pre-drift themis-0.8 lock (now superseded locally).
  3. **The themis-0.9 path regresses matmul.** Built leto 0.24.0 against the
     local themis-0.9 stack (path-patches): all 122 tests + 4 doctests pass, but
     the required hermes bump `efac0454`→`e6761ac` (dispatch/AXPY refactor)
     regresses dense matmul **64² 17.4→24.9 µs (+43%)**, **128² 109→176 µs
     (+61%)**, 256² ~unchanged. So themis-0.9 adoption is blocked not just by pin
     coordination but by a **measured hermes matmul regression**.
  Resolution = stack-wide re-pin cascade (themis → mnemosyne → moirai → hermes →
  leto → apollo/coeus, in order, since each pins old revs of the others) PLUS a
  hermes-side fix restoring the AXPY/dispatch perf on its themis-0.9 line; then
  leto migrates and re-measures matmul. Owned at the meta/stack level, not
  leto-local. Process gap to fix alongside: either commit a frozen `Cargo.lock`
  (un-gitignore) or pin the whole stack so fresh resolution is reproducible.
  Interim: leto builds locally via the apollo/coeus-style path-patch set
  (uncommitted, flagged in `Cargo.toml`); consumer rev-bumps to leto 0.24.0 wait
  on the cascade (coeus already verified compatible — 22/22 contract tests green
  against working-tree leto 0.24.0).
- Evidence tier of this audit: codebase scan + existing test suites +
  criterion benchmark measurements. No machine-checked proof was performed.

## Leto rank-deficient singular-values parity [patch]
- Performed: split `leto-ops::singular_values` from the full-vector `svd_decompose` path. Singular-values-only now diagonalizes the smaller Gram matrix and maps near-zero eigenvalues to zero singular values for finite rank-deficient inputs.
- Architecture effect: Leto matches the common nalgebra singular-values surface for rank-deficient matrices without fabricating null-space singular vectors. `svd_decompose` keeps explicit rank-deficient rejection until a rank-revealing SVD contract exists.
- Evidence tier: value-semantic tall/wide rank-deficient singular-value tests. No machine-checked proof was performed.

## Leto wide thin SVD parity [patch]
- Performed: generalized `leto-ops::svd_decompose` and `singular_values` from tall/square full-column-rank inputs to all full-rank thin SVD shapes. Wide full-row-rank matrices now use `A A^T`, then derive right singular vectors via `V = A^T U Σ^-1`.
- Architecture effect: Leto closes the current wide-matrix SVD nalgebra-parity gap without a second API or downstream Apollo-specific adapter. Rank-deficient inputs remain explicit errors until a rank-revealing SVD contract is implemented.
- Evidence tier: value-semantic reconstruction and orthonormality tests. No machine-checked proof was performed.
