# Leto Parity Matrix vs ndarray 0.16 / nalgebra 0.35

Scored inventory backing [`PLAN.md`](PLAN.md). Status legend:

- **Complete** — generic surface + differential test + oracle benchmark.
- **Verified** — generic surface + differential test; no perf benchmark yet.
- **Partial** — subset of oracle behavior present.
- **Missing** — no Leto counterpart.
- **Excluded(reason)** — out of parity denominator with recorded rationale.

Evidence column: `parity.rs` / `oracle_parity.rs` = differential test file;
`kernels.rs` = criterion oracle benchmark. This matrix is seeded from the
2026-06-14 first pass; Stage 1–2 of the plan fills the remaining oracle surface
(geometry, full iterator/slicing surface, companion crates) to a complete count.

## A. Arrays / ndarray surface

| Operation | Oracle | Leto API | Status | Evidence |
| --- | --- | --- | --- | --- |
| Construct: zeros/ones/from_elem/from_vec/from_shape_vec/from_shape_fn | ndarray | `Array::*` | Verified | core_tests |
| Rank aliases Array1–3, views, mut views | ndarray `Ix1..` | `Array1..3`, `ArrayView*` | Verified | core_tests |
| Slicing (ranges, neg index/step, newaxis, ellipsis) | `s![]` / `slice` | `SliceArg`/`slice_with` | Verified | core/slicing |
| Transpose / permute | `.t()`/`permuted_axes` | `transpose`/permute | Verified | core/transform |
| Broadcast | `broadcast` | layout broadcast | Verified | layout_property |
| Axis iteration, rows/columns | `axis_iter`, `rows` | `AxisIter*`, `rows/columns` | Verified | core_tests |
| reshape / into_shape | `into_shape_with_order` | reshape/to_contiguous | Verified | apollo_ndarray_contract |
| Elementwise add/sub/mul/div | `+ - * /` | `add/sub/mul/div` | Complete | parity.rs, kernels.rs `add_*` |
| Scalar–array arithmetic | `&a + s` | `scalar_map` | Verified | parity.rs |
| Unary math (exp/ln/sin/cos/sqrt/abs/neg/recip/powf) | `mapv(f)` | `unary_map`+ZST ops | Complete | parity.rs, kernels.rs `exp_*` |
| map/mapv/map_into | `map`/`mapv` | `map`/`mapv`/`map_into` | Verified | differential.rs |
| map_inplace | `mapv_inplace` | `map_inplace` | Verified | unary_math |
| Multi-array zip (3+), indexed zip | `Zip`, `Zip::indexed` | `zip2_mut_with`, `indexed_*` | Verified | ops tests |
| sum / mean / min / max (all) | `.sum/.mean/...` | `sum`, leto `*_all` | Complete (sum) | parity.rs, kernels.rs `sum_*` |
| sum/mean/min/max axis (keep-dim) | `sum_axis(Axis)` | `sum_axis`/`mean_axis`/… | Verified | parity.rs |
| argmin / argmax | ndarray-stats | `leto::argmin/argmax` | Partial | core reductions |
| cumsum / prefix scan | (no native) | `cumsum`/`scan_axis` | Verified | parity.rs |
| matmul 2D | `.dot()` | `matmul` | Complete | parity.rs, kernels.rs matmul |
| matmul transposed/strided | `.dot(t())` | `matmul` | Verified | parity.rs |
| batched matmul (rank-3) | (manual) | `batched_matmul` | Verified | parity.rs |
| vector dot | `Array1::dot` | `dot` | Complete | parity.rs, kernels.rs `dot_*` |
| concat / stack | `concatenate`/`stack` | `concat`/`stack` | Verified | parity.rs |
| pad / split | (manual) | `pad`/`split` | Verified | structure tests |
| Random (uniform/normal, seeded) | ndarray-rand | `uniform/normal_with_seed` | Partial | closed-form ref |
| std::ops operator overloads | `+ - * /` on arrays, `-` neg, scalar ops | `&a op &b`, `&a op s`, `-&a` | Verified | core/arithmetic (ADR 0004; `*` elementwise) |
| Dynamic rank `IxDyn` | `IxDyn` | — | **Missing** | ADR 0002 reopened |
| Full iterator surface (`indexed_iter`, windows, lanes, fold/accumulate) | ndarray iters | partial | **Partial→audit** | Stage 1 |
| Stats (var/std/quantile/corr) | ndarray-stats | — | **Missing** | Stage 1 column |

## B. Linear algebra / nalgebra surface

Ergonomic note (0.20.0, ADR 0003): every decomposition/solve/norm/product row
below is now reachable as a **fluent method** on any rank-2 receiver
(`m.lu()`, `m.solve(&b)`, `m.det()`, `m.matmul(&b)`, `m.norm_l2()`, …) via the
`MatrixProduct`/`MatrixNorm`/`MatrixDecompose`/`MatrixSolve` traits, in addition
to the free functions. This closes the nalgebra *method-surface* ergonomic gap
for the implemented kernels; `Missing` rows are still missing *kernels*.


| Operation | Oracle | Leto API | Status | Evidence |
| --- | --- | --- | --- | --- |
| LU + solve + det + inverse | `LU`/`try_inverse` | `lu_decompose/solve/det/inv` | Verified | oracle_parity.rs |
| QR + least squares | `QR` | `qr_decompose/solve_least_squares` | Verified | parity.rs (lstsq) |
| Cholesky factor/solve/det/inv | `Cholesky` | `cholesky_*` | Verified | oracle_parity.rs |
| Symmetric eigen (values+vectors) | `SymmetricEigen` | `symmetric_eigen_jacobi` | Verified | eigen.rs |
| Symmetric eigenvalues-only | `SymmetricEigen::eigenvalues` | `symmetric_eigenvalues_jacobi` | Verified | oracle_parity.rs |
| Thin full-rank SVD (Gram) | `SVD` subset | `svd_decompose` | Verified | svd/gram.rs (full-rank; rejects rank-deficient) |
| Rank-revealing SVD (incl. rank-deficient U/V) | `SVD` | `svd_rank_revealing` / `MatrixDecompose::svd_rank_revealing` | Verified | svd/jacobi.rs — one-sided Jacobi (ADR 0005); vs nalgebra singular values (tall/wide/deficient) + reconstruction + orthonormal V |
| Singular values (incl. rank-deficient) | `SVD::singular_values` | `singular_values` | Verified | oracle_parity.rs |
| Norms L1/L2/max | `norm`/`norm_squared` | `norm_l1/l2/max` | Verified | norms.rs |
| Pseudo-inverse (full-rank **and** rank-deficient) | `pseudo_inverse` | `pinv` / `MatrixSolve::pinv` | Verified | svd/pseudoinverse.rs — rank-revealing via Jacobi SVD; vs nalgebra `pseudo_inverse` + Moore-Penrose `A A⁺ A = A`, `A⁺ A A⁺ = A⁺` |
| Non-symmetric eigenvalues (real + complex) | `complex_eigenvalues` | `eigenvalues` / `MatrixDecompose::eigenvalues` | Verified | eigenvalues/ — shifted complex QR (ADR 0006); vs nalgebra `complex_eigenvalues` battery (real/complex/various sizes) + exact known spectra |
| Hessenberg reduction | `Hessenberg` | `hessenberg` / `MatrixDecompose::hessenberg` | Verified | hessenberg/ (Householder; ADR 0006); reconstruction + orthogonality + structure + trace/Frobenius invariants + nalgebra Frobenius parity |
| Real Schur form (Q, T with vectors) | `Schur` | — | **Missing** | [major] — eigenvalues above cover the spectrum; Schur *vectors* pending |
| Bidiagonalization | `Bidiagonal` | `bidiagonalize` / `MatrixDecompose::bidiagonalize` | Verified | bidiagonal/ — Golub–Kahan two-sided Householder (ADR 0006); reconstruction + orthogonality + structure + singular-value preservation vs leto & nalgebra SVD |
| ColPivQR / FullPivLU / UDU | nalgebra decomps | — | **Missing** | Stage 5 |
| Trace | `Matrix::trace` | `trace` / `MatrixProperties::trace` | Verified | properties (vs nalgebra; spectral + cyclic theorems; `Scalar`-generic incl. integers) |
| Numerical rank | `Matrix::rank` | `matrix_rank` / `MatrixProperties::rank` | Verified | properties (vs nalgebra `rank`; rank = #nonzero σ; full/deficient/tall) |
| Kronecker product | `kronecker` | `kron` / `MatrixProduct::kron` | Verified | properties (vs nalgebra `kronecker` + mixed-product `(A⊗B)(C⊗D)=(AC)⊗(BD)`) |
| Matrix exp / power | nalgebra inherent | — | **Missing** | Stage 5 |
| Small fixed `MatrixN`/`VectorN` | `Matrix3`/`Vector3` | — | **Excluded?** | different abstraction; decide Stage 2 |
| Geometry: Rotation/Isometry/Quaternion/Perspective | nalgebra geometry | — | **Excluded?** | not an array substrate; decide Stage 2 |

## C. First-pass performance comparison (2026-06-14, AVX2 Win11 x86_64)

Median, criterion sample-size 10, identical pinned f64 inputs.

| Benchmark | Leto | ndarray | nalgebra | Note |
| --- | --- | --- | --- | --- |
| add 64k | 18.9 µs | 13.3 µs | — | ~1.43× slower |
| exp 64k | 638 µs | 761 µs | — | ~0.84× (faster) |
| sum 64k | 3.53 µs | 4.53 µs | — | ~0.78× (faster) |
| dot 64k | 7.06 µs | 9.35 µs | — | ~0.76× (faster) |
| matmul 64² | 17.4 µs | 8.49 µs | 8.78 µs | ~2.05× slower (open) |
| matmul 128² | 109 µs | 66.5 µs | 62.9 µs | ~1.64× slower (open) |
| matmul 256² | 1.06 ms | 496 µs | 505 µs | ~2.14× slower (open) |
| sum reverse 256² | 4.78 µs | 6.07 µs | — | faster |
| norm_l2 reverse 256² | 9.35 µs | 30.9 µs | — | faster |

## D. Headline completeness (seed, pre-Stage-1)

Counting only rows enumerated above (not yet the full oracle surface):

- ndarray array families tested/present: ~22 of ~26 rows at Verified+ →
  remaining: operator overloading, `IxDyn`, full iterator surface, stats.
- nalgebra dense-decomposition families: 8 of ~15 at Verified+ → remaining:
  rank-revealing SVD, non-symmetric/Schur/Hessenberg, secondary factorizations,
  Kron/trace/rank; geometry + small-fixed pending exclude-vs-implement decision.

These counts are a **seed**. The authoritative percentage is produced after
Stage 1 enumerates the complete oracle surface from locked source; this file is
the living scoreboard updated per closed row.
