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
| Construct: zeros/ones/from_elem/from_vec/from_shape_vec/from_shape_fn/from_fn/eye/from_iter | ndarray | `Array::*` + `FromIterator` | Verified | core_tests — constructor shape/value semantics, identity, closure generation, iterator collection |
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
| argmin / argmax (axis index-array + whole-array multi-index) | ndarray-stats | `argmin`/`argmax` (axis) + `argmin_all`/`argmax_all` (`[usize; N]`) | Verified | core reductions — axis variants + whole-array multi-index; first-occurrence tie-break; value-agrees-with-`min_all`/`max_all` cross-check |
| cumsum / prefix scan | (no native) | `cumsum`/`scan_axis` | Verified | parity.rs |
| matmul 2D | `.dot()` | `matmul` | Complete | parity.rs, kernels.rs matmul |
| matmul transposed/strided | `.dot(t())` | `matmul` | Verified | parity.rs |
| batched matmul (rank-3) | (manual) | `batched_matmul` | Verified | parity.rs |
| vector dot | `Array1::dot` | `dot` | Complete | parity.rs, kernels.rs `dot_*` |
| concat / stack | `concatenate`/`stack` | `concat`/`stack` | Verified | parity.rs |
| pad / split | (manual) | `pad`/`split` | Verified | structure tests |
| Random (uniform/normal, seeded) | ndarray-rand | `uniform/normal_with_seed` | Complete | structure_ops.rs, kernels.rs |
| std::ops operator overloads | `+ - * /` on arrays, `-` neg, scalar ops | `&a op &b`, `&a op s`, `-&a` | Verified | core/arithmetic (ADR 0004; `*` elementwise) |
| Dynamic rank `IxDyn` (runtime-rank carrier + zero-copy bridge) | `IxDyn` | `ArrayD<T,S>` + `LayoutDyn`; `into_dyn` / `into_dimensionality::<N>` | Verified | core/dynamic (ADR 0007) — boundary carrier (construct/inspect/index/reshape/materialize) + zero-copy rank bridge to the const-rank kernels; round-trip, strided, runtime-rank dispatch. **Realized at the PyO3 boundary** (`leto_python.sum_dyn`, arbitrary-rank numpy → zero-copy `ArrayD` → bridge). Compute via rank recovery by design (not a parallel substrate) |
| Element iteration `iter`/`iter_mut`/`indexed_iter`/`indexed_iter_mut` (logical-order, double-ended) | ndarray `iter`/`iter_mut`/`indexed_iter`/`indexed_iter_mut` | `Array`/`ArrayView::iter`/`indexed_iter` → `ElementIter`/`IndexedIter`; dense `Array::iter_mut`; `Array`/`ArrayViewMut::indexed_iter_mut` → `IndexedIterMut`; `IntoIterator for &ArrayView` | Verified | core/iteration — row-major order, transposed/strided logical order, `([usize;N], &T)` and `([usize;N], &mut T)` pairs, mutable alias rejection, `DoubleEndedIterator`+`ExactSizeIterator`, empty |
| Iterator surface: exact chunk streaming | ndarray `exact_chunks` | `Array`/`ArrayView::exact_chunks` → `ExactChunks` | Verified | core/chunks — `∏ floor(sᵢ/cᵢ)` count theorem+proof; skipped remainders; transposed/strided zero-copy correctness; `DoubleEndedIterator`+`ExactSizeIterator`; zero-extent rejection |
| Iterator surface: axis chunk streaming | ndarray `axis_chunks_iter` | `Array`/`ArrayView::axis_chunks_iter` → `AxisChunks` | Verified | core/chunks — `ceil(s_axis / c)` coverage theorem+proof; final remainder chunk; transposed/strided zero-copy correctness; `DoubleEndedIterator`+`ExactSizeIterator`; invalid-axis and zero-length rejection |
| Iterator surface: sliding `windows` | ndarray `windows` | `Array`/`ArrayView::windows` → `Windows` | Verified | core/windows — zero-copy `∏(sᵢ−wᵢ+1)` window-count theorem+proof; row-major content; transposed/strided zero-copy correctness; `DoubleEndedIterator`+`ExactSizeIterator`; zero/oversize rejection |
| Iterator surface: `lanes`/`lanes_mut` | ndarray `lanes`/`lanes_mut` | `Array`/`ArrayView::lanes`, `Array`/`ArrayViewMut::lanes_mut` → `Lanes`/`LanesMut` | Verified | core/lanes — 1-D views along an axis; partition theorem+proof; dual-to-rows/columns; strided; double-ended; mutable disjoint writes **miri-clean** |
| Stats: variance / std (population + sample, axis) | ndarray-stats / ndarray `var` | `var_all`/`std_all`/`var_axis`/`std_axis` (leto core) | Verified | core/variance — two-pass; closed-form + ndarray `var`/`std`/`var_axis` differential; ddof |
| Stats: quantile / median (all + axis, 5 interpolations) | ndarray-stats / numpy | `quantile_all`/`median_all`/`quantile_axis`/`median_axis` + `Interpolation` | Verified | core/quantile — fractional-rank `h=q·(n−1)`; closed-form linear/lower/higher/nearest/midpoint oracles; per-lane equivalence; NaN/range rejection |
| Stats: covariance / Pearson correlation (rowvar) | ndarray-stats `cov`/`pearson_correlation` | `covariance` / `pearson_correlation` (leto core `statistics/`) | Verified | core/statistics — two-pass centered cross-products; closed-form sample/population oracles; diagonal == `var_axis`; symmetry; perfect ±1 correlation; ddof/empty rejection |

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
| Thin full-rank SVD (bidiagonal QR) | `SVD` subset | `svd_decompose` / `svd_via_bidiagonal` | Verified | svd/bidiagonal_qr — Golub–Reinsch implicit-shift QR with U/V (const-generic `VEC`); reconstruction + orthonormality + nalgebra σ across tall/square/wide; current criterion gap ~3.5–4.1× nalgebra (was 10–18× Gram/Jacobi); rejects rank-deficient (full-rank contract) |
| Rank-revealing SVD (incl. rank-deficient U/V) | `SVD` | `svd_rank_revealing` / `MatrixDecompose::svd_rank_revealing` | Verified | svd/jacobi.rs — one-sided Jacobi (ADR 0005); vs nalgebra singular values (tall/wide/deficient) + reconstruction + orthonormal V |
| Singular values (incl. rank-deficient) | `SVD::singular_values` | `singular_values` | Verified | svd/bidiagonal_qr — implicit-shift bidiagonal QR (Golub–Kahan), no `AᵀA` (κ(A) not κ(A)²); values-only bidiagonal reduction skips U/V factor allocation; σ-preservation theorem; 21-matrix nalgebra battery + closed-form + rank-deficient + wide-dynamic-range (1e-6→1e-15); current criterion gap ~3.6–3.8× nalgebra |
| Norms L1/L2/max | `norm`/`norm_squared` | `norm_l1/l2/max` | Complete | norms.rs, kernels.rs |
| Pseudo-inverse (full-rank **and** rank-deficient) | `pseudo_inverse` | `pinv` / `MatrixSolve::pinv` | Verified | svd/pseudoinverse.rs — rank-revealing via Jacobi SVD; vs nalgebra `pseudo_inverse` + Moore-Penrose `A A⁺ A = A`, `A⁺ A A⁺ = A⁺` |
| Non-symmetric eigenvalues (real + complex) | `complex_eigenvalues` | `eigenvalues` / `MatrixDecompose::eigenvalues` | Verified | eigenvalues/ — now delegates to the real Schur (Francis double-shift) iteration (one QR iteration, SSOT; ~1.7× faster than the removed complex single-shift QR); vs nalgebra `complex_eigenvalues` battery + exact known spectra |
| Hessenberg reduction | `Hessenberg` | `hessenberg` / `MatrixDecompose::hessenberg` | Verified | hessenberg/ (Householder; ADR 0006); reconstruction + orthogonality + structure + trace/Frobenius invariants + nalgebra Frobenius parity |
| Real Schur form (Q, T with vectors) | `Schur` | `schur` / `MatrixDecompose::schur` → `RealSchur` (`q`/`t`/`eigenvalues`) | Complete | schur/, kernels.rs — Francis double-shift QR (real arithmetic, Q accumulation) with theorem+proof; reuses Hessenberg (SSOT); exact reconstruction `A = Q T Qᵀ`, `Q` orthogonality, quasi-triangular structure (2×2 blocks only for complex pairs), spectrum vs `eigenvalues` kernel + nalgebra |
| Bidiagonalization | `Bidiagonal` | `bidiagonalize` / `MatrixDecompose::bidiagonalize` | Verified | bidiagonal/ — Golub–Kahan two-sided Householder (ADR 0006); reconstruction + orthogonality + structure + singular-value preservation vs leto & nalgebra SVD |
| LU, complete pivoting (rank-revealing) | `FullPivLU` | `full_piv_lu` / `MatrixDecompose::full_piv_lu` | Verified | full_piv_lu/ — `P A Q = L U`; reconstruction + rank + det/solve/inv vs nalgebra `FullPivLU` + rank-deficiency revelation |
| QR, column pivoting (rank-revealing) | `ColPivQR` | `col_piv_qr` / `MatrixDecompose::col_piv_qr` | Verified | col_piv_qr/ — `A P = Q R`; reconstruction + orthogonality + rank + full-rank least squares vs leto QR & nalgebra normal equations |
| UDU / LDLᵀ (symmetric indefinite, unpivoted) | `UDU` | `udu_decompose` / `MatrixDecompose::udu` | Verified | udu/ — unpivoted `A = U D Uᵀ`; reconstruction + determinant/solve/inverse vs nalgebra + zero-pivot rejection |
| Bunch–Kaufman (symmetric indefinite, pivoted) | `BunchKaufman` | `bunch_kaufman` / `MatrixDecompose::bunch_kaufman` | Complete | bunch_kaufman/, kernels.rs — partial-pivot `P A Pᵀ = L D Lᵀ` (1×1/2×2 blocks, α=(1+√17)/8) with theorem+proof; **exact reconstruction** identity, det/solve/inverse vs LU, zero-diagonal indefinite (forces 2×2 pivot), rejection. Stable general form of UDU |
| Trace | `Matrix::trace` | `trace` / `MatrixProperties::trace` | Verified | properties + doctest (vs nalgebra; spectral + cyclic theorems; `Scalar`-generic incl. integers; crate-root and `application` exports pinned) |
| Numerical rank | `Matrix::rank` | `matrix_rank` / `matrix_rank_with_tolerance` / `MatrixProperties::rank` | Verified | properties + doctest (vs nalgebra `rank`; rank = #nonzero σ; full/deficient/tall; explicit tolerance trait/free-function parity) |
| Kronecker product | `kronecker` | `kron` / `MatrixProduct::kron` | Verified | properties + doctest (vs nalgebra `kronecker` + mixed-product `(A⊗B)(C⊗D)=(AC)⊗(BD)`; crate-root and `application` exports pinned) |
| Matrix exp / power | nalgebra `exp` / `pow` | `matexp` / `matpow` + `MatrixFunction` | Verified | matrix_function/ — `matpow` exp-by-squaring (`Θ(log k)`, exact incl. integers); `matexp` scaling-and-squaring + diagonal Padé(6) with theorem+proof; closed-form (zero/diagonal/nilpotent/skew→rotation) + nalgebra `exp`/`pow` differential; reuses `matmul`+LU-inv (SSOT) |
| Stack-allocated fixed-size arrays (allocation-free) | `Matrix3`/`Vector3` (stack aspect) | `StackStorage<T, CAP>` + `Array::from_stack`/`from_stack_elem` | Verified | core/stack_storage (ADR 0008) — inline `[T; CAP]` backing, `no_std`/`Copy`; reuses the **full** op surface via the `Storage` trait (SSOT, no per-backend code); construction/validation/reductions/iteration/transpose |
| Compile-time fixed *shape* (type-level dims) | `Matrix3` shape-in-type | — | **Excluded(architecture)** | ADR 0008 — leto is const-*rank*/runtime-*dims* (ADR 0002); type-level dims fork the core type, no consumer driver |
| Geometry: Rotation/Isometry/Quaternion/Perspective | nalgebra geometry | — | **Excluded(bounded-context)** | ADR 0008 — spatial transforms are a downstream domain crate, not the array substrate (SRP/bounded-context isolation) |

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

- ndarray array families tested/present: 28 of 28 rows at Verified+ / Complete.
  No remaining Partial or Missing rows.
- nalgebra dense-decomposition/property families enumerated above are
  Verified+ / Complete for every implemented row, including rank-revealing SVD,
  non-symmetric eigenvalues, Schur/Hessenberg, secondary factorizations,
  trace, rank, and Kronecker. The remaining enumerated nalgebra rows are
  explicit exclusions, not missing kernels: compile-time fixed shape and
  geometry stay outside the parity denominator per ADR 0008.

These counts are a **seed**. The authoritative percentage is produced after
Stage 1 enumerates the complete oracle surface from locked source; this file is
the living scoreboard updated per closed row.
