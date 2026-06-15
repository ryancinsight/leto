# ADR 0003: Fluent rank-2 linear-algebra trait layer

- Status: Accepted
- Date: 2026-06-15
- Class: [minor] (additive surface) with an architectural seam — design recorded here

## Context

The request is to consolidate the ndarray "n-dimensional strided array" model and
the nalgebra "matrix with a rich linear-algebra method surface" model into one
type. Leto already owns the strided core both models share: `Array<T, S, N>` =
`Layout<N>` (shape + strides + offset) + `Storage S`, with `Array2<T>` the strided
matrix. It already supports C-contiguous, F-contiguous, and arbitrary strided
layouts (slice/transpose/broadcast yield strided views), and the dense LA kernels
already accept strided/transposed input (they copy once into row-major working
storage).

The divergence from nalgebra is the *surface*, not the buffer. Leto exposes linear
algebra as **free functions** — `lu_decompose(&view)`, `solve(&a, &b)`,
`norm_l2(&view)` — which is ndarray's idiom. nalgebra exposes the same operations as
**methods on the matrix** — `m.lu()`, `m.solve(&b)`, `m.det()`. To consolidate, Leto
needs the fluent method surface layered onto the existing strided matrix without a
second buffer type and without a second implementation of any kernel.

## Options

1. New `Matrix<T, S>` newtype (`#[repr(transparent)]` over `Array2`) carrying the LA
   methods, with `Deref`/`From` bridges.
2. New standalone matrix type with its own buffer + shape + strides.
3. Layer role-segmented LA traits onto the existing `Array2`/`ArrayView2` via blanket
   impls that delegate to the existing free-function kernels.

## Decision

Adopt option 3. The LA traits live in `leto-ops` (which depends on `leto`), so
implementing them for `leto::Array2`/`ArrayView2` is legal (local trait, foreign
type) — the orphan-rule friction recorded in ADR 0001 for *std* operator traits does
not apply to leto-owned traits. Surface is method-based this increment; operator
overloading (`*` matmul, `+`/`-`) remains deferred per ADR 0001 and builds on this
layer later.

Structure (interface segregation — no god trait):

- `AsMatrixView<T>` — bridge: `fn as_matrix_view(&self) -> ArrayView2<'_, T>`, impl'd
  for `ArrayView2` and for `Array<T, S: Storage<T>, 2>`. Normalizes owned/borrowed/
  view receivers to one rank-2 view so each LA trait is written once.
- `MatrixProduct<T: Scalar>` — `matmul(&self, rhs) -> Result<Array2<T>>` (allocating
  ergonomic wrapper; allocates the output then calls the caller-owned `matmul`
  kernel — no second contraction path).
- `MatrixNorm<T: RealScalar>` — `norm_l1`/`norm_l2` (Frobenius)/`norm_max`.
- `MatrixDecompose<T: RealScalar>` — `lu`/`qr`/`cholesky`/`svd`/`singular_values`/
  `symmetric_eigen`/`symmetric_eigenvalues`.
- `MatrixSolve<T: RealScalar>` — `solve`/`solve_least_squares`/`inv`/`det`.

Each trait has exactly one blanket impl `impl<T, M: AsMatrixView<T>> Trait<T> for M`;
every method builds `self.as_matrix_view()` and calls the existing kernel. Public
method names drop the algorithm suffix (`symmetric_eigen`, not `_jacobi`) — the
algorithm is an implementation detail of the kernel.

Rejected: option 1/2 reintroduce a parallel type and a conversion boundary, which
contradicts the consolidation goal and violates SSOT/DRY against the existing strided
core (the whole reason `leto` exists).

## Consequences

- The free functions remain the **single authoritative implementation**; the trait
  methods are thin zero-cost delegators (monomorphized; no `dyn`). No kernel is
  duplicated, so SSOT holds.
- A rank-2 array, an owned `Array2`, and any strided `ArrayView2` all gain the same
  fluent surface; arbitrary-layout support is inherited from the kernels (covered by
  a transposed-receiver differential test).
- `MatrixProduct::matmul` requires `T: Scalar` only; it builds output via
  `from_shape_vec` with `T::ZERO` rather than `zeros` (which needs `T: Default`).
- Adding a new decomposition kernel automatically warrants a matching trait method in
  the same change (keeps the fluent surface complete as nalgebra parity grows; see
  `docs/completeness/parity_matrix.md` §B for the missing kernels).
- Operator overloading stays deferred (ADR 0001); when taken it builds on these
  traits, not on a separate type.
