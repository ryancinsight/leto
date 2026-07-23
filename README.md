# Leto: Systems-Optimized N-Dimensional Strided Arrays

Leto is a Rust workspace for N-dimensional strided array layouts, zero-copy
views, storage backends, and array operations. It replaces direct `ndarray`
and `nalgebra` usage as the shared non-differentiable memory, layout, and
dense-linear-algebra vocabulary between Atlas spectral transforms (`apollo`)
and tensor/autodiff systems (`coeus`, the Atlas replacement for `burn`).

## Role In Atlas

Leto sits between:

- `mnemosyne`: optional aligned allocation and memory policy (which itself
  consumes `themis` placement law and `melinoe` branded-capability proofs).
- `moirai`: parallel scheduling for elementwise and reduction operations.
- `hermes`: SIMD-backed scalar/vector execution.
- `apollo`: spectral transforms that need shared 1D/2D/3D array views.
- `coeus`: tensor and autodiff systems that need layout-compatible storage
  without making Apollo depend on Coeus.

Layer boundary: Leto owns layout, storage, views, slicing, broadcasting,
elementwise math, reductions, matmul, shape ops, dense linear algebra, and the
narrow CPU CSR sparse-dense parity kernels. Coeus owns autodiff graphs, NN kernels
(conv/pool/attention), optimizers, higher sparse formats/backends, and GPU
backends behind its `ComputeBackend` trait. Apollo owns Fourier, spectral, and
transform kernels. `themis` and `melinoe` are consumed indirectly via
`mnemosyne`/`moirai`, not as direct leto dependencies.

## Naming

The name is intentional. In Greek mythology, Leto is the daughter of Coeus and
mother of Apollo. In Atlas architecture, `leto` is the shared array substrate
between `coeus` and `apollo`, so the name matches both the repository naming
scheme and the crate responsibility.

## Workspace

| Crate | Responsibility |
| --- | --- |
| `leto` | Core const-rank layout, slicing, array, view, and storage primitives. |
| `leto-ops` | Elementwise arithmetic, reductions, matrix multiplication, SIMD hooks, and Moirai-backed parallel loops. |
| `leto-python` | Thin PyO3/NumPy boundary over Rust operations with GIL release around compute. |

## Python Releases

GitHub Releases tagged `leto-python-v<version>` build locked CPython 3.9–3.13
wheels for Linux, Windows, and macOS. The workflow installs and imports each
wheel as `leto_python`, verifies that its `leto-python` metadata version matches
the release tag, attests and attaches the exact wheel set to the GitHub Release,
then publishes those same artifacts to PyPI through OIDC Trusted Publishing.
The tag version must equal the workspace Cargo version.

## Rust Crate Releases

The `Crates.io Release` workflow validates a named workspace package on manual
dispatch. After that package's required first release is published locally and
its crates.io Trusted Publisher is registered, a GitHub Release tagged
`crate-<package>-v<version>` packages, verifies, and publishes the matching
Cargo version with a short-lived OIDC token. Validation runs in a separate
read-only job. The publish job is bound to the GitHub `crates-io` environment;
register each package's Trusted Publisher with that environment. `leto-python`
remains a wheel-only artifact and is marked `publish = false` for crates.io.

## Core API

The core type model separates layout from storage:

```rust
use leto::{Array, Layout, SliceArg, VecStorage};

let layout = Layout::c_contiguous([2, 3, 4])?;
let storage = VecStorage::new((0..24).collect::<Vec<_>>());
let array = Array::new(layout, storage)?;

let view = array.slice_with::<2>(&[
    SliceArg::Index(-1),
    SliceArg::NewAxis,
    SliceArg::range(Some(1), None, 1),
    SliceArg::Index(2),
])?;

assert_eq!(view.shape(), [1, 2]);
# Ok::<(), leto::LetoError>(())
```

### Layout Features

- C-contiguous and Fortran-contiguous layout construction.
- Const-rank shape and stride storage.
- Rank-readable aliases for `Array1`, `Array2`, `Array3`, `ArrayView1`,
  `ArrayView2`, `ArrayView3`, and mutable view variants.
- Owned-array constructors for `zeros`, `from_elem`, `from_vec`,
  `from_shape_vec`, `from_shape_fn`, `from_fn`, `eye`, iterator `collect`, and
  `into_vec`.
- `AxisIter` and `AxisIterMut` subview iteration over a selected axis.
- Named rank-2 `rows`, `columns`, `rows_mut`, and `columns_mut` helpers over
  the same zero-copy axis iterator implementation.
- Physical offset calculation.
- Zero-copy slicing, transposition, and broadcasting.
- Broadcast preserves source strides for same-shape axes and uses zero strides
  only for expanded singleton axes.
- ndarray-style slicing with:
  - full-axis selection,
  - optional signed bounds,
  - negative indices,
  - negative steps,
  - integer indexing that removes an axis,
  - inserted new axes,
  - ellipsis expansion,
  - implicit trailing full axes.

### Storage Features

- `SliceStorage<'a, T>` for borrowed read-only storage.
- `SliceStorageMut<'a, T>` for borrowed mutable storage.
- `VecStorage<T>` for owned heap-backed storage.
- `MnemosyneStorage<T>` behind `mnemosyne-alloc` for optional aligned
  allocation. `new(len)` requires `T: Default` and initializes elements before
  exposing safe slices.

### Operation Features

`leto-ops` routes elementwise arithmetic through one generic
`binary_map::<Op, T, N>` traversal. Public wrappers such as `add`, `sub`, `mul`,
and `div` are thin calls into that kernel using zero-sized operation markers
(`AddOp`, `SubOp`, `MulOp`, `DivOp`). This keeps one authoritative contiguous,
strided, SIMD, and parallel dispatch path. Inputs may broadcast to the
caller-owned output shape, so `[N, 1]` and `[1, C]` views write directly into
`[N, C]` outputs without materializing broadcasted arrays.

- Contiguous views use slice kernels on the `Scalar` trait. Native `f32` and
  `f64` implementations always route through Hermes SIMD (which itself
  runtime-dispatches AVX-512/AVX2/NEON and a scalar fallback via CPUID); a
  per-method scalar loop remains as the fallback for types Hermes does not cover
  (e.g. `f16`/`bf16`). SIMD is not a build feature — Hermes is an unconditional
  dependency, so the best available path is always selected at runtime.
- Large contiguous and strided elementwise operations use Moirai through the
  `parallel` feature after layout storage spans are validated.
- Axis reductions use caller-owned output views and keep the reduced dimension
  as length one, matching Coeus tensor semantics such as `[N, C] -> [N, 1]`.
  `sum_axis_into`, `mean_axis_into`, `min_axis_into`, and `max_axis_into` share
  one ZST-selected reduction traversal and use Moirai for large output domains.
- Allocating axis-reduction wrappers (`sum_axis`, `mean_axis`, `min_axis`, and
  `max_axis`) produce C-contiguous output by delegating to the caller-owned
  reduction core after constructing `VecStorage`.
- Unary mapping APIs provide `map_into` for caller-owned output and `mapv` /
  `map` for allocating C-contiguous output. Precision changes are explicit in
  the caller-provided closure rather than hidden in the traversal. `map_inplace`
  mutates a view in place (the `mapv_inplace` analogue).
- Named real unary math operations (`ExpOp`, `LnOp`, `SinOp`, `CosOp`,
  `SqrtOp`, `AbsOp`, `NegOp`, `RecipOp`, `PowfOp`) are ZST/value-carrying
  markers implementing the `UnaryOp` trait, routed through the same traversal
  kernel via `unary_map` / `unary_map_into`. They are bounded on `RealScalar`,
  a segregated transcendental extension of `Scalar` (native for `f32`/`f64`,
  documented `f32` fallback for `f16`/`bf16`).
- `scalar_map` / `scalar_map_into` apply an array–scalar operation reusing the
  `BinaryOp` markers (`AddOp`/`SubOp`/`MulOp`/`DivOp`); no scalar-specific
  kernel exists.
- Elementwise operators on `Array` (leto core, ADR 0004): `&a + &b`, `&a - &b`,
  `&a * &b`, `&a / &b`, `&a op scalar` (scalar bounded by `ScalarOperand`), and
  `-&a`. These are the allocating convenience tier (one shared `iter_elements`
  traversal); `binary_map`/`scalar_map` above remain the SIMD/broadcasting
  performance tier. `*` is elementwise (Hadamard), matching ndarray; matrix
  product is the `matmul` method. Unequal-shape array operators panic.
- `dot` computes a rank-1 dot product (contiguous fast path plus strided
  fallback) accumulating in native precision.
- Contiguity queries (`is_c_contiguous`, `is_f_contiguous`, `is_contiguous`)
  and memory-order slice access (`as_slice_memory_order`,
  `as_mut_slice_memory_order`) expose dense blocks of sliced or iterated
  subviews at non-zero offsets, the access pattern Apollo's in-place FFT
  butterfly kernels require. `as_slice` / `as_mut_slice` expose C-order dense
  blocks independent of offset.
- Exact chunk streaming (`exact_chunks`) yields non-overlapping zero-copy block
  views of a fixed shape, skipping remainders along each axis and preserving the
  parent strides for transposed or sliced inputs.
- Axis chunk streaming (`axis_chunks_iter`) yields non-overlapping zero-copy
  chunks along one axis, including the final remainder chunk while preserving
  the full extent and strides of every other axis.
- Logical mutable traversal is available through `iter_mut` for dense arrays
  and `indexed_iter_mut` for provably disjoint strided layouts; `index_axis` and
  `index_axis_mut` reduce rank without copying.
- Matrix multiplication lives in a dedicated matrix module, writes into
  caller-owned output, rejects zero-stride mutable output aliasing, and supports
  contiguous plus strided/transposed inputs. `batched_matmul` contracts rank-3
  `[B,M,K] x [B,K,N]` batches (batch dim broadcasts when 1) by dispatching each
  batch to the rank-2 kernel.
- Prefix/suffix scans (`scan_axis`, `cumsum`) keep shape and run along an axis
  through `CumSumOp`/`CumProdOp` markers and a `ScanDirection` (Forward/Reverse).
- Structural ops in leto core (`concat`, `pad`, `split`, `stack`) compose and
  partition arrays along an axis; `concat`/`pad`/`stack` allocate C-contiguous
  output in logical row-major order, `split` returns zero-copy subviews.
  `stack` is rank-increasing (`N→N+1`) via the `InsertAxis` compile-time rank
  helper (the dual of `RemoveAxis`), so dimension changes stay in the type
  system on stable Rust.
- Deterministic seeded random constructors (`uniform_with_seed`,
  `normal_with_seed` via Box-Muller) over an `Xorshift64` PRNG produce leto
  arrays; sampling runs in native precision.
- `zip2_mut_with` is the three-operand in-place zip (one mutable output, two
  read inputs), the `Zip::from(out).and(a).and(b)` analogue; indexed variants
  (`indexed_zip_mut_with`, `indexed_zip2_mut_with`) pass logical `[usize; N]`
  coordinates into the closure for `Zip::indexed`-style call sites.
- Strided output layouts that can alias mutable writes through zero strides do
  not enter parallel write paths.
- The core `leto` crate remains independent of Hermes and Moirai; integration
  stays in `leto-ops` so layout/storage types can compile separately.

## Current Verification

The current local gate is clean:

```sh
cargo fmt --check
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

Current value-semantic coverage includes:

- C and Fortran contiguous layouts.
- Offset calculation and out-of-bounds rejection.
- Array construction and indexing.
- Legacy same-rank slicing.
- ndarray-style reverse slicing.
- integer-index axis dropping.
- new-axis insertion.
- ellipsis and implicit trailing axes.
- named rank-2 row and column iteration.
- transposition and broadcasting.
- property tests for C/F offset formulas, transpose value preservation,
  reverse slicing, composed slicing, empty-axis storage validation,
  singleton-axis broadcasting, and negative-stride storage span validation.
- elementwise arithmetic through the shared ZST `binary_map` kernel.
- strided/transposed elementwise traversal.
- keep-dim `sum_axis_into`, `mean_axis_into`, `min_axis_into`, and
  `max_axis_into` reductions over contiguous and strided inputs.
- allocating keep-dim `sum_axis`, `mean_axis`, `min_axis`, and `max_axis`
  reductions over contiguous, strided, and empty-axis inputs.
- `map_into`, `mapv`, and `map` over contiguous and strided inputs.
- differential tests against `ndarray` for map-style contiguous/transposed
  traversal and keep-dim axis reductions.
- `sum` and 2D `matmul`, including differential matmul checks against
  `ndarray` for contiguous and transposed inputs.
- symmetric Jacobi eigendecomposition value tests (eigenvalue ordering,
  reconstruction, orthonormality, symmetry/finiteness rejection).
- PyO3 output conversion consumes owned Leto vectors into NumPy instead of
  cloning through an intermediate slice.
- PyO3 boundary tests cover value parity for `add`, `sum`, and `matmul`, shape
  mismatch rejection, and rejection of non-contiguous NumPy inputs.
- Apollo/Coeus migration fixtures cover representative `Array1`/`Array2`/
  `Array3` construction, complex precision mapping, half-pair storage,
  keep-dim reduction plus broadcasted elementwise ops, and dense-layer matmul
  shapes.

### Linear Algebra Features

- `symmetric_eigen_jacobi` and `symmetric_eigen_jacobi_with_tolerance`
  compute symmetric eigendecompositions (ascending eigenvalues, orthonormal
  column eigenvectors) via Jacobi rotations, generic over `T: RealScalar` and
  running in native precision. This closed Apollo's `nalgebra` dependency:
  FrFT/GFT eigendecomposition now runs on Leto.
- Further decompositions (LU, QR, Cholesky, SVD) are added only with a named
  consumer driver and a differential oracle; see `gap_audit.md` §B.
- A fluent rank-2 trait layer (ADR 0003) consolidates the ndarray strided-array
  and nalgebra matrix-method models onto the existing `Array2`/`ArrayView2`: any
  rank-2 receiver gains `matmul`, `norm_l1`/`norm_l2`/`norm_max`, `lu`, `qr`,
  `cholesky`, `svd`, `singular_values`, `symmetric_eigen`,
  `symmetric_eigenvalues`, `solve`, `solve_least_squares`, `inv`, and `det`
  through the `MatrixProduct`/`MatrixNorm`/`MatrixDecompose`/`MatrixSolve`
  traits. Each method is a zero-cost delegator to the free-function kernel above
  (single source of truth); the `AsMatrixView` bridge lets owned arrays,
  borrowed arrays, and strided views all carry the surface. The full ndarray
  0.16 / nalgebra 0.35 completeness program lives in `docs/completeness/`.

### Runnable Migration Evidence

`leto-ops` ships two deterministic, CI-safe examples for consumer migration
checks:

```sh
cargo run --locked -p leto-ops --example ndarray_parity
cargo run --locked -p leto-ops --example nalgebra_parity
```

`ndarray_parity` compares construction, elementwise addition, dot product,
matrix multiplication, sum, and `mapv`, reporting every absolute differential
against exact-operation or `γₙ` reduction bounds. `nalgebra_parity` compares a
manufactured Dirichlet Poisson solve through nalgebra dense LU and Leto Ops
COO→CSR plus `SparseLuSolver`; it independently checks normalized residuals and
the exact discrete sine eigenmode with condition-number-scaled bounds. These
are runnable workflow examples, not benchmarks. Controlled performance
comparisons remain in the Criterion targets.

## Replacement Status

- **nalgebra**: replaced for Apollo. Apollo removed its `nalgebra`
  dependency by migrating eigendecomposition to
  `leto_ops::symmetric_eigen_jacobi` and graph adjacency storage to
  `leto::Array2<f64>`.
- **ndarray, Apollo**: replaced. Apollo commit `324f380` uses native Leto host
  arrays across its transform families; its manifests and resolved Rust graph
  contain no `ndarray` or retired `ndarray-compat` dependency edge. Leto retains
  `ndarray` only as its own dev-dependency differential oracle.
- **Coeus backend**: CPU array layer consolidated onto Leto (verified
  2026-06-15 against coeus HEAD `037fdd5`). Coeus's CPU `BackendOps` route every
  array primitive (elementwise, matmul + batched, axis reductions,
  argmax/argmin, cumsum/suffix, concat/pad/split/stack, seeded RNG,
  to_contiguous/reshape/permute, cross-backend transfer) through the
  `coeus-leto` const-rank dispatch shim (ADR 0002) into Leto/`leto-ops` kernels,
  with cross-repo contract and per-op differential tests (coeus workspace 255
  tests green). Coeus keeps `ComputeBackend`, autodiff, NN kernels
  (conv/pool/attention), higher sparse formats/backends, and wgpu/CUDA backends;
  Leto owns CPU CSR sparse-dense parity kernels. `coeus-tensor` is the
  autodiff-integrated tensor wrapper (not duplicated layout). Consumer rev-bump
  to Leto 0.20.0 is pending the stack-wide themis-0.9 re-pin cascade
  (`gap_audit.md` §D).

The full gap analysis against `ndarray` 0.16 and `nalgebra` lives in
`gap_audit.md`; the tracked migration plan lives in `checklist.md` and
`backlog.md`.

## Dependency Policy

Core Leto crates must not depend on `ndarray` or `nalgebra` in production. Leto
packages use them only as dev-dependency differential oracles for replacement
tests and examples; no public feature, re-export, or conversion implementation
exposes either provider.
Language and FFI consumers construct native Leto arrays at their ownership
boundaries instead of routing through a provider compatibility module.

Downstream Atlas repositories consume Leto through a Git remote. Provider-side
changes must be committed and pushed before Apollo, Coeus, or other consumers
update their lockfiles.
