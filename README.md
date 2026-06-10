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
elementwise math, reductions, matmul, shape ops, and dense linear algebra
(currently the symmetric Jacobi eigensolver). Coeus owns autodiff graphs,
NN kernels (conv/pool/attention), optimizers, sparse formats, and GPU
backends behind its `ComputeBackend` trait. Apollo owns Fourier, spectral,
and transform kernels. `themis` and `melinoe` are consumed indirectly via
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
  `from_shape_vec`, `from_shape_fn`, and `into_vec`.
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
strided, SIMD, and parallel dispatch path.

- Contiguous views use slice kernels on the `Scalar` trait. Native `f32` and
  `f64` implementations call Hermes SIMD when the `simd` feature is enabled
  and fall back to scalar loops when Hermes cannot handle the slice.
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
  the caller-provided closure rather than hidden in the traversal.
- Matrix multiplication lives in a dedicated matrix module, writes into
  caller-owned output, rejects zero-stride mutable output aliasing, and supports
  contiguous plus strided/transposed inputs.
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
  column eigenvectors) via Jacobi rotations. This closed Apollo's `nalgebra`
  dependency: FrFT/GFT eigendecomposition now runs on Leto.
- Further decompositions (LU, QR, Cholesky, SVD) are added only with a named
  consumer driver and a differential oracle; see `gap_audit.md` §B.

## Replacement Status

- **nalgebra**: replaced for Apollo. Apollo removed its `nalgebra`
  dependency by migrating eigendecomposition to
  `leto_ops::symmetric_eigen_jacobi` and graph adjacency storage to
  `leto::Array2<f64>`.
- **ndarray, Apollo**: partial. Apollo pins Leto as a Git dependency and
  exposes `forward_leto`/`inverse_leto` boundaries on FFT, CZT, DHT, NUFFT,
  SHT, Radon, and STFT; `ndarray` remains Apollo's internal CPU compute
  substrate and differential oracle. The named blocker for hot-kernel
  migration is contiguous-slice access on Leto views
  (`as_slice`/`as_slice_mut` with memory-order guarantees).
- **Coeus backend**: not started. Coeus currently carries its own
  layout/storage/traversal stack (`coeus-tensor`, `coeus-core`) duplicating
  Leto's layer over the same Mnemosyne/Moirai substrate. The plan of record
  consolidates that non-differentiable layer into Leto while Coeus keeps
  `ComputeBackend`, autodiff, NN kernels, and GPU backends. Blocking gaps:
  broadcast-aware binary ops into caller-owned output layouts, a named
  unary math-op suite, reshape/permute/to_contiguous, concat/pad/split,
  batched matmul, seeded RNG fill, and the const-rank vs dynamic-rank
  boundary decision.

The full gap analysis against `ndarray` 0.16 and `nalgebra` lives in
`gap_audit.md`; the tracked migration plan lives in `checklist.md` and
`backlog.md`.

## Dependency Policy

Core Leto crates must not depend on `ndarray`. `leto-ops` uses `ndarray` only as
a dev-dependency differential oracle for replacement tests; production features
must remain independent of `ndarray`.

Downstream Atlas repositories consume Leto through a Git remote. Provider-side
changes must be committed and pushed before Apollo, Coeus, or other consumers
update their lockfiles.
