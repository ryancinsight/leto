# Leto: Systems-Optimized N-Dimensional Strided Arrays

Leto is a Rust workspace for N-dimensional strided array layouts, zero-copy
views, storage backends, and array operations. It is intended to replace direct
`ndarray` usage as the shared non-differentiable memory and layout vocabulary
between Atlas spectral transforms (`apollo`) and tensor/autodiff systems
(`coeus`).

## Role In Atlas

Leto sits between:

- `mnemosyne`: optional aligned allocation and memory policy.
- `moirai`: parallel scheduling for elementwise and reduction operations.
- `hermes`: SIMD-backed scalar/vector execution.
- `apollo`: spectral transforms that need shared 1D/2D/3D array views.
- `coeus`: tensor and autodiff systems that need layout-compatible storage
  without making Apollo depend on Coeus.

Leto owns layout, storage, views, slicing, and non-differentiable operations.
Coeus owns autodiff graphs, gradients, optimizers, and neural-network state.
Apollo owns Fourier, spectral, and transform kernels.

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
| `leto-python` | Thin PyO3/NumPy boundary over Rust operations. |

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
- Physical offset calculation.
- Zero-copy slicing, transposition, and broadcasting.
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
- Strided output layouts that can alias mutable writes through zero strides do
  not enter the parallel path.
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
- transposition and broadcasting.
- elementwise arithmetic through the shared ZST `binary_map` kernel.
- strided/transposed elementwise traversal.
- `sum` and 2D `matmul`.

## Replacement Status

Leto is not yet a complete `ndarray` replacement for Atlas. Before Apollo or
Coeus can remove `ndarray`, Leto still needs:

- rank aliases or constructors for `Array1`, `Array2`, `Array3`;
- `zeros`, `from_elem`, `from_vec`, `from_shape_fn`, `from_shape_vec`, and
  `into_vec` equivalents;
- row, column, and axis iteration APIs;
- axis reductions with caller-owned output;
- `map`, `map_into`, `mapv`-equivalent, and zip-map APIs;
- differential tests against `ndarray` for all Apollo-facing behavior;
- Python output conversion that avoids clone-through-`Vec` result paths.

See `checklist.md` and `backlog.md` for the tracked migration plan.

## Dependency Policy

Core Leto crates must not depend on `ndarray`. An optional compatibility or
test-only feature may be added later for differential verification and
transitional conversions.

Downstream Atlas repositories consume Leto through a Git remote. Provider-side
changes must be committed and pushed before Apollo, Coeus, or other consumers
update their lockfiles.
