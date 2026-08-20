# leto

N-dimensional strided array layouts, zero-copy views, and storage backends.

`leto` is the core crate of the [Leto workspace](../../README.md): it owns the
type model that separates *layout* (shape, strides, offset) from *storage*
(where the elements live), plus the slicing, transposition, broadcasting, and
structural operations that reshape a view without touching memory. It carries
no SIMD or threading dependency; kernels live in
[`leto-ops`](../leto-ops/README.md).

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

## What is here

- Const-rank `Layout` with C- and Fortran-contiguous construction, physical
  offset calculation, and contiguity queries.
- `Array` / `ArrayView` / `ArrayViewMut` with rank-readable aliases
  (`Array1`..`Array3` and view variants).
- Slicing with signed bounds, negative indices and steps, axis-dropping integer
  indexing, inserted axes, ellipsis, and implicit trailing axes.
- Zero-copy transposition and broadcasting; broadcast preserves source strides
  for same-shape axes and uses zero strides only for expanded singleton axes.
- Storage backends: `VecStorage` (owned), `SliceStorage` / `SliceStorageMut`
  (borrowed), and `MnemosyneStorage` behind the `mnemosyne-alloc` feature for
  aligned allocation.
- Axis and chunk iteration (`AxisIter`, `exact_chunks`, `axis_chunks_iter`) and
  structural ops (`concat`, `pad`, `split`, `stack`).
- Elementwise operators on `Array` (`&a + &b`, `&a * scalar`, `-&a`) as the
  allocating convenience tier.

## Documentation

API docs: <https://docs.rs/leto>

The [Leto domain book](../../docs/book/README.md) explains the layout,
storage, view, structural, and sparse contracts with executable examples.

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
