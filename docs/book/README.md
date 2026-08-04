# leto — N-Dimensional Arrays for Atlas

`leto` is the host-array and CPU linear algebra layer of the Atlas compute
stack.  It provides N-dimensional strided arrays, zero-copy views, sparse
matrix formats, and reduction/statistics operations.  It replaces `ndarray`
and `nalgebra` for Atlas consumers.

## Design goals

- **Generic over element type** — `Array<T, S, N>` is parametric in the
  element type `T`, the storage `S`, and the rank `N`.  No scalar-type suffixes.
- **Zero-copy views** — `ArrayView<T, S, N>` borrows from any storage; the
  lifetime ensures the original array is not modified while the view is live.
- **Storage backends** — `VecStorage<T>` is heap-backed; `StackStorage<T, N>`
  is stack-backed for small, fixed-size arrays; `MnemosyneStorage<T>` routes
  through the Atlas allocator.
- **Rank-specific aliases** — `Array1<T>`, `Array2<T>`, `Array3<T>`,
  `Array4<T>`, `ArrayD<T>` (dynamic rank) make the most common cases concise.

## What this book covers

1. The `Array<T, S, N>` type and the `Array1`/`Array2` aliases.
2. Row-major layout, strides, and the `Layout` contract.
3. Zero-copy `ArrayView` and mutable `ArrayViewMut`.
4. Elementwise arithmetic via `mapv` and operator overloads.
5. Reductions (`sum_all`, `mean_all`) and statistics (`pearson_correlation`).
6. Structural operations: `concat`, `pad`, `split`, `stack`.
7. Storage backends and when to choose each.
8. Sparse formats: CSR, CSC, COO.
