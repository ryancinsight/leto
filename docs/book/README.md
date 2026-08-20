# Leto: N-Dimensional Arrays for Atlas

Leto is the host-array substrate of the Atlas compute stack. It provides
N-dimensional strided arrays, zero-copy views, storage backends, sparse matrix
formats, and the layout vocabulary shared by CPU consumers. The companion
`leto-ops` crate owns numerical kernels such as elementwise arithmetic,
reductions, matrix products, and dense and sparse linear algebra.

## Design goals

- **Generic representation** — `Array<T, S, N>` is parameterized by element,
  storage, and const rank. The rank is part of the type, while runtime shape
  values remain data.
- **Zero-copy views** — `ArrayView` and `ArrayViewMut` borrow the same storage
  with lifetimes and exclusive mutable access enforcing the aliasing contract.
- **Explicit storage** — `VecStorage`, `StackStorage`, borrowed slice storage,
  and optional `MnemosyneStorage` make ownership and allocation policy visible.
- **Rank-specific aliases** — `Array1`, `Array2`, `Array3`, `Array4`, and
  `ArrayD` keep common call sites concise without duplicating implementations.

## What this book covers

1. The `Array<T, S, N>` type and rank aliases.
2. Row-major layout, strides, offsets, and validation.
3. Borrowed views, slicing, broadcasting, and mutable access.
4. Elementwise mapping in `leto` and provider-owned kernels in `leto-ops`.
5. Reductions and statistics with explicit empty-input behavior.
6. Structural operations: `concat`, `pad`, `split`, and `stack`.
7. Storage selection and ownership boundaries.
8. COO, CSR, and CSC sparse formats.
9. Leto's position between Atlas tensor, transform, memory, and accelerator
   providers.

The examples are intentionally small but executable. They construct arrays,
perform input-sensitive operations, and assert the values described in the
text. The Pages workflow runs them through Cargo package staging so the book
does not rely on a developer's pre-existing target directory.
