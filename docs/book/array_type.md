# The Array Type

`Array<T, S, N>` separates three contracts that are often conflated in an
array API:

- `T` is the element type and is never silently converted by the container.
- `S` owns or borrows the element storage through the `Storage` trait.
- `N` is the const rank, so an `Array2<T>` cannot be passed where an
  `Array3<T>` is required without an explicit shape-changing operation.

The runtime shape is `[usize; N]` in the `Layout`. Construction validates that
the layout can address the supplied storage before the array becomes visible.
`Array::new` therefore returns `Result` rather than allowing an invalid offset,
stride, or storage length to enter the safe API.

## Constructing values

The rank aliases expose constructors for the common cases. `Array1::from_vec`
and `Array2::from_vec` require the supplied length to equal the requested
shape. `zeros`, `ones`, `from_elem`, and `eye` construct values with the shape
encoded by their arguments. The identity constructor is rank-2 because its
diagonal has matrix semantics.

The constructors retain the element type. An `Array1<f32>` performs its
element operations in `f32`; the container does not widen to another
precision. Generic numerical kernels are provided by `leto-ops` and express
their scalar requirements through `Scalar` or `RealScalar`.

## Reading and ownership

`Array::view` produces an `ArrayView` without copying. `as_slice` returns a
slice only when the logical elements are C-dense. `as_slice_memory_order` also
accepts a contiguous non-C view, such as a Fortran-contiguous or transposed
block. If a consumer requires row-major ownership, `to_contiguous` materializes
one `VecStorage` array explicitly.

`mapv` is the allocating elementwise transformation in the core crate. It
returns a new `Array<U, VecStorage<U>, N>` and leaves the source unchanged.
Use a view or a caller-owned output in a hot path when an allocation is not
part of the operation's contract.

## Shape changes

`reshape` returns a view when the layout can express the new shape without
moving elements. `into_shape` consumes the array and retains its storage while
changing the const rank. `slice_with`, `index_axis`, `transpose`, and
`broadcast` all change the layout and return a view. These operations are
fallible because their bounds and stride conditions are part of the layout
contract.
