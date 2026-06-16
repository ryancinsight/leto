# Oracle API Inventory: ndarray 0.16 & Companion Crates

This document catalogs the public API surface of `ndarray 0.16` and its key companion crates (`ndarray-rand`, `ndarray-stats`, `ndarray-linalg`), serving as the Stage 1 oracle reference for Leto.

## 1. Array Representation and Typenames
- `ArrayBase<S, D>`: Core array structure parameterized by storage `S` and dimension `D`.
- `Array<A, D>`: Owned array type alias (backed by `OwnedRepr<A>`).
- `ArrayView<A, D>`: Shared read-only view alias (backed by `ViewRepr<&A>`).
- `ArrayViewMut<A, D>`: Exclusive mutable view alias (backed by `ViewRepr<&mut A>`).
- `ArcArray<A, D>`: Shared reference-counted array alias (backed by `OwnedArcRepr<A>`).
- `COwnedArray<A, D>`: Alias for clone-on-write arrays.
- `Dim<Ix>`: Static rank representation (e.g., `Ix1`, `Ix2`, `Ix3`, `IxDyn`).
- `IxDyn`: Dynamic rank representation.

## 2. Constructors
- `ArrayBase::zeros(shape)`: Allocate C-contiguous array filled with zero.
- `ArrayBase::ones(shape)`: Allocate C-contiguous array filled with one.
- `ArrayBase::from_elem(shape, elem)`: Allocate array filled with a clone of `elem`.
- `ArrayBase::from_shape_vec(shape, vec)`: Construct array from flat 1D vector (C-contiguous).
- `ArrayBase::from_shape_vec_unchecked(shape, vec)`: Construct array without layout validation.
- `ArrayBase::from_shape_fn(shape, f)`: Construct array by calling coordinate generator closure.
- `ArrayBase::from_vec(vec)`: Construct 1D array from vector.
- `ArrayBase::linspace(start, end, n)`: Construct 1D array with linearly spaced values.
- `ArrayBase::logspace(base, start, end, n)`: Construct 1D array with logarithmically spaced values.
- `ArrayBase::range(start, end, step)`: Construct 1D array with stepped range.
- `ArrayBase::eye(n)`: Construct 2D identity matrix.
- `ArrayBase::from_iter(iter)`: Construct 1D array from iterator.

## 3. Properties and Accessors
- `.ndim()`: Return number of dimensions.
- `.shape()`: Return shape as slice of `usize`.
- `.strides()`: Return strides as slice of `isize`.
- `.len()`: Return total element count.
- `.is_empty()`: Return true if element count is zero.
- `.raw_dim()`: Return raw dimension object.
- `.dim()`: Return shape as tuple/struct.
- `.as_slice()`: Borrow contiguous elements as standard slice.
- `.as_slice_memory_order()`: Borrow elements as slice if contiguous in any layout.
- `.as_ptr()` / `.as_mut_ptr()`: Return raw pointers to first element.

## 4. Indexing and Slicing
- `Index` and `IndexMut` traits: `array[index]` (returns element, supports static tuples and arrays).
- `.uget(index)` / `.uget_mut(index)`: Unsafe element retrieval skipping bounds checks.
- `s![]` macro: Syntax helper for slicing arguments.
- `.slice(info)` / `.slice_mut(info)`: Return sub-view by slicing along axes.
- `.slice_move(info)`: Consume array and return sliced view.
- `.slice_inplace(info)`: Modify view boundaries in-place.
- `.multi_slice_mut(...)`: Return multiple disjoint mutable views.

## 5. View Conversions & Transformations
- `.view()` / `.view_mut()`: Return shared / mutable view.
- `.to_owned()`: Clone view elements into an owned `Array`.
- `.into_owned()`: Convert `COwnedArray` or owned `ArrayBase` to owned `Array`.
- `.t()` / `.transpose()`: Return transposed view (reverses axes).
- `.permuted_axes(axes)`: Permute dimensions according to axis indices.
- `.into_shape(shape)`: Zero-copy reshape array if contiguous, returns error if layout copy needed.
- `.into_shape_with_order(shape)`: Reshape supporting C or F layout conversions.
- `.into_dyn()`: Convert const-rank array to dynamic rank `ArrayD`.
- `.into_dimensionality::<N>()`: Downcast dynamic rank array to static rank `N`.

## 6. Iterators and Traversals
- `.iter()` / `.iter_mut()`: Iterate over elements in logical/strided order.
- `.indexed_iter()` / `.indexed_iter_mut()`: Yield `(index, &element)` pairs.
- `.axis_iter(axis)` / `.axis_iter_mut(axis)`: Iterate over sub-views along an axis.
- `.lanes(axis)` / `.lanes_mut(axis)`: Iterate over 1D lane views along an axis.
- `.rows()` / `.rows_mut()`: Iterate over row views.
- `.columns()` / `.columns_mut()`: Iterate over column views.
- `IntoIterator` implementations for owned arrays and views.

## 7. Arithmetic and Operators
- `Add`, `Sub`, `Mul`, `Div`, `Rem` implementations for:
  - `ArrayBase op ArrayBase` (elementwise)
  - `ArrayBase op Scalar`
  - `Scalar op ArrayBase`
  - Reference and mutable variants of the above.
- `Neg` trait for signed element negation.
- `.map(f)`: Return new array with mapped elements.
- `.mapv(f)`: Version of `map` optimized for element values.
- `.map_inplace(f)` / `.mapv_inplace(f)`: Mutate elements in place.
- `Zip`: Coordinate multi-array lock-step iteration supporting:
  - `Zip::from(a).and(b).map_collect(...)`
  - `Zip::from(a).for_each(...)`
  - `Zip::indexed(...)` for index-aware loops.

## 8. Reductions
- `.sum()` / `.product()`: Accumulate all elements.
- `.sum_axis(axis)`: Reduce dimension along axis by summation.
- `.mean()` / `.mean_axis(axis)`: Compute arithmetic mean.
- `.std(ddof)` / `.var(ddof)`: Compute standard deviation and variance.

## 9. Structural Operations
- `concatenate(axis, views)`: Join arrays along a given axis.
- `stack(axis, views)`: Stack arrays along a new axis.
- `.split_at(axis, index)`: Split array into two views along an axis.

## 10. Companion Crate Surface
### ndarray-rand
- `ArrayBase::random(shape, distribution)`: Generate array with random values.
- `ArrayBase::random_using(shape, distribution, rng)`: Seeded generation.

### ndarray-stats
- `.quantile_axis_mut(axis, q, interpolator)`: Compute quantile along axis.
- `.median_axis_mut(axis)`: Compute median along axis.
- `.argmin()` / `.argmax()`: Find indices of minimum/maximum elements.
- `.argmin_skipnan()` / `.argmax_skipnan()`: Handle NaNs in argmin/argmax.

### ndarray-linalg
- `.inv()`: Inverse of square matrix.
- `.det()`: Determinant.
- `.solve(b)` / `.solve_mut(b)`: Solve linear system.
- `.qr()`: QR decomposition.
- `.cholesky()`: Cholesky decomposition.
- `.eig()` / `.eigh()`: Eigenvalue decomposition.
- `.svd()`: Singular value decomposition.
