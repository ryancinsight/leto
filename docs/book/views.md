# Zero-Copy Views

An `ArrayView<'a, T, N>` is a read-only window over storage that lives for
`'a`. It carries the same layout information as an owned array, so a view can
be sliced, transposed, reshaped, broadcast, iterated, or converted back to a
borrowed array without copying elements.

## Borrowing is the boundary

`array.view()` borrows the source. While that view exists, Rust prevents a
mutable borrow of the source. `array.view_mut()` requires exclusive access and
returns `ArrayViewMut`; the lifetime prevents the mutable view from outliving
that access. The safety contract is therefore expressed in the type system,
not in a caller convention such as “do not mutate while this view is used.”

`get` and `get_mut` return `Result` because the logical index must be checked
against the layout. `iter`, `indexed_iter`, `axis_iter`, `lanes`, and the chunk
iterators preserve the logical shape while allowing callers to choose a
traversal that matches the kernel.

## Transforming a view

`slice_with` accepts signed ranges, negative indices, negative steps, inserted
axes, and ellipsis expansion. `transpose` only permutes metadata. `reshape`
requires a compatible stride pattern. `broadcast` can add dimensions or
expand singleton axes and introduces zero strides for the expanded axes.

`ArrayViewMut` provides the corresponding mutable operations. It can be
materialized with `to_contiguous` when a downstream routine needs an owned
C-order buffer. That copy is explicit and observable; ordinary slicing and
transposition remain zero-copy.

## Iteration and disjoint mutation

Use `indexed_iter_mut` when a strided layout has to be mutated through logical
coordinates. `task_partitions_mut` splits a mutable view into disjoint logical
regions for task-based execution. The implementation rejects layouts whose
zero strides could make two logical outputs alias. This is the boundary that
lets `leto-ops` and Moirai parallelize real work without weakening aliasing
guarantees.
