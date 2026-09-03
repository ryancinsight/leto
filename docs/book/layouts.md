# Layouts and Strides

`Layout<N>` maps a logical index to a physical storage offset. For a shape
`d = [d_0, ..., d_(N-1)]`, strides `s`, and base offset `o`, the address of
`i` is

`o + i_0*s_0 + i_1*s_1 + ... + i_(N-1)*s_(N-1)`.

The mapping is valid only when every index in the shape stays within the
storage span. Leto validates this relationship at construction and whenever a
fallible transformation creates a new layout. This makes indexing through a
safe `ArrayView` a contract-preserving operation rather than a repeated proof
at each caller.

## C and Fortran order

`Layout::c_contiguous` assigns the last axis stride one and walks earlier axes
through the product of later extents. `Layout::f_contiguous` does the reverse.
The two layouts contain the same logical values but expose different physical
iteration orders. `is_c_contiguous` and `is_f_contiguous` describe canonical
order at the base offset; `is_contiguous` describes a dense block in either
order.

`as_slice` is deliberately stricter than `as_slice_memory_order`: the former
requires logical C order, while the latter permits any dense physical order.
Consumers that assume row-major order must use the former or materialize with
`to_contiguous`.

## Strided views

Slicing changes the base offset and can introduce a negative or non-unit
stride. Transpose permutes shape and stride pairs. Broadcasting expands a
singleton axis by assigning it a zero stride; it never copies the repeated
value. A zero-stride mutable output would alias one location, so Leto rejects
that layout from mutable parallel paths.

These rules explain why a view is cheap: it contains a layout and a borrow of
the original storage, not a second element buffer. They also explain why a
layout should be preserved through provider boundaries instead of flattened
into a temporary vector.

## Runtime-rank layout metadata

`LayoutDyn` stores shape and strides in boxed slices when rank is data-dependent.
Its broadcast and injectivity operations use the same shared kernels as
`Layout<N>`. A provider can therefore validate a runtime-rank zero-copy view
without creating a second layout algorithm or materializing the values.
