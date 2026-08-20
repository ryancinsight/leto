# Arithmetic and Kernel Ownership

The core `leto` crate supplies representation-preserving transformations:
`mapv` allocates a C-contiguous result, `zip_map` combines two equal-shaped
arrays, and mutable views provide caller-owned destinations. These operations
are useful for simple transformations and for defining small reference
oracles.

The numerical kernel layer is `leto-ops`. It owns the generic `Scalar` and
`RealScalar` contracts, elementwise binary and unary operations, reductions,
matrix products, scans, and linear algebra. The ownership split prevents the
storage crate from depending on a scheduling or SIMD implementation.

## One traversal, several policies

`leto-ops` routes binary arithmetic through one `binary_map` traversal selected
by a zero-sized operation marker such as addition or multiplication. The
operation is generic over `T` and rank; scalar, Hermes SIMD, and Moirai-backed
execution are policies of that traversal rather than copied algorithms.
Broadcasting is represented by the input layout. A singleton axis can be
expanded with a zero stride, so a broadcasted input does not require a
materialized repeated array.

Unary operations use the same pattern through `unary_map`. An operation such
as exponential or square root is a value-level marker implementing the unary
contract, not a type-suffixed function family. Array–scalar operations reuse
the binary markers through `scalar_map`.

## Precision and output ownership

The scalar contract preserves native precision. A caller that needs a wider
accumulator must select an API whose trait contract names that accumulator;
the generic kernel does not cast every input to a fixed type. For allocation
control, the `*_into` operations write into caller-owned views. The allocating
wrappers are convenience APIs that construct a validated C-contiguous output
and delegate to the same kernel.

Use `mapv` for a small owned transformation, `map_into` for a reusable output,
and `leto-ops` when the operation has broadcasting, SIMD, parallel, reduction,
or linear-algebra semantics.
