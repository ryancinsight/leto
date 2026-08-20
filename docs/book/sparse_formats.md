# Sparse Formats

Leto represents sparse matrices as typed storage formats rather than dense
arrays containing mostly zero values. `CooArray`, `CsrArray`, and `CscArray`
all preserve the matrix dimensions and numeric element type.

## COO: construction format

COO stores `(row, column, value)` triplets. It is convenient while assembling
a matrix because entries can be collected from an iterator and sorted once.
`sort_by_row_column` establishes the order required by CSR conversion. COO is
usually an assembly representation, not the best traversal format for a
repeated row or column kernel.

## CSR and CSC: traversal formats

CSR stores one row pointer range per row plus column indices and values. A row
lookup is a bounded slice and `row_entries` returns the row's `(column, value)`
pairs without allocating. CSC is the transposed analogue: `col_entries`
provides column traversal through column pointers and row indices.

`CooArray::to_csr` sorts a clone and builds normalized row storage.
`CsrArray::to_csc` and the reverse conversion pass through COO semantics so
the dimension and coordinate interpretation remain explicit. Conversion is a
deliberate materialization; row and column iteration on an existing compressed
format is zero-copy.

## Numerical ownership

The core crate owns representation and format conversion. Sparse matrix-vector
products, sparse factorizations, and iterative solver preconditioners are
`leto-ops` responsibilities. This keeps sparse storage usable by consumers
that need layout and serialization without pulling in solver or execution
policy dependencies.
